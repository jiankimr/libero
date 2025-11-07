import collections
import dataclasses
import logging
import math
import pathlib
from typing import Optional
import datetime

import imageio
import tqdm
import tyro
import csv
import numpy as np

from PIL import Image

from service import ExternalRobotInferenceClient

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from analyze.basic_metric import (
    compute_basic_metrics, 
    infer_control_dt as infer_dt_from_env
)
from analyze.extra_metric import (
    compute_extra_metrics,
    save_metrics_to_csv,
    save_summary_csv
)
from robosuite.utils.buffers import DeltaBuffer

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "node7"
    port: int = 5555
    resize_size: int = 224
    replan_steps: int = 16
    #################################################################################################################
    # Action noise parameters
    #################################################################################################################
    action_noise_scale: float = 0.0  # Scale of alternating noise to add to the action chunk
    action_noise_dim: Optional[str] = None # Dimension to add noise to. e.g. "state.x"
    
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_10"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 20    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: Optional[str] = None  # Path to save videos
    result_file_name: Optional[str] = None  # Path to save result """
    #video_out_path: str = "./video_libero_10_1105"  # Path to save videos
    #result_file_name: str = "result_libero_10_1105.csv"  # Path to save result

    seed: int = 42  # Random Seed (for reproducibility)
    #################################################################################################################
    # Debugging
    #################################################################################################################
    debug_action: bool = False  # Enable detailed logging of action chunks and state
    #################################################################################################################
    # Save physical quantities
    #################################################################################################################
    save_metrics: bool = True  # Save all physical metrics (torque, velocity, acceleration, jerk, energy, life_ratio)


def unchunk(action_chunk, action_plan, replan_steps, debug=False):
    """
    Convert action chunk (dict of arrays) into individual timesteps.
    Handles mixed dimensionalities: (T, D) and (T,) arrays.
    
    Args:
        action_chunk: dict with keys -> arrays of shape (T, ...) or (T,)
        action_plan: deque to append actions to
        replan_steps: number of steps to extract
        debug: if True, log detailed information about action processing
    """
    lengths = [v.shape[0] for v in action_chunk.values()]
    if len(set(lengths)) != 1:
        raise ValueError(f"Inconsistent lengths: {lengths}")
    assert (
        lengths[0] >= replan_steps
    ), f"We want to replan every {replan_steps} steps, but policy only predicts {lengths[0]} steps."
    
    T = replan_steps
    keys = list(action_chunk.keys())
    
    if debug:
        logging.info(f"Unchunking: T={T}, keys={keys}")
        logging.info(f"Action shapes: {[(k, action_chunk[k].shape) for k in keys]}")
    
    for t in range(T):
        action_list = []
        for k in keys:
            val = action_chunk[k][t]
            # Ensure val is at least 1D
            if np.isscalar(val):
                val = np.array([val], dtype=np.float32)
            elif val.ndim == 0:
                val = np.array([val.item()], dtype=np.float32)
            # val is now at least 1D
            action_list.append(val)
        
        # Flatten all to 1D and concatenate
        try:
            flattened = [np.atleast_1d(a).flatten() for a in action_list]
            new_arr = np.concatenate(flattened, axis=0)
            action_plan.append(new_arr)
        except Exception as e:
            logging.error(f"Error at timestep {t}: action_list shapes = {[a.shape for a in action_list]}")
            raise e


def add_alternating_noise(
    action_chunk: dict, noise_scale: float, noise_dim: Optional[str], replan_steps: int, debug: bool = False
) -> dict:
    """Adds alternating noise to a specific dimension of the action chunk.
    
    For multi-dimensional arrays, supports index notation:
    - "action.eef_pos_delta[0]" applies noise only to x (index 0)
    - "action.eef_rot_delta[1]" applies noise only to pitch (index 1)
    - "action.gripper_close[0]" applies noise only to finger 1 (index 0)
    
    Args:
        action_chunk: dict with action arrays
        noise_scale: magnitude of noise
        noise_dim: dimension to add noise to (with optional index notation)
        replan_steps: number of steps to add noise to
        debug: if True, log noise application details
    """
    # Always create a copy to ensure consistent data structure
    noisy_action_chunk = {key: np.copy(value) for key, value in action_chunk.items()}
    
    if noise_scale == 0.0 or noise_dim is None:
        return noisy_action_chunk

    # Check if noise_dim uses index notation (e.g., "action.eef_pos_delta[0]")
    index = None
    actual_key = noise_dim
    
    if '[' in noise_dim and ']' in noise_dim:
        # Parse index notation
        try:
            actual_key = noise_dim[:noise_dim.index('[')]
            index_str = noise_dim[noise_dim.index('[') + 1:noise_dim.index(']')]
            index = int(index_str)
        except (ValueError, IndexError) as e:
            logging.error(f"Invalid index notation in '{noise_dim}': {e}")
            return noisy_action_chunk

    if actual_key not in noisy_action_chunk:
        logging.warning(
            f"Noise dimension '{actual_key}' not in action chunk keys: {list(noisy_action_chunk.keys())}. No noise added."
        )
        return noisy_action_chunk

    values = noisy_action_chunk[actual_key]
    num_steps_to_noise = min(values.shape[0], replan_steps)

    # Log original values (only if debug enabled)
    if debug:
        if index is not None:
            logging.info(f"Adding noise to {actual_key}[{index}] (scale={noise_scale})")
            logging.info(f"Original values (first {num_steps_to_noise} steps): {values[:num_steps_to_noise, index]}")
        else:
            logging.info(f"Adding noise to {actual_key} (scale={noise_scale})")
            logging.info(f"Original values (first {num_steps_to_noise} steps): {values[:num_steps_to_noise]}")

    # Create an alternating pattern of [1, -1, 1, -1, ...]
    #alternating_pattern = np.array([1 if i % 2 == 0 else -1 for i in range(num_steps_to_noise)])
    
    # Create an alternating pattern of [1, 1, -1, -1, 1, 1, -1, -1, ...] (++ -- ++ --)
    alternating_pattern = np.array([1 if (i // 2) % 2 == 0 else -1 for i in range(num_steps_to_noise)])


    # Scale the pattern by the noise_scale
    noise = alternating_pattern * noise_scale
    
    if debug:
        logging.info(f"Noise pattern: {noise}")

    # Safely add the noise to avoid potential array corruption
    try:
        # Create a new modified array instead of in-place modification
        modified_values = values.copy()
        
        if index is not None:
            # Apply noise to specific index
            modified_values[:num_steps_to_noise, index] = modified_values[:num_steps_to_noise, index] + noise
            
            if debug:
                logging.info(f"Values after adding noise: {modified_values[:num_steps_to_noise, index]}")
                # Clip values to valid ranges to prevent simulation instability
                if "gripper" in actual_key:
                    logging.info(f"Clipping gripper to [0, 1] range")
                else:
                    logging.info(f"Clipping {actual_key}[{index}] to [-1, 1] range")
            
            if "gripper" in actual_key:
                clipped_values = np.clip(modified_values, 0.0, 1.0)
            else:
                clipped_values = np.clip(modified_values, -1.0, 1.0)
        else:
            # Apply noise to entire array
            modified_values[:num_steps_to_noise] = modified_values[:num_steps_to_noise] + noise
            
            if debug:
                logging.info(f"Values after adding noise: {modified_values[:num_steps_to_noise]}")
                # Clip values to valid ranges to prevent simulation instability
                if "gripper" in actual_key:
                    logging.info(f"Clipping gripper to [0, 1] range")
                else:
                    logging.info(f"Clipping {actual_key} to [-1, 1] range")
            
            if "gripper" in actual_key:
                clipped_values = np.clip(modified_values, 0.0, 1.0)
            else:
                clipped_values = np.clip(modified_values, -1.0, 1.0)
        
        if debug:
            if index is not None:
                logging.info(f"Final clipped values: {clipped_values[:num_steps_to_noise, index]}")
            else:
                logging.info(f"Final clipped values: {clipped_values[:num_steps_to_noise]}")

        noisy_action_chunk[actual_key] = clipped_values
    except Exception as e:
        logging.error(f"Error adding noise to dimension {noise_dim}: {e}")
        return noisy_action_chunk

    return noisy_action_chunk


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # create paths for video and result with datetime
    if args.video_out_path is None or args.result_file_name is None:
        date_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        noise_str = f"noise_{args.action_noise_scale:.4f}".replace('.', '')
        dim_str = f"_dim_{args.action_noise_dim}" if args.action_noise_dim else ""
        base_name = f"{date_time_str}_{noise_str}{dim_str}"
        
        if args.video_out_path is None:
            args.video_out_path = f"./videos/video_{args.task_suite_name}_{base_name}"
        if args.result_file_name is None:
            args.result_file_name = f"./results/result_{args.task_suite_name}_{base_name}.csv"
    else:
        # Extract base_name from existing paths
        base_name = pathlib.Path(args.video_out_path).name.replace(f"video_{args.task_suite_name}_", "")

    # Create analysis output path
    analysis_out_path = f"./analysis/analysis_{args.task_suite_name}_{base_name}"

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(pathlib.Path(args.result_file_name).parent).mkdir(parents=True, exist_ok=True)
    pathlib.Path(analysis_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    
    policy = ExternalRobotInferenceClient(host=args.host, port=args.port)

    # store task_description, success rate, episode count in a csv file
    result = dict()
    
    # Collect all episode metrics for summary CSV (if save_metrics enabled)
    all_episode_metrics = []

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            
            # Infer dt from environment and initialize metrics if needed
            dt = None
            recent_joint_acc = None
            recent_torques = None
            recent_velocities = None
            torque_list, velocity_list, acceleration_list = [], [], []
            kinematic_jerk_delta_list, actuator_jerk_delta_list = [], []
            electric_energy_list = []
            
            if args.save_metrics:
                dt = infer_dt_from_env(env)
                logging.info(f"Control dt = {dt:.6f}s (≈ {1.0/dt:.2f} Hz)")
                
                # Initialize DeltaBuffers and metric lists
                # Note: joint_indexes is available after setup_references() is called during reset
                n_joints = len(env.env.robots[0].joint_indexes)
                recent_joint_acc = DeltaBuffer(dim=n_joints)
                recent_torques = DeltaBuffer(dim=n_joints)
                recent_velocities = DeltaBuffer(dim=n_joints)
                
                # Separate lists for torque (for life ratio calculation, we need current torque, not average)
                torque_avg_list = []  # For energy calculation (average of prev and current)
                torque_current_list = []  # For life ratio calculation (current step only)

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        eef_pos = obs["robot0_eef_pos"]  # Shape (3,)
                        axis_angle = _quat2axisangle(obs["robot0_eef_quat"])  # Shape (3,) - euler angles (roll, pitch, yaw)
                        gripper_qpos = obs["robot0_gripper_qpos"]  # Shape (2,) - both fingers

                        # Log input state for debugging
                        if args.debug_action:
                            logging.info(f"EEF pos: {eef_pos}, axis_angle: {axis_angle}, gripper: {gripper_qpos}")

                        # For debugging, we can also expand to individual states
                        element = {
                            # Video keys matching model metadata
                            "video.front_view": np.expand_dims(img, axis=0),  # Batch of 1: (1, H, W, C)
                            "video.left_wrist_view": np.expand_dims(wrist_img, axis=0),  # Batch of 1: (1, H, W, C)
                            # State keys matching model metadata (shape from metadata.json)
                            "state.eef_pos_absolute": eef_pos[np.newaxis, :],  # (1, 3) - absolute position
                            "state.eef_rot_absolute": axis_angle[np.newaxis, :],  # (1, 3) - euler angles (roll, pitch, yaw)
                            "state.gripper_close": gripper_qpos[np.newaxis, :],  # (1, 2) - both gripper fingers
                            "annotation.human.action.task_description": [str(task_description)],
                        }
                        
                        # Note: Action chunk structure from model:
                        # - action.eef_pos_delta: (T, 3) [x, y, z]
                        # - action.eef_rot_delta: (T, 3) [roll, pitch, yaw]
                        # - action.gripper_close: (T, 2) [finger1, finger2]
                        # To apply noise to individual elements, use:
                        #   --action_noise_dim "action.eef_pos_delta" with index notation (see add_alternating_noise_indexed function)
                        
                        if args.debug_action:
                            logging.info(f"Element keys: {element.keys()}")

                        # Query model to get action
                        action_chunk = policy.get_action(element)
                        
                        # Log action chunk for debugging (only if debug_action is enabled)
                        if args.debug_action:
                            logging.info(f"Action chunk keys: {action_chunk.keys()}")
                            for key, val in action_chunk.items():
                                if val.ndim == 1:
                                    logging.warning(f"  {key}: shape={val.shape} (1D) - may cause stacking issues!")
                                else:
                                    logging.info(f"  {key}: shape={val.shape}, dtype={val.dtype}, sample values={val[0] if len(val) > 0 else 'empty'}")
                        
                        # Add alternating noise if specified
                        if args.action_noise_scale > 0:
                            action_chunk = add_alternating_noise(
                                action_chunk, args.action_noise_scale, args.action_noise_dim, args.replan_steps,
                                debug=args.debug_action
                            )
                        
                        unchunk(action_chunk, action_plan, args.replan_steps, debug=args.debug_action)

                    action = action_plan.popleft()
                    # action from policy has shape [7]: [eef_pos_delta (3), eef_rot_delta (3), gripper_close (1)]
                    # unchunk stacks them in order of action_keys from data_config
                    # action_keys = ["action.eef_pos_delta", "action.eef_rot_delta", "action.gripper_close"]
                    
                    if args.debug_action:
                        logging.info(f"Action from plan: shape={action.shape}, dtype={action.dtype}, values={action}")
                    
                    eef_pos_delta = action[:3]      # First 3 elements: [dx, dy, dz]
                    eef_rot_delta = action[3:6]     # Next 3 elements: [roll_delta, pitch_delta, yaw_delta]
                    gripper_cmd = action[6]         # Last element: gripper command ([-1, 1] range)
                    
                    if args.debug_action:
                        logging.info(f"  eef_pos_delta={eef_pos_delta}, eef_rot_delta={eef_rot_delta}, gripper_cmd={gripper_cmd}")
                    
                    # Libero expects: [dx, dy, dz, rot_x, rot_y, rot_z, gripper]
                    # where gripper: -1 = open, 1 = close (model outputs already in this range)
                    libero_action = np.concatenate([
                        eef_pos_delta,
                        eef_rot_delta,
                        [gripper_cmd]
                    ])
                    
                    if args.debug_action:
                        logging.info(f"Libero action: {libero_action}")
                    obs, reward, done, info = env.step(libero_action.tolist())
                    
                    # Collect basic metrics (only if save_metrics is enabled)
                    if args.save_metrics:
                        # Torque - two versions:
                        # 1) Average (for energy calculation): (prev + current) / 2
                        # 2) Current (for life ratio calculation): current step only
                        torque_avg = np.array(env.env.robots[0].recent_torques.average, dtype=float)
                        torque_current = np.array(env.env.robots[0].recent_torques.current, dtype=float)
                        
                        torque_avg_list.append(torque_avg)
                        torque_current_list.append(torque_current)
                        recent_torques.push(torque_current)
                        
                        if args.debug_action:
                            logging.debug(f"Torque - avg: {torque_avg}, current: {torque_current}, diff: {torque_current - torque_avg}")
                        
                        # Velocity
                        velocity = np.array(env.env.robots[0]._joint_velocities, dtype=float)
                        velocity_list.append(velocity)
                        recent_velocities.push(velocity)
                        
                        # Acceleration
                        acc = np.array(env.sim.data.qacc[env.env.robots[0]._ref_joint_vel_indexes], dtype=float)
                        acceleration_list.append(acc)
                        recent_joint_acc.push(acc)
                        
                        # Kinematic Jerk (DeltaBuffer delta method)
                        kinematic_jerk_delta = recent_joint_acc.delta / dt
                        kinematic_jerk_delta_list.append(kinematic_jerk_delta)
                        
                        # Actuator Jerk (DeltaBuffer delta method)
                        actuator_jerk_delta = recent_torques.delta / dt
                        actuator_jerk_delta_list.append(actuator_jerk_delta)
                        
                        # Electric Energy (부호 유지)
                        t_avg = recent_torques.average  # (torque_prev + torque_curr) / 2
                        v_avg = recent_velocities.average  # (vel_prev + vel_curr) / 2
                        P = np.sum(t_avg * v_avg)  # Power [W]
                        E = P * dt  # Energy [J]
                        electric_energy_list.append(E)
                    
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
            
            # Compute and save physical metrics if enabled
            if args.save_metrics:
                try:
                    # Save basic metrics (npy files)
                    compute_basic_metrics(
                        torque_list=torque_avg_list if args.save_metrics else [],  # Use average for energy calculation
                        velocity_list=velocity_list,
                        acceleration_list=acceleration_list,
                        kinematic_jerk_delta_list=kinematic_jerk_delta_list,
                        actuator_jerk_delta_list=actuator_jerk_delta_list,
                        electric_energy_list=electric_energy_list,
                        dt=dt,
                        output_dir=analysis_out_path,
                        task_segment=task_segment,
                        episode_idx=episode_idx,
                        debug=args.debug_action,
                        suffix=suffix
                    )
                    
                    # Also save current torque for life ratio calculation
                    if args.save_metrics and len(torque_current_list) > 0:
                        torque_current_array = np.array(torque_current_list)
                        pathlib.Path(analysis_out_path).mkdir(parents=True, exist_ok=True)
                        np.save(
                            pathlib.Path(analysis_out_path) / f"torque_current_{task_segment}_{episode_idx}_{suffix}.npy",
                            torque_current_array
                        )
                        logging.info(f"Saved torque_current: shape {torque_current_array.shape}")
                    logging.info(f"Basic metrics saved for episode {episode_idx}")
                    
                    # Compute and save extra metrics (csv)
                    extra_metrics = compute_extra_metrics(
                        npy_dir=analysis_out_path,
                        task_segment=task_segment,
                        episode_idx=episode_idx,
                        suffix=suffix,
                        output_dir=analysis_out_path
                    )
                    
                    csv_path = save_metrics_to_csv(
                        all_metrics=extra_metrics,
                        task_segment=task_segment,
                        episode_idx=episode_idx,
                        suffix=suffix,
                        output_dir=analysis_out_path,
                        task_description=task_description
                    )
                    logging.info(f"Extra metrics saved to {csv_path}")
                    
                    # Collect metrics for summary CSV
                    episode_metric_dict = {
                        "task": task_description,
                        "episode": episode_idx,
                        "success": 1 if done else 0,
                        "dt": dt,
                        "metrics": extra_metrics
                    }
                    all_episode_metrics.append(episode_metric_dict)
                    
                except Exception as e:
                    logging.error(f"Failed to compute metrics: {e}")

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
        
        env.close() # Explicitly close the environment after all episodes for this task are done.

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

        result[task_description] = {
            "num_success": task_successes,
            "success_rate": float(task_successes) / float(task_episodes),
            "num_episodes": task_episodes
        }

    logging.info(f"{args.video_out_path}")
    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")

    # Save summary metrics CSV if metrics were collected
    if args.save_metrics and all_episode_metrics:
        summary_csv_file = pathlib.Path(analysis_out_path).parent / f"analysis_summary_{args.task_suite_name}_{base_name}.csv"
        save_summary_csv(
            all_episode_metrics=all_episode_metrics,
            output_file=str(summary_csv_file),
            task_suite_name=args.task_suite_name,
            base_name=base_name
        )
    
    # save result to csv
    with open(args.result_file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_description", "num_episodes", "num_success", "success_rate"])
        for task_description, data in result.items():
            writer.writerow([task_description, data["num_episodes"], data["num_success"], data["success_rate"]])


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
