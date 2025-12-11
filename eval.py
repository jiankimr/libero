import collections
import dataclasses
import logging
import math
import pathlib
import sys
from typing import Optional, List, Dict, Any
import datetime
import json
import os

import h5py
import imageio
import tqdm
import tyro
import csv
import numpy as np

from PIL import Image
#import robosuite.utils.transform_utils as T

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
    replan_steps: int = 16 #total 16, base 4~5
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
    output_suffix: Optional[str] = None  # Suffix to append to output file names (e.g., "clean", "noisy")
    #################################################################################################################
    # Rollout data collection (for creating training datasets)
    #################################################################################################################
    save_rollouts: bool = True  # Save rollout data as HDF5 for training dataset creation
    rollout_save_path: Optional[str] = None  # Directory to save rollout HDF5 files (default: ./rollouts/)


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
            # Handle different array shapes: (T,), (T, 1), (T, D)
            if modified_values.ndim == 1:
                # Shape (T,) - add noise directly
                modified_values[:num_steps_to_noise] = modified_values[:num_steps_to_noise] + noise
            elif modified_values.shape[1] == 1:
                # Shape (T, 1) - reshape noise to (T, 1)
                modified_values[:num_steps_to_noise] = modified_values[:num_steps_to_noise] + noise[:, np.newaxis]
            else:
                # Shape (T, D) - broadcast noise across all dimensions
                modified_values[:num_steps_to_noise] = modified_values[:num_steps_to_noise] + noise[:, np.newaxis]
            
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
        suffix_str = f"_{args.output_suffix}" if args.output_suffix else ""
        base_name = f"{date_time_str}_{noise_str}{dim_str}{suffix_str}"
        
        if args.video_out_path is None:
            args.video_out_path = f"./videos/video_{args.task_suite_name}_{base_name}"
        if args.result_file_name is None:
            args.result_file_name = f"./results/result_{args.task_suite_name}_{base_name}.csv"
    else:
        # Extract base_name from existing paths
        base_name = pathlib.Path(args.video_out_path).name.replace(f"video_{args.task_suite_name}_", "")

    # Create analysis output path
    analysis_out_path = f"./analysis/analysis_{args.task_suite_name}_{base_name}"

    # Setup rollout save path if enabled
    if args.save_rollouts:
        if args.rollout_save_path is None:
            args.rollout_save_path = f"./rollouts/rollout_{args.task_suite_name}_{base_name}"
        pathlib.Path(args.rollout_save_path).mkdir(parents=True, exist_ok=True)
        logging.info(f"Rollout data will be saved to: {args.rollout_save_path}")

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

        # Initialize rollout episode data list for this task (if enabled)
        task_rollout_episodes = []

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
            
            # Initialize rollout data collection lists (if enabled)
            rollout_agentview_images = []
            rollout_eye_in_hand_images = []
            rollout_ee_states = []
            rollout_gripper_states = []
            rollout_joint_states = []
            rollout_actions = []
            
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
            # Debug: Check available image keys in initial observation
            image_keys = [k for k in obs.keys() if 'image' in k.lower()]
            print(f"[Episode {task_episodes+1}] Available image keys: {image_keys}", flush=True)
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
                    
                    # Check if wrist camera is available
                    if "robot0_eye_in_hand_image" not in obs:
                        print(f"WARNING: robot0_eye_in_hand_image not in obs! Available keys: {[k for k in obs.keys() if 'image' in k.lower()]}", flush=True)
                        # Use agentview as fallback (not ideal but prevents crash)
                        wrist_img = img.copy()
                    else:
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
                        if t == args.num_steps_wait:  # First action of episode
                            print(f"[STATE] eef_pos={eef_pos}, axis_angle={axis_angle}, gripper={gripper_qpos}", flush=True)
                        if args.debug_action:
                            logging.info(f"EEF pos: {eef_pos}, axis_angle: {axis_angle}, gripper: {gripper_qpos}")

                        # For debugging, we can also expand to individual states
                        element = {
                            # Video keys matching model metadata (use video.image as expected by the model)
                            "video.image": np.expand_dims(img, axis=0),  # Batch of 1: (1, H, W, C)
                            "video.wrist_image": np.expand_dims(wrist_img, axis=0),  # Batch of 1: (1, H, W, C)
                            # State keys matching model metadata (shape from metadata.json)
                            "state.x": eef_pos[np.newaxis, 0:1],  # (1, 1) - x position
                            "state.y": eef_pos[np.newaxis, 1:2],  # (1, 1) - y position
                            "state.z": eef_pos[np.newaxis, 2:3],  # (1, 1) - z position
                            "state.axis_angle1": axis_angle[np.newaxis, 0:1],  # (1, 1) - roll
                            "state.axis_angle2": axis_angle[np.newaxis, 1:2],  # (1, 1) - pitch
                            "state.axis_angle3": axis_angle[np.newaxis, 2:3],  # (1, 1) - yaw
                            "state.gripper_left_finger": gripper_qpos[np.newaxis, 0:1],  # (1, 1) - left finger
                            "state.gripper_right_finger": gripper_qpos[np.newaxis, 1:2],  # (1, 1) - right finger
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

                        # DEBUG: Verify wrist_image is in element right before sending
                        if "video.wrist_image" not in element:
                            print(f"[EVAL BUG] wrist_image NOT in element! Keys: {list(element.keys())}", flush=True)
                            import traceback
                            traceback.print_stack()
                        else:
                            print(f"[EVAL OK] wrist_image present, shape: {element['video.wrist_image'].shape}", flush=True)

                        # Query model to get action
                        action_chunk = policy.get_action(element)
                        
                        # TEMPORARY: Always log action chunk to debug failures
                        #t=step
                        if t == args.num_steps_wait:  # Only log first action per episode to reduce noise
                            print(f"[ACTION] First action chunk - keys: {list(action_chunk.keys())}", flush=True)
                            for key, val in action_chunk.items():
                                if val.ndim >= 1 and len(val) > 0:
                                    print(f"  {key}: shape={val.shape}, min={val.min():.4f}, max={val.max():.4f}, mean={val.mean():.4f}", flush=True)
                        
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
                            # Log before noise
                            if args.debug_action and args.action_noise_dim in action_chunk:
                                before_vals = action_chunk[args.action_noise_dim][:args.replan_steps].flatten()
                                print(f"[NOISE] Before: {args.action_noise_dim} = {before_vals}", flush=True)
                            
                            action_chunk = add_alternating_noise(
                                action_chunk, args.action_noise_scale, args.action_noise_dim, args.replan_steps,
                                debug=args.debug_action
                            )
                            
                            # Log after noise
                            if args.debug_action and args.action_noise_dim in action_chunk:
                                after_vals = action_chunk[args.action_noise_dim][:args.replan_steps].flatten()
                                print(f"[NOISE] After:  {args.action_noise_dim} = {after_vals}", flush=True)
                        
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
                    
                    # DEBUG: Log action values to understand failure
                    logging.info(f"Action values - pos_delta: {eef_pos_delta}, rot_delta: {eef_rot_delta}, gripper: {gripper_cmd}")
                    logging.info(f"Action ranges - pos: [{eef_pos_delta.min():.4f}, {eef_pos_delta.max():.4f}], rot: [{eef_rot_delta.min():.4f}, {eef_rot_delta.max():.4f}]")
                    
                    if args.debug_action:
                        logging.info(f"  eef_pos_delta={eef_pos_delta}, eef_rot_delta={eef_rot_delta}, gripper_cmd={gripper_cmd}")
                    
                    # Libero expects: [dx, dy, dz, rot_x, rot_y, rot_z, gripper]
                    # where gripper: +1 = open, -1 = close
                    # Model outputs gripper in range [0, 1] where 0 = open, 1 = close
                    # Convert: gripper_libero = 1 - 2 * gripper_model
                    gripper_libero = 1.0 - 2.0 * gripper_cmd
                    
                    libero_action = np.concatenate([
                        eef_pos_delta,
                        eef_rot_delta,
                        [gripper_libero]
                    ])
                    
                    # DEBUG: Print first few actions of each episode
                    if t < args.num_steps_wait + 3:
                        print(f"[DEBUG t={t}] pos=[{libero_action[0]:.3f},{libero_action[1]:.3f},{libero_action[2]:.3f}] rot=[{libero_action[3]:.3f},{libero_action[4]:.3f},{libero_action[5]:.3f}] grip={libero_action[6]:.3f}", flush=True)
                    
                    logging.info(f"Libero action (before env.step): {libero_action}")
                    
                    if args.debug_action:
                        logging.info(f"Libero action: {libero_action}")
                    obs, reward, done, info = env.step(libero_action.tolist())
                    
                    # Collect rollout data (if enabled)
                    if args.save_rollouts:
                        # Images (rotate 180 degrees to match train preprocessing)
                        agentview_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        eye_in_hand_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                        rollout_agentview_images.append(agentview_img)
                        rollout_eye_in_hand_images.append(eye_in_hand_img)
                        
                        # Proprioceptive states
                        eef_pos = obs["robot0_eef_pos"]
                        eef_quat = obs["robot0_eef_quat"]
                        axis_angle = _quat2axisangle(eef_quat.copy())
                        ee_state = np.concatenate([eef_pos, axis_angle])
                        rollout_ee_states.append(ee_state)
                        
                        gripper_state = obs["robot0_gripper_qpos"]
                        rollout_gripper_states.append(gripper_state)
                        
                        joint_state = obs["robot0_joint_pos"]
                        rollout_joint_states.append(joint_state)
                        
                        # Action (7D: eef_pos_delta (3) + eef_rot_delta (3) + gripper (1))
                        # Save in TRAINING format: gripper in [0, 1] range (0=open, 1=close)
                        # NOT libero format ([-1, 1] where +1=open, -1=close)
                        rollout_action = np.concatenate([
                            eef_pos_delta,
                            eef_rot_delta,
                            [gripper_cmd]  # Original model output [0, 1], not gripper_libero
                        ])
                        rollout_actions.append(rollout_action)
                    
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

            # Collect rollout episode data (if enabled and we have data)
            if args.save_rollouts and len(rollout_actions) > 0:
                episode_data = {
                    "agentview_images": rollout_agentview_images,
                    "eye_in_hand_images": rollout_eye_in_hand_images,
                    "ee_states": rollout_ee_states,
                    "gripper_states": rollout_gripper_states,
                    "joint_states": rollout_joint_states,
                    "actions": rollout_actions,
                    "success": done,
                    "task_description": task_description,
                }
                task_rollout_episodes.append(episode_data)
                logging.info(f"Collected rollout data: {len(rollout_actions)} steps, success={done}")

            # Log current results (both logging and print for visibility)
            success_str = "✓ SUCCESS" if done else "✗ FAILURE"
            print(f"\n[Episode Result] {success_str}", flush=True)
            print(f"[Progress] Episodes: {total_episodes}, Successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", flush=True)
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
        
        env.close() # Explicitly close the environment after all episodes for this task are done.
        
        # Save rollout data for this task (if enabled)
        if args.save_rollouts and len(task_rollout_episodes) > 0:
            task_segment = task_description.replace(" ", "_")
            # Include task_id to avoid filename conflicts when running multiple times
            hdf5_path = os.path.join(args.rollout_save_path, f"rollout_task{task_id:02d}_{task_segment}.hdf5")
            save_rollout_to_hdf5(
                hdf5_path=hdf5_path,
                episode_data_list=task_rollout_episodes,
                task_suite_name=args.task_suite_name,
                env_name="LIBERO",
            )
            # Count successes and failures
            num_success = sum(1 for ep in task_rollout_episodes if ep["success"])
            num_failure = len(task_rollout_episodes) - num_success
            print(f"[Rollout] Saved {len(task_rollout_episodes)} episodes ({num_success} success, {num_failure} failure) to {hdf5_path}", flush=True)

        # Log final results (both logging and print for visibility)
        task_success_rate = float(task_successes) / float(task_episodes) * 100
        total_success_rate = float(total_successes) / float(total_episodes) * 100
        print(f"\n{'='*60}", flush=True)
        print(f"[Task Complete] {task_description}", flush=True)
        print(f"[Task Success Rate] {task_successes}/{task_episodes} = {task_success_rate:.1f}%", flush=True)
        print(f"[Total Success Rate] {total_successes}/{total_episodes} = {total_success_rate:.1f}%", flush=True)
        print(f"{'='*60}\n", flush=True)
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

        result[task_description] = {
            "num_success": task_successes,
            "success_rate": float(task_successes) / float(task_episodes),
            "num_episodes": task_episodes
        }

    # Final summary (both logging and print for visibility)
    final_success_rate = float(total_successes) / float(total_episodes) * 100
    print(f"\n{'#'*60}", flush=True)
    print(f"#  EVALUATION COMPLETE", flush=True)
    print(f"#  Video output: {args.video_out_path}", flush=True)
    print(f"#  Total Episodes: {total_episodes}", flush=True)
    print(f"#  Total Successes: {total_successes}", flush=True)
    print(f"#  FINAL SUCCESS RATE: {final_success_rate:.1f}%", flush=True)
    print(f"{'#'*60}\n", flush=True)
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


###############################################################################
# Rollout Data Collection Functions (for creating training datasets)
###############################################################################

def save_rollout_to_hdf5(
    hdf5_path: str,
    episode_data_list: List[Dict[str, Any]],
    task_suite_name: str,
    env_name: str = "LIBERO",
) -> None:
    """
    Save collected rollout data to HDF5 file in LIBERO-compatible format.
    
    This follows the format used by LIBERO's create_dataset.py:
    - data/demo_X/obs/agentview_rgb
    - data/demo_X/obs/eye_in_hand_rgb
    - data/demo_X/obs/ee_states (eef_pos + axis_angle)
    - data/demo_X/obs/gripper_states
    - data/demo_X/obs/joint_states
    - data/demo_X/actions
    - data/demo_X/rewards
    - data/demo_X/dones
    
    Args:
        hdf5_path: Path to save the HDF5 file
        episode_data_list: List of episode data dictionaries, each containing:
            - 'agentview_images': List[np.ndarray] (H, W, 3)
            - 'eye_in_hand_images': List[np.ndarray] (H, W, 3)
            - 'ee_states': List[np.ndarray] (6,) - eef_pos (3) + axis_angle (3)
            - 'gripper_states': List[np.ndarray] (2,)
            - 'joint_states': List[np.ndarray] (n_joints,)
            - 'actions': List[np.ndarray] (7,)
            - 'success': bool
            - 'task_description': str
        task_suite_name: Name of the task suite (e.g., "libero_10")
        env_name: Environment name
    """
    pathlib.Path(hdf5_path).parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(hdf5_path, "w") as f:
        # Create data group
        grp = f.create_group("data")
        
        # Store metadata
        now = datetime.datetime.now()
        grp.attrs["date"] = f"{now.month}-{now.day}-{now.year}"
        grp.attrs["time"] = f"{now.hour}:{now.minute}:{now.second}"
        grp.attrs["env_name"] = env_name
        grp.attrs["task_suite_name"] = task_suite_name
        grp.attrs["num_demos"] = len(episode_data_list)
        
        total_samples = 0
        
        for i, episode_data in enumerate(episode_data_list):
            ep_grp = grp.create_group(f"demo_{i}")
            
            # Create obs group
            obs_grp = ep_grp.create_group("obs")
            
            # Save images
            agentview_images = np.stack(episode_data["agentview_images"], axis=0)
            eye_in_hand_images = np.stack(episode_data["eye_in_hand_images"], axis=0)
            obs_grp.create_dataset("agentview_rgb", data=agentview_images, compression="gzip")
            obs_grp.create_dataset("eye_in_hand_rgb", data=eye_in_hand_images, compression="gzip")
            
            # Save proprioceptive states
            ee_states = np.stack(episode_data["ee_states"], axis=0)
            obs_grp.create_dataset("ee_states", data=ee_states)
            obs_grp.create_dataset("ee_pos", data=ee_states[:, :3])
            obs_grp.create_dataset("ee_ori", data=ee_states[:, 3:])
            
            gripper_states = np.stack(episode_data["gripper_states"], axis=0)
            obs_grp.create_dataset("gripper_states", data=gripper_states)
            
            joint_states = np.stack(episode_data["joint_states"], axis=0)
            obs_grp.create_dataset("joint_states", data=joint_states)
            
            # Save actions
            actions = np.stack(episode_data["actions"], axis=0)
            ep_grp.create_dataset("actions", data=actions)
            
            # Save rewards and dones
            num_steps = len(episode_data["actions"])
            rewards = np.zeros(num_steps, dtype=np.uint8)
            dones = np.zeros(num_steps, dtype=np.uint8)
            if episode_data["success"]:
                rewards[-1] = 1
            dones[-1] = 1
            ep_grp.create_dataset("rewards", data=rewards)
            ep_grp.create_dataset("dones", data=dones)
            
            # Store episode metadata
            ep_grp.attrs["num_samples"] = num_steps
            ep_grp.attrs["task_description"] = episode_data["task_description"]
            ep_grp.attrs["success"] = episode_data["success"]
            
            total_samples += num_steps
        
        grp.attrs["total"] = total_samples
    
    logging.info(f"Saved {len(episode_data_list)} episodes ({total_samples} samples) to {hdf5_path}")


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


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_names": ["agentview", "robot0_eye_in_hand"],  # Include wrist camera
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


if __name__ == "__main__":
    # Configure logging with timestamp and flush
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Force immediate output
    for handler in logging.root.handlers:
        handler.flush()
    tyro.cli(eval_libero)
