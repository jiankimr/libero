"""
Parallel evaluation of LIBERO tasks at 500Hz with batched inference.

Runs N environments in separate subprocesses (via multiprocessing.Process + Pipe)
and batches observations from all environments that need new action chunks into a
single inference call.  This gives two sources of speedup:

  1. GPU batch inference -- one forward pass for N observations instead of N passes
  2. Parallel MuJoCo stepping -- N env.step() calls happen concurrently

Usage (same CLI as eval_500hz.py, with extra --num_parallel_envs):
  python eval_500hz_parallel.py --host node7 --port 5555 --task_suite_name libero_10
"""

import collections
import dataclasses
import logging
import multiprocessing as mp
import pathlib
import sys
import time
from typing import Optional, List, Dict, Any, Tuple
import datetime
import os
import traceback

import imageio
import tqdm
import tyro
import csv
import numpy as np

# ---------------------------------------------------------------------------
# Re-use utilities from eval_500hz.py (imported at module level)
# ---------------------------------------------------------------------------
from eval_500hz import (
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    patch_env_for_500hz_logging,
    unchunk,
    add_alternating_noise,
    save_rollout_to_hdf5,
    _quat2axisangle,
)

from service import ExternalRobotInferenceClient

from libero.libero import benchmark
from libero.libero import get_libero_path
# OffScreenRenderEnv imported inside worker process to avoid pickling issues

from analyze.basic_metric import compute_basic_metrics
from analyze.extra_metric import (
    compute_extra_metrics,
    save_metrics_to_csv,
    save_summary_csv,
)


###############################################################################
# Args dataclass (mirrors eval_500hz.py with extra parallel field)
###############################################################################

@dataclasses.dataclass
class Args:
    # Model server parameters
    host: str = "node7"
    port: int = 5555
    resize_size: int = 224
    replan_steps: int = 16

    # Action noise parameters
    action_noise_scale: float = 0.0
    action_noise_dim: Optional[str] = None
    noise_half_period: int = 2

    # LIBERO environment-specific parameters
    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10
    num_trials_per_task: int = 20

    # Utils
    video_out_path: Optional[str] = None
    result_file_name: Optional[str] = None
    seed: int = 42

    # Debugging
    debug_action: bool = False

    # Save physical quantities
    save_metrics: bool = True
    output_suffix: Optional[str] = None

    # Rollout data collection
    save_rollouts: bool = True
    rollout_save_path: Optional[str] = None

    # ---- Parallel-specific ----
    num_parallel_envs: int = 20  # Number of parallel environments (default = num_trials_per_task)


###############################################################################
# Subprocess worker function
###############################################################################

def _env_worker(
    pipe: mp.connection.Connection,
    worker_id: int,
    task_bddl_file: str,
    resolution: int,
    seed: int,
    save_metrics: bool,
):
    """
    Target function for each subprocess.  Creates its own OffScreenRenderEnv,
    applies the 500Hz monkey-patch, and services commands from the main process.

    Protocol (receive cmd tuple, send result tuple):
      ("reset", None)              -> ("reset_ok", obs_dict)
      ("set_init_state", state)    -> ("init_ok", obs_dict)
      ("step", action_list)        -> ("step_ok", obs_dict, reward, done, info, substep_data)
      ("close", None)              -> ("closed",)
    """
    try:
        # Import inside subprocess to avoid OpenGL context issues
        from libero.libero.envs import OffScreenRenderEnv

        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": resolution,
            "camera_widths": resolution,
            "camera_names": ["agentview", "robot0_eye_in_hand"],
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(seed)

        # Apply 500Hz monkey-patch
        patch_env_for_500hz_logging(env)

        while True:
            try:
                cmd, data = pipe.recv()
            except EOFError:
                break

            if cmd == "reset":
                obs = env.reset()
                pipe.send(("reset_ok", _serialise_obs(obs)))

            elif cmd == "set_init_state":
                obs = env.set_init_state(data)
                pipe.send(("init_ok", _serialise_obs(obs)))

            elif cmd == "step":
                obs, reward, done, info = env.step(data)

                substep_data = None
                if save_metrics:
                    substep_data = {
                        "torques": [t.copy() for t in env.env._substep_torques],
                        "qvel":    [v.copy() for v in env.env._substep_qvel],
                        "qacc":    [a.copy() for a in env.env._substep_qacc],
                    }

                pipe.send(("step_ok", _serialise_obs(obs), float(reward), bool(done), info, substep_data))

            elif cmd == "get_env_info":
                # Return environment timing info for metric setup
                info = {
                    "model_timestep": float(env.env.model_timestep),
                    "control_timestep": float(env.env.control_timestep),
                    "n_joints": len(env.env.robots[0].joint_indexes),
                }
                pipe.send(("env_info", info))

            elif cmd == "close":
                env.close()
                pipe.send(("closed",))
                break

    except Exception as e:
        # Make sure the main process is not left hanging
        try:
            pipe.send(("error", str(e), traceback.format_exc()))
        except Exception:
            pass


def _serialise_obs(obs: dict) -> dict:
    """
    Ensure every value in the observation dict is a plain numpy array
    (so it can be sent through a Pipe without issues).
    """
    out = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            out[k] = v.copy()
        else:
            out[k] = v
    return out


###############################################################################
# ParallelEnvManager
###############################################################################

class ParallelEnvManager:
    """
    Manages N environment worker processes for a single LIBERO task.

    Lifecycle:
        mgr = ParallelEnvManager.create(task, n_envs, ...)
        mgr.reset_all()
        obs_list = mgr.set_init_states(states)
        ...
        obs_list, rewards, dones, infos, substep_list = mgr.step(actions, mask)
        ...
        mgr.close()
    """

    def __init__(self, pipes, processes, n_envs, save_metrics):
        self.pipes = pipes          # list[Connection]  (parent side)
        self.processes = processes   # list[Process]
        self.n_envs = n_envs
        self.save_metrics = save_metrics

    # ------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        task,
        n_envs: int,
        resolution: int,
        seed: int,
        save_metrics: bool,
    ) -> "ParallelEnvManager":
        """Spawn *n_envs* worker processes (uses 'spawn' context)."""
        task_bddl_file = str(
            pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        )

        ctx = mp.get_context("spawn")
        pipes = []
        procs = []
        for i in range(n_envs):
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(
                target=_env_worker,
                args=(child_conn, i, task_bddl_file, resolution, seed, save_metrics),
                daemon=True,
            )
            p.start()
            child_conn.close()  # parent does not need child end
            pipes.append(parent_conn)
            procs.append(p)

        return cls(pipes, procs, n_envs, save_metrics)

    # ------------------------------------------------------------------
    def get_env_info(self, idx: int = 0) -> dict:
        """Get environment timing info from worker *idx*."""
        self.pipes[idx].send(("get_env_info", None))
        tag, info = self.pipes[idx].recv()
        if tag == "error":
            raise RuntimeError(f"Worker error: {info}")
        return info

    # ------------------------------------------------------------------
    def reset_all(self) -> List[dict]:
        for p in self.pipes:
            p.send(("reset", None))
        obs_list = []
        for p in self.pipes:
            msg = p.recv()
            if msg[0] == "error":
                raise RuntimeError(f"Worker error: {msg[1]}\n{msg[2]}")
            obs_list.append(msg[1])
        return obs_list

    # ------------------------------------------------------------------
    def set_init_states(self, states) -> List[dict]:
        """Send initial state to each worker. *states* is indexable."""
        for i, p in enumerate(self.pipes):
            p.send(("set_init_state", states[i]))
        obs_list = []
        for p in self.pipes:
            msg = p.recv()
            if msg[0] == "error":
                raise RuntimeError(f"Worker error: {msg[1]}\n{msg[2]}")
            obs_list.append(msg[1])
        return obs_list

    # ------------------------------------------------------------------
    def step(
        self,
        actions: List,
        active_mask: List[bool],
    ) -> Tuple[List[Optional[dict]], List[float], List[bool], List[dict], List[Optional[dict]]]:
        """
        Step all active environments simultaneously.

        Args:
            actions:      list of length n_envs (action list for active, None for inactive)
            active_mask:  bool list -- True for envs that should be stepped

        Returns:
            obs_list, reward_list, done_list, info_list, substep_data_list
        """
        # Send step commands to active workers
        for i in range(self.n_envs):
            if active_mask[i]:
                self.pipes[i].send(("step", actions[i]))

        # Receive results (preserve ordering)
        obs_list: List[Optional[dict]] = [None] * self.n_envs
        reward_list = [0.0] * self.n_envs
        done_list = [False] * self.n_envs
        info_list: List[dict] = [{}] * self.n_envs
        substep_list: List[Optional[dict]] = [None] * self.n_envs

        for i in range(self.n_envs):
            if active_mask[i]:
                msg = self.pipes[i].recv()
                if msg[0] == "error":
                    raise RuntimeError(f"Worker {i} error: {msg[1]}\n{msg[2]}")
                _, obs, reward, done, info, substep_data = msg
                obs_list[i] = obs
                reward_list[i] = reward
                done_list[i] = done
                info_list[i] = info
                substep_list[i] = substep_data

        return obs_list, reward_list, done_list, info_list, substep_list

    # ------------------------------------------------------------------
    def close(self):
        for p in self.pipes:
            try:
                p.send(("close", None))
            except Exception:
                pass
        for p in self.pipes:
            try:
                p.recv()
            except Exception:
                pass
        for proc in self.processes:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()


###############################################################################
# Observation batching / action unbatching helpers
###############################################################################

def _build_single_element(obs: dict, task_description: str) -> dict:
    """
    Construct the inference-ready observation dict for a single env
    with explicit batch dimension: (B=1, T=1, ...).

    Gr00tPolicy.get_action expects batched inputs as:
        video:  (B, T, H, W, C)
        state:  (B, T, D)
    When multiple elements are concatenated on axis=0 by batch_observations(),
    the resulting shapes (N, T, ...) are correctly recognised as batched.
    """
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])

    if "robot0_eye_in_hand_image" not in obs:
        wrist_img = img.copy()
    else:
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

    eef_pos = obs["robot0_eef_pos"]
    axis_angle = _quat2axisangle(obs["robot0_eef_quat"].copy())
    gripper_qpos = obs["robot0_gripper_qpos"]

    return {
        # Video: (B=1, T=1, H, W, C)
        "video.image": img[np.newaxis, np.newaxis],
        "video.wrist_image": wrist_img[np.newaxis, np.newaxis],
        # State: (B=1, T=1, D=1)
        "state.x": eef_pos[np.newaxis, np.newaxis, 0:1],
        "state.y": eef_pos[np.newaxis, np.newaxis, 1:2],
        "state.z": eef_pos[np.newaxis, np.newaxis, 2:3],
        "state.axis_angle1": axis_angle[np.newaxis, np.newaxis, 0:1],
        "state.axis_angle2": axis_angle[np.newaxis, np.newaxis, 1:2],
        "state.axis_angle3": axis_angle[np.newaxis, np.newaxis, 2:3],
        "state.gripper_left_finger": gripper_qpos[np.newaxis, np.newaxis, 0:1],
        "state.gripper_right_finger": gripper_qpos[np.newaxis, np.newaxis, 1:2],
        "annotation.human.action.task_description": [str(task_description)],
    }


def batch_observations(elements: List[dict]) -> dict:
    """
    Stack N single-element dicts into one batched dict.

    numpy arrays:  (1, ...) per env  ->  (N, ...)
    string lists:  ["desc"] per env  ->  ["desc0", "desc1", ...]
    """
    if len(elements) == 1:
        return elements[0]  # no stacking needed

    batched: dict = {}
    keys = elements[0].keys()
    for k in keys:
        vals = [e[k] for e in elements]
        if isinstance(vals[0], np.ndarray):
            batched[k] = np.concatenate(vals, axis=0)   # (N, ...)
        elif isinstance(vals[0], list):
            # annotation strings -- flat concatenation
            batched[k] = sum(vals, [])
        else:
            batched[k] = vals
    return batched


def unbatch_actions(batched_action: dict, n: int) -> List[dict]:
    """
    Split a batched action dict  { key: (B, T, ...) }  into B single dicts
    { key: (T, ...) }.

    We always index into axis-0 (batch dim) because the server returns
    (B, T, D) shaped arrays even when B == 1.
    """
    result = []
    for i in range(n):
        single: dict = {}
        for k, v in batched_action.items():
            if isinstance(v, np.ndarray):
                single[k] = v[i]
            else:
                single[k] = v
        result.append(single)
    return result


###############################################################################
# Metric helpers (500Hz, processed in main process)
###############################################################################

def _process_substep_metrics(
    substep_data: dict,
    torque_list: list,
    velocity_list: list,
    acceleration_list: list,
    kinematic_jerk_delta_list: list,
    actuator_jerk_delta_list: list,
    electric_energy_list: list,
    recent_joint_acc,
    recent_torques,
    recent_velocities,
    dt_sim: float,
):
    """Append 500Hz sub-step data from one env.step() to the running lists."""
    from robosuite.utils.buffers import DeltaBuffer  # local import to keep top-level light

    for sub_torque, sub_vel, sub_acc in zip(
        substep_data["torques"], substep_data["qvel"], substep_data["qacc"]
    ):
        torque_list.append(sub_torque)
        recent_torques.push(sub_torque)

        velocity_list.append(sub_vel)
        recent_velocities.push(sub_vel)

        acceleration_list.append(sub_acc)
        recent_joint_acc.push(sub_acc)

        kinematic_jerk_delta_list.append(recent_joint_acc.delta / dt_sim)
        actuator_jerk_delta_list.append(recent_torques.delta / dt_sim)

        t_avg = recent_torques.average
        v_avg = recent_velocities.average
        P = np.sum(t_avg * v_avg)
        E = P * dt_sim
        electric_energy_list.append(E)


###############################################################################
# Main evaluation function
###############################################################################

def eval_libero_parallel(args: Args) -> None:
    np.random.seed(args.seed)

    # ---- output paths (same logic as eval_500hz.py) ----
    if args.video_out_path is None or args.result_file_name is None:
        date_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        noise_str = f"noise_{args.action_noise_scale:.4f}".replace(".", "")
        dim_str = f"_dim_{args.action_noise_dim}" if args.action_noise_dim else ""
        hp_str = f"_hp{args.noise_half_period}" if args.noise_half_period != 2 else ""
        suffix_str = f"_{args.output_suffix}" if args.output_suffix else ""
        base_name = f"{date_time_str}_{noise_str}{dim_str}{hp_str}{suffix_str}"

        if args.video_out_path is None:
            args.video_out_path = f"./videos/video_{args.task_suite_name}_{base_name}"
        if args.result_file_name is None:
            args.result_file_name = f"./results/result_{args.task_suite_name}_{base_name}.csv"
    else:
        base_name = pathlib.Path(args.video_out_path).name.replace(
            f"video_{args.task_suite_name}_", ""
        )

    analysis_out_path = f"./analysis/analysis_{args.task_suite_name}_{base_name}"

    if args.save_rollouts:
        if args.rollout_save_path is None:
            args.rollout_save_path = f"./rollouts/rollout_{args.task_suite_name}_{base_name}"
        pathlib.Path(args.rollout_save_path).mkdir(parents=True, exist_ok=True)

    # ---- task suite ----
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name} ({num_tasks_in_suite} tasks)")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(pathlib.Path(args.result_file_name).parent).mkdir(parents=True, exist_ok=True)
    pathlib.Path(analysis_out_path).mkdir(parents=True, exist_ok=True)

    max_steps_map = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    if args.task_suite_name not in max_steps_map:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    max_steps = max_steps_map[args.task_suite_name]

    # ---- inference client (single ZMQ connection) ----
    policy = ExternalRobotInferenceClient(host=args.host, port=args.port)

    result: Dict[str, dict] = {}
    all_episode_metrics: List[dict] = []

    total_episodes, total_successes = 0, 0

    # ======================================================================
    # Iterate over tasks
    # ======================================================================
    for task_id in tqdm.tqdm(range(num_tasks_in_suite), desc="Tasks"):
        task = task_suite.get_task(task_id)
        task_description = task.language
        initial_states = task_suite.get_task_init_states(task_id)

        n_trials = args.num_trials_per_task
        n_envs = min(args.num_parallel_envs, n_trials)

        task_successes = 0
        task_episodes = 0
        task_rollout_episodes: List[dict] = []

        # Process trials in batches of n_envs
        for batch_start in range(0, n_trials, n_envs):
            batch_end = min(batch_start + n_envs, n_trials)
            batch_size = batch_end - batch_start
            episode_indices = list(range(batch_start, batch_end))

            print(
                f"\n[Parallel] Task {task_id}: episodes {batch_start}-{batch_end-1} "
                f"({batch_size} envs)",
                flush=True,
            )

            # ----------------------------------------------------------
            # Spawn workers for this batch
            # ----------------------------------------------------------
            t_spawn_start = time.time()
            mgr = ParallelEnvManager.create(
                task,
                n_envs=batch_size,
                resolution=LIBERO_ENV_RESOLUTION,
                seed=args.seed,
                save_metrics=args.save_metrics,
            )
            print(
                f"[Parallel] Workers spawned in {time.time() - t_spawn_start:.1f}s",
                flush=True,
            )

            # Get env timing info for metrics
            env_info = mgr.get_env_info(0)
            dt_sim = env_info["model_timestep"]    # 0.002s (500Hz)
            dt_ctrl = env_info["control_timestep"]  # 0.05s  (20Hz)
            n_substeps = int(dt_ctrl / dt_sim)       # 25
            n_joints = env_info["n_joints"]

            # ----------------------------------------------------------
            # Reset & set initial states
            # ----------------------------------------------------------
            mgr.reset_all()
            obs_list = mgr.set_init_states(
                [initial_states[ei] for ei in episode_indices]
            )

            # ----------------------------------------------------------
            # Per-env episode state
            # ----------------------------------------------------------
            action_plans: List[collections.deque] = [
                collections.deque() for _ in range(batch_size)
            ]
            done_flags = [False] * batch_size
            step_counts = [0] * batch_size

            # Per-env image buffers (for video)
            replay_images: List[List[np.ndarray]] = [[] for _ in range(batch_size)]

            # Per-env rollout data
            rollout_data: List[dict] = [
                {
                    "agentview_images": [],
                    "eye_in_hand_images": [],
                    "ee_states": [],
                    "gripper_states": [],
                    "joint_states": [],
                    "actions": [],
                }
                for _ in range(batch_size)
            ]

            # Per-env 500Hz metric accumulators
            from robosuite.utils.buffers import DeltaBuffer

            metric_acc: List[dict] = []
            for _ in range(batch_size):
                acc: dict = {
                    "torque": [],
                    "velocity": [],
                    "acceleration": [],
                    "kin_jerk": [],
                    "act_jerk": [],
                    "energy": [],
                    "recent_acc": DeltaBuffer(dim=n_joints),
                    "recent_torque": DeltaBuffer(dim=n_joints),
                    "recent_vel": DeltaBuffer(dim=n_joints),
                }
                metric_acc.append(acc)

            # ----------------------------------------------------------
            # Main stepping loop
            # ----------------------------------------------------------
            max_t = max_steps + args.num_steps_wait
            t_loop_start = time.time()

            n_active_prev = batch_size
            pbar = tqdm.tqdm(
                total=max_t,
                desc=f"Task {task_id} ep{batch_start}-{batch_end-1}",
                leave=False,
            )

            while True:
                # Check if all envs are done or past max steps
                all_finished = all(
                    done_flags[i] or step_counts[i] >= max_t
                    for i in range(batch_size)
                )
                if all_finished:
                    break

                # Build active mask
                active_mask = [
                    not done_flags[i] and step_counts[i] < max_t
                    for i in range(batch_size)
                ]

                # ---- Wait phase: dummy actions ----
                in_wait = [
                    active_mask[i] and step_counts[i] < args.num_steps_wait
                    for i in range(batch_size)
                ]
                if any(in_wait):
                    wait_actions = [
                        LIBERO_DUMMY_ACTION if in_wait[i] else None
                        for i in range(batch_size)
                    ]
                    wait_mask = in_wait
                    new_obs, _, _, _, _ = mgr.step(wait_actions, wait_mask)
                    for i in range(batch_size):
                        if in_wait[i]:
                            obs_list[i] = new_obs[i]
                            step_counts[i] += 1
                    # Update progress bar (use min step across active envs)
                    min_step = min(
                        step_counts[i] for i in range(batch_size)
                        if not done_flags[i]
                    ) if not all(done_flags) else max_t
                    pbar.n = min_step
                    n_active = sum(1 for i in range(batch_size) if not done_flags[i])
                    pbar.set_postfix(active=n_active, refresh=False)
                    pbar.refresh()
                    continue

                # ---- Identify envs that need new action chunks ----
                needs_inference = [
                    i
                    for i in range(batch_size)
                    if active_mask[i] and len(action_plans[i]) == 0
                ]

                if needs_inference:
                    # Build single-env observation dicts
                    single_elements = [
                        _build_single_element(obs_list[i], task_description)
                        for i in needs_inference
                    ]

                    # Batch observations and call inference server ONCE
                    batched_obs = batch_observations(single_elements)
                    batched_action = policy.get_action(batched_obs)

                    # Unbatch returned actions
                    per_env_actions = unbatch_actions(batched_action, len(needs_inference))

                    # Fill action plans
                    for j, env_idx in enumerate(needs_inference):
                        action_chunk = per_env_actions[j]

                        # Add alternating noise if specified
                        if args.action_noise_scale > 0:
                            action_chunk = add_alternating_noise(
                                action_chunk,
                                args.action_noise_scale,
                                args.action_noise_dim,
                                args.replan_steps,
                                half_period=args.noise_half_period,
                                debug=args.debug_action,
                            )

                        unchunk(
                            action_chunk,
                            action_plans[env_idx],
                            args.replan_steps,
                            debug=args.debug_action,
                        )

                # ---- Pop one action per active env and step ----
                step_actions: List = [None] * batch_size
                per_env_raw_action: List = [None] * batch_size  # for rollout saving
                step_mask = [False] * batch_size

                for i in range(batch_size):
                    if active_mask[i] and len(action_plans[i]) > 0:
                        raw_action = action_plans[i].popleft()
                        per_env_raw_action[i] = raw_action

                        eef_pos_delta = raw_action[:3]
                        eef_rot_delta = raw_action[3:6]
                        gripper_cmd = raw_action[6]
                        gripper_libero = 1.0 - 2.0 * gripper_cmd

                        libero_action = np.concatenate(
                            [eef_pos_delta, eef_rot_delta, [gripper_libero]]
                        )
                        step_actions[i] = libero_action.tolist()
                        step_mask[i] = True

                if not any(step_mask):
                    # Nothing to step (should not happen normally)
                    break

                new_obs, rewards, dones, infos, substep_list = mgr.step(
                    step_actions, step_mask
                )

                # ---- Process results ----
                for i in range(batch_size):
                    if not step_mask[i]:
                        continue

                    obs_list[i] = new_obs[i]

                    # Save replay image
                    img = np.ascontiguousarray(
                        new_obs[i]["agentview_image"][::-1, ::-1]
                    )
                    replay_images[i].append(img)

                    # Rollout data collection
                    if args.save_rollouts and per_env_raw_action[i] is not None:
                        obs_i = new_obs[i]
                        agentview_img = np.ascontiguousarray(
                            obs_i["agentview_image"][::-1, ::-1]
                        )
                        eye_in_hand_img = np.ascontiguousarray(
                            obs_i["robot0_eye_in_hand_image"][::-1, ::-1]
                        )
                        rollout_data[i]["agentview_images"].append(agentview_img)
                        rollout_data[i]["eye_in_hand_images"].append(eye_in_hand_img)

                        eef_pos = obs_i["robot0_eef_pos"]
                        eef_quat = obs_i["robot0_eef_quat"]
                        axis_angle = _quat2axisangle(eef_quat.copy())
                        rollout_data[i]["ee_states"].append(
                            np.concatenate([eef_pos, axis_angle])
                        )
                        rollout_data[i]["gripper_states"].append(
                            obs_i["robot0_gripper_qpos"]
                        )
                        rollout_data[i]["joint_states"].append(
                            obs_i["robot0_joint_pos"]
                        )

                        raw = per_env_raw_action[i]
                        rollout_data[i]["actions"].append(
                            np.concatenate([raw[:3], raw[3:6], [raw[6]]])
                        )

                    # 500Hz metric accumulation
                    if args.save_metrics and substep_list[i] is not None:
                        acc = metric_acc[i]
                        _process_substep_metrics(
                            substep_list[i],
                            acc["torque"],
                            acc["velocity"],
                            acc["acceleration"],
                            acc["kin_jerk"],
                            acc["act_jerk"],
                            acc["energy"],
                            acc["recent_acc"],
                            acc["recent_torque"],
                            acc["recent_vel"],
                            dt_sim,
                        )

                    # Check done
                    if dones[i]:
                        done_flags[i] = True

                    step_counts[i] += 1

                # Update progress bar
                min_step = min(
                    step_counts[i] for i in range(batch_size)
                    if not done_flags[i]
                ) if not all(done_flags) else max_t
                pbar.n = min_step
                n_active = sum(1 for i in range(batch_size) if not done_flags[i])
                pbar.set_postfix(active=n_active, refresh=False)
                pbar.refresh()

            pbar.n = max_t
            pbar.refresh()
            pbar.close()

            elapsed = time.time() - t_loop_start
            print(
                f"[Parallel] Batch finished in {elapsed:.1f}s",
                flush=True,
            )

            # ----------------------------------------------------------
            # Post-batch: save videos, metrics, rollouts
            # ----------------------------------------------------------
            task_segment = task_description.replace(" ", "_")

            for local_idx in range(batch_size):
                episode_idx = episode_indices[local_idx]
                success = done_flags[local_idx]
                suffix = "success" if success else "failure"

                if success:
                    task_successes += 1
                    total_successes += 1
                task_episodes += 1
                total_episodes += 1

                # --- Video ---
                if replay_images[local_idx]:
                    imageio.mimwrite(
                        pathlib.Path(args.video_out_path)
                        / f"rollout_{task_segment}_ep{episode_idx}_{suffix}.mp4",
                        [np.asarray(x) for x in replay_images[local_idx]],
                        fps=10,
                    )

                # --- 500Hz Metrics ---
                if args.save_metrics:
                    try:
                        acc = metric_acc[local_idx]
                        if len(acc["torque"]) > 0:
                            compute_basic_metrics(
                                torque_list=acc["torque"],
                                velocity_list=acc["velocity"],
                                acceleration_list=acc["acceleration"],
                                kinematic_jerk_delta_list=acc["kin_jerk"],
                                actuator_jerk_delta_list=acc["act_jerk"],
                                electric_energy_list=acc["energy"],
                                dt=dt_sim,
                                output_dir=analysis_out_path,
                                task_segment=task_segment,
                                episode_idx=episode_idx,
                                debug=args.debug_action,
                                suffix=suffix,
                            )

                            # Torque current (backwards compat)
                            torque_current_array = np.array(acc["torque"])
                            np.save(
                                pathlib.Path(analysis_out_path)
                                / f"torque_current_{task_segment}_{episode_idx}_{suffix}.npy",
                                torque_current_array,
                            )

                        extra_metrics = compute_extra_metrics(
                            npy_dir=analysis_out_path,
                            task_segment=task_segment,
                            episode_idx=episode_idx,
                            suffix=suffix,
                            output_dir=analysis_out_path,
                        )

                        save_metrics_to_csv(
                            all_metrics=extra_metrics,
                            task_segment=task_segment,
                            episode_idx=episode_idx,
                            suffix=suffix,
                            output_dir=analysis_out_path,
                            task_description=task_description,
                        )

                        all_episode_metrics.append(
                            {
                                "task": task_description,
                                "episode": episode_idx,
                                "success": 1 if success else 0,
                                "dt": dt_sim,
                                "metrics": extra_metrics,
                            }
                        )
                    except Exception as e:
                        logging.error(f"Failed to compute metrics for ep {episode_idx}: {e}")

                # --- Rollout HDF5 data ---
                if args.save_rollouts and len(rollout_data[local_idx]["actions"]) > 0:
                    episode_data = {
                        **rollout_data[local_idx],
                        "success": success,
                        "task_description": task_description,
                    }
                    task_rollout_episodes.append(episode_data)

                # --- Progress log ---
                success_str = "SUCCESS" if success else "FAILURE"
                print(
                    f"  [Ep {episode_idx}] {success_str} | "
                    f"Total: {total_successes}/{total_episodes} "
                    f"({total_successes / total_episodes * 100:.1f}%)",
                    flush=True,
                )

            # Close workers for this batch
            mgr.close()

        # ==============================================================
        # End of task: save rollout HDF5 & log
        # ==============================================================
        if args.save_rollouts and task_rollout_episodes:
            task_segment = task_description.replace(" ", "_")
            hdf5_path = os.path.join(
                args.rollout_save_path,
                f"rollout_task{task_id:02d}_{task_segment}.hdf5",
            )
            save_rollout_to_hdf5(
                hdf5_path=hdf5_path,
                episode_data_list=task_rollout_episodes,
                task_suite_name=args.task_suite_name,
                env_name="LIBERO",
            )
            num_success = sum(1 for ep in task_rollout_episodes if ep["success"])
            num_failure = len(task_rollout_episodes) - num_success
            print(
                f"[Rollout] Saved {len(task_rollout_episodes)} episodes "
                f"({num_success} success, {num_failure} failure) to {hdf5_path}",
                flush=True,
            )

        task_success_rate = float(task_successes) / float(task_episodes) * 100
        total_success_rate = float(total_successes) / float(total_episodes) * 100
        print(f"\n{'='*60}", flush=True)
        print(f"[Task Complete] {task_description}", flush=True)
        print(
            f"[Task Success Rate] {task_successes}/{task_episodes} = "
            f"{task_success_rate:.1f}%",
            flush=True,
        )
        print(
            f"[Total Success Rate] {total_successes}/{total_episodes} = "
            f"{total_success_rate:.1f}%",
            flush=True,
        )
        print(f"{'='*60}\n", flush=True)

        result[task_description] = {
            "num_success": task_successes,
            "success_rate": float(task_successes) / float(task_episodes),
            "num_episodes": task_episodes,
        }

    # ==================================================================
    # Final summary
    # ==================================================================
    final_success_rate = float(total_successes) / float(total_episodes) * 100
    print(f"\n{'#'*60}", flush=True)
    print(f"#  EVALUATION COMPLETE (parallel, {args.num_parallel_envs} envs)", flush=True)
    print(f"#  Video output: {args.video_out_path}", flush=True)
    print(f"#  Total Episodes: {total_episodes}", flush=True)
    print(f"#  Total Successes: {total_successes}", flush=True)
    print(f"#  FINAL SUCCESS RATE: {final_success_rate:.1f}%", flush=True)
    print(f"{'#'*60}\n", flush=True)

    # Save summary metrics CSV
    if args.save_metrics and all_episode_metrics:
        summary_csv_file = (
            pathlib.Path(analysis_out_path).parent
            / f"analysis_summary_{args.task_suite_name}_{base_name}.csv"
        )
        save_summary_csv(
            all_episode_metrics=all_episode_metrics,
            output_file=str(summary_csv_file),
            task_suite_name=args.task_suite_name,
            base_name=base_name,
        )

    # Save result CSV
    with open(args.result_file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["task_description", "num_episodes", "num_success", "success_rate"]
        )
        for td, data in result.items():
            writer.writerow(
                [td, data["num_episodes"], data["num_success"], data["success_rate"]]
            )


###############################################################################
# Entry point
###############################################################################

if __name__ == "__main__":
    # Use spawn for safe MuJoCo/OpenGL subprocess creation
    mp.set_start_method("spawn", force=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    for handler in logging.root.handlers:
        handler.flush()

    tyro.cli(eval_libero_parallel)
