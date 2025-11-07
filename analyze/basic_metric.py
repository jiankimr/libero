"""
Basic Metrics 계산 및 저장

- Torque, Velocity, Acceleration 측정
- Kinematic Jerk (DeltaBuffer delta & Gradient)
- Actuator Jerk (DeltaBuffer delta & Gradient)
- Electric Energy (부호 유지)
- 모든 데이터를 joint별 npy 파일로 저장
"""

import numpy as np
import pathlib
import logging
from typing import Optional, Dict, List

from robosuite.utils.buffers import DeltaBuffer


def infer_control_dt(env, debug: bool = False) -> float:
    """
    Infer control timestep from environment.
    
    Priority: control_timestep > control_freq > MuJoCo timestep
    
    Args:
        env: Environment object
        debug: If True, log detailed dt inference process
    """
    candidates = []
    for attr in ("env", "wrapped_env", "unwrapped"):
        if hasattr(env, attr):
            candidates.append(getattr(env, attr))
    candidates.insert(0, env)

    if debug:
        logging.info(f"[dt] Starting inference with {len(candidates)} candidate objects")
    
    dt_candidates = []
    
    for i, obj in enumerate(candidates):
        if obj is None:
            if debug:
                logging.info(f"[dt] Candidate {i} is None, skipping")
            continue
            
        if debug:
            logging.info(f"[dt] Checking candidate {i}: {type(obj).__name__}")
        
        if hasattr(obj, "control_timestep"):
            dt = float(obj.control_timestep)
            dt_candidates.append(("control_timestep", dt, type(obj).__name__))
            if debug:
                logging.info(f"[dt] Found control_timestep={dt:.6f}s from {type(obj).__name__}")
            
        if hasattr(obj, "control_freq"):
            dt = 1.0 / float(obj.control_freq)
            dt_candidates.append(("control_freq", dt, type(obj).__name__))
            if debug:
                logging.info(f"[dt] Found control_freq={1.0/dt:.1f}Hz -> dt={dt:.6f}s from {type(obj).__name__}")
            
        sim = getattr(obj, "sim", None)
        if sim is not None and hasattr(sim, "model") and hasattr(sim.model, "opt"):
            try:
                mj_dt = float(sim.model.opt.timestep)
                if debug:
                    logging.info(f"[dt] Found MuJoCo timestep={mj_dt:.6f}s from {type(obj).__name__}")
                
                for mult_name in ("control_decimation", "frame_skip", "n_substeps", "action_repeat"):
                    if hasattr(obj, mult_name):
                        multiplier = float(getattr(obj, mult_name))
                        dt = mj_dt * multiplier
                        dt_candidates.append((f"mujoco:{mult_name}", dt, type(obj).__name__))
                        if debug:
                            logging.info(f"[dt] Found MuJoCo with {mult_name}={multiplier} -> dt={dt:.6f}s from {type(obj).__name__}")
                
                dt_candidates.append(("mujoco:raw", mj_dt, type(obj).__name__))
                
            except Exception as e:
                logging.warning(f"[dt] Error processing MuJoCo timestep from {type(obj).__name__}: {e}")
                pass
        else:
            if debug:
                logging.info(f"[dt] No sim attribute or MuJoCo model found in {type(obj).__name__}")
    
    if not dt_candidates:
        logging.error("[dt] FAILED: No dt values found, using default dt=0.05s (20 Hz)")
        return 0.05
    
    dt_candidates.sort(key=lambda x: x[1])
    
    if debug:
        logging.info(f"[dt] Found {len(dt_candidates)} dt candidates:")
        for source, dt, obj_type in dt_candidates:
            logging.info(f"[dt]   {source}: {dt:.6f}s from {obj_type}")
    
    if len(dt_candidates) > 1:
        min_dt, max_dt = dt_candidates[0][1], dt_candidates[-1][1]
        if max_dt / min_dt > 10:
            logging.warning(f"[dt] WARNING: Large variation in dt values: min={min_dt:.6f}s, max={max_dt:.6f}s (ratio={max_dt/min_dt:.1f})")
    
    priority_order = ["control_timestep", "control_freq", "mujoco:raw"]
    
    for priority in priority_order:
        for source, dt, obj_type in dt_candidates:
            if source.startswith(priority):
                if debug:
                    logging.info(f"[dt] SUCCESS: Selected {source}={dt:.6f}s from {obj_type} (highest priority)")
                return dt
    
    selected_source, selected_dt, selected_obj = dt_candidates[0]
    if debug:
        logging.info(f"[dt] SUCCESS: Selected {selected_source}={selected_dt:.6f}s from {selected_obj} (first available)")
    return selected_dt


def compute_kinematic_jerk_gradient(acceleration_list: List[np.ndarray], dt: float) -> Optional[np.ndarray]:
    """
    Compute kinematic jerk using gradient method from full acceleration history.
    
    Args:
        acceleration_list: List of acceleration arrays (T,), each shape (n_joints,)
        dt: Control timestep
    
    Returns:
        jerk_gradient: shape (T, n_joints) or None if insufficient data
    """
    if len(acceleration_list) < 2:
        return None
    
    try:
        acc_array = np.array(acceleration_list)  # shape (T, n_joints)
        # gradient along time axis (axis=0)
        jerk_gradient = np.gradient(acc_array, dt, axis=0, edge_order=2 if len(acc_array) >= 3 else 1)
        return jerk_gradient
    except Exception as e:
        logging.error(f"Error computing kinematic jerk gradient: {e}")
        return None


def compute_actuator_jerk_gradient(torque_list: List[np.ndarray], dt: float) -> Optional[np.ndarray]:
    """
    Compute actuator jerk using gradient method from full torque history.
    
    Args:
        torque_list: List of torque arrays, each shape (n_joints,)
        dt: Control timestep
    
    Returns:
        jerk_gradient: shape (T, n_joints) or None if insufficient data
    """
    if len(torque_list) < 2:
        return None
    
    try:
        torque_array = np.array(torque_list)  # shape (T, n_joints)
        # gradient along time axis (axis=0)
        jerk_gradient = np.gradient(torque_array, dt, axis=0, edge_order=2 if len(torque_array) >= 3 else 1)
        return jerk_gradient
    except Exception as e:
        logging.error(f"Error computing actuator jerk gradient: {e}")
        return None


def compute_basic_metrics(
    torque_list: List[np.ndarray],
    velocity_list: List[np.ndarray],
    acceleration_list: List[np.ndarray],
    kinematic_jerk_delta_list: List[np.ndarray],
    actuator_jerk_delta_list: List[np.ndarray],
    electric_energy_list: List[float],
    dt: float,
    output_dir: str,
    task_segment: str,
    episode_idx: int,
    suffix: str,
    debug: bool = False
) -> Dict[str, np.ndarray]:
    """
    Compute and save basic metrics as npy files.
    
    Args:
        torque_list: List of torque arrays
        velocity_list: List of velocity arrays
        acceleration_list: List of acceleration arrays
        kinematic_jerk_delta_list: List of kinematic jerk (DeltaBuffer delta method)
        actuator_jerk_delta_list: List of actuator jerk (DeltaBuffer delta method)
        electric_energy_list: List of electric energy values
        dt: Control timestep
        output_dir: Output directory path
        task_segment: Task segment name
        episode_idx: Episode index
        suffix: "success" or "failure"
    
    Returns:
        Dictionary with computed metrics
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = {}
    
    # Stack lists into arrays
    if len(torque_list) > 0:
        torque_array = np.array(torque_list)  # shape (T, n_joints)
        np.save(output_dir / f"torque_{task_segment}_{episode_idx}_{suffix}.npy", torque_array)
        metrics["torque"] = torque_array
        logging.info(f"Saved torque: shape {torque_array.shape}")
    
    if len(velocity_list) > 0:
        velocity_array = np.array(velocity_list)  # shape (T, n_joints)
        np.save(output_dir / f"velocity_{task_segment}_{episode_idx}_{suffix}.npy", velocity_array)
        metrics["velocity"] = velocity_array
        logging.info(f"Saved velocity: shape {velocity_array.shape}")
    
    if len(acceleration_list) > 0:
        acceleration_array = np.array(acceleration_list)  # shape (T, n_joints)
        np.save(output_dir / f"acceleration_{task_segment}_{episode_idx}_{suffix}.npy", acceleration_array)
        metrics["acceleration"] = acceleration_array
        logging.info(f"Saved acceleration: shape {acceleration_array.shape}")
    
    # Kinematic Jerk - DeltaBuffer delta method
    if len(kinematic_jerk_delta_list) > 0:
        kinematic_jerk_delta_array = np.array(kinematic_jerk_delta_list)  # shape (T, n_joints)
        np.save(output_dir / f"kinematic_jerk_delta_{task_segment}_{episode_idx}_{suffix}.npy", kinematic_jerk_delta_array)
        metrics["kinematic_jerk_delta"] = kinematic_jerk_delta_array
        logging.info(f"Saved kinematic_jerk_delta: shape {kinematic_jerk_delta_array.shape}")
    
    # Kinematic Jerk - Gradient method
    if len(acceleration_list) > 1:
        kinematic_jerk_gradient = compute_kinematic_jerk_gradient(acceleration_list, dt)
        if kinematic_jerk_gradient is not None:
            np.save(output_dir / f"kinematic_jerk_gradient_{task_segment}_{episode_idx}_{suffix}.npy", kinematic_jerk_gradient)
            metrics["kinematic_jerk_gradient"] = kinematic_jerk_gradient
            logging.info(f"Saved kinematic_jerk_gradient: shape {kinematic_jerk_gradient.shape}")
    
    # Actuator Jerk - DeltaBuffer delta method
    if len(actuator_jerk_delta_list) > 0:
        actuator_jerk_delta_array = np.array(actuator_jerk_delta_list)  # shape (T, n_joints)
        np.save(output_dir / f"actuator_jerk_delta_{task_segment}_{episode_idx}_{suffix}.npy", actuator_jerk_delta_array)
        metrics["actuator_jerk_delta"] = actuator_jerk_delta_array
        logging.info(f"Saved actuator_jerk_delta: shape {actuator_jerk_delta_array.shape}")
    
    # Actuator Jerk - Gradient method
    if len(torque_list) > 1:
        actuator_jerk_gradient = compute_actuator_jerk_gradient(torque_list, dt)
        if actuator_jerk_gradient is not None:
            np.save(output_dir / f"actuator_jerk_gradient_{task_segment}_{episode_idx}_{suffix}.npy", actuator_jerk_gradient)
            metrics["actuator_jerk_gradient"] = actuator_jerk_gradient
            logging.info(f"Saved actuator_jerk_gradient: shape {actuator_jerk_gradient.shape}")
    
    # Electric Energy (부호 유지)
    if len(electric_energy_list) > 0:
        electric_energy_array = np.array(electric_energy_list)  # shape (T,)
        np.save(output_dir / f"electric_energy_{task_segment}_{episode_idx}_{suffix}.npy", electric_energy_array)
        metrics["electric_energy"] = electric_energy_array
        logging.info(f"Saved electric_energy: shape {electric_energy_array.shape}")
        
        # 부호 분석
        draw_energy = np.sum(np.maximum(electric_energy_array, 0))
        regen_energy = np.sum(np.abs(np.minimum(electric_energy_array, 0)))
        net_energy = np.sum(electric_energy_array)
        total_absolute_energy = draw_energy + regen_energy  # 절댓값 합 (총 에너지 소비)
        logging.info(f"Energy Summary: draw={draw_energy:.6f}J, regen={regen_energy:.6f}J, net={net_energy:.6f}J, total_abs={total_absolute_energy:.6f}J")
    
    return metrics

