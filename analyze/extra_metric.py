"""
Extra Metrics 계산 및 저장

- basic_metric에서 저장된 npy 파일 읽기
- 각 metric별 통계 계산 (평균, 최대, 최소, 합)
- Electric Energy 부호 처리 (draw, regen, net)
- Life Ratio 계산 (rainflow-miner)
- _iqr_scale 적용 (추가 통계)
- CSV 형식으로 저장
"""

import numpy as np
import pathlib
import logging
import csv
from typing import Optional, Dict, Tuple, List

try:
    import rainflow
    HAS_RAINFLOW = True
except ImportError:
    HAS_RAINFLOW = False
    logging.warning("rainflow package not installed. Life ratio calculation will be skipped.")


def _iqr_scale(arr: np.ndarray, eps: float = 1e-12) -> Optional[Tuple[float, float]]:
    """
    Calculate IQR-based scaling (Q1, IQR).
    
    Args:
        arr: numpy array
        eps: small epsilon to avoid division by zero
    
    Returns:
        (Q1, IQR) tuple or None if insufficient data
    """
    if len(arr) == 0:
        return None
    try:
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = max(q3 - q1, eps)
        return q1, iqr
    except Exception as e:
        logging.warning(f"Error computing IQR: {e}")
        return None


def compute_statistics(arr: np.ndarray, metric_name: str = "metric") -> Dict[str, float]:
    """
    Compute basic statistics for an array.
    
    Args:
        arr: numpy array (1D or 2D)
        metric_name: name of metric (for logging)
    
    Returns:
        Dictionary with statistics
    """
    stats = {}
    
    try:
        if arr.ndim == 1:
            # 1D array
            stats["mean"] = float(np.mean(arr))
            stats["max"] = float(np.max(arr))
            stats["min"] = float(np.min(arr))
            stats["sum"] = float(np.sum(arr))
            stats["std"] = float(np.std(arr))
        else:
            # 2D array - compute per joint, then aggregate
            stats["mean"] = float(np.mean(arr))
            stats["max"] = float(np.max(arr))
            stats["min"] = float(np.min(arr))
            stats["sum"] = float(np.sum(arr))
            stats["std"] = float(np.std(arr))
            
            # Per-joint statistics
            for joint_idx in range(arr.shape[1]):
                joint_data = arr[:, joint_idx]
                stats[f"joint_{joint_idx}_mean"] = float(np.mean(joint_data))
                stats[f"joint_{joint_idx}_max"] = float(np.max(joint_data))
                stats[f"joint_{joint_idx}_min"] = float(np.min(joint_data))
    
    except Exception as e:
        logging.error(f"Error computing statistics for {metric_name}: {e}")
    
    return stats




def compute_extra_metrics(
    npy_dir: str,
    task_segment: str,
    episode_idx: int,
    suffix: str,
    output_dir: str
) -> Dict[str, Dict[str, float]]:
    """
    Compute extra metrics from basic_metric npy files.
    
    Args:
        npy_dir: directory containing npy files
        task_segment: task segment name
        episode_idx: episode index
        suffix: "success" or "failure"
        output_dir: output directory for csv
    
    Returns:
        Dictionary with all computed metrics
    """
    npy_dir = pathlib.Path(npy_dir)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = {}
    
    # Load npy files
    torque_avg_file = npy_dir / f"torque_{task_segment}_{episode_idx}_{suffix}.npy"  # Average torque (for energy)
    torque_current_file = npy_dir / f"torque_current_{task_segment}_{episode_idx}_{suffix}.npy"  # Current torque (for life ratio)
    velocity_file = npy_dir / f"velocity_{task_segment}_{episode_idx}_{suffix}.npy"
    acceleration_file = npy_dir / f"acceleration_{task_segment}_{episode_idx}_{suffix}.npy"
    kinematic_jerk_delta_file = npy_dir / f"kinematic_jerk_delta_{task_segment}_{episode_idx}_{suffix}.npy"
    kinematic_jerk_gradient_file = npy_dir / f"kinematic_jerk_gradient_{task_segment}_{episode_idx}_{suffix}.npy"
    actuator_jerk_delta_file = npy_dir / f"actuator_jerk_delta_{task_segment}_{episode_idx}_{suffix}.npy"
    actuator_jerk_gradient_file = npy_dir / f"actuator_jerk_gradient_{task_segment}_{episode_idx}_{suffix}.npy"
    electric_energy_file = npy_dir / f"electric_energy_{task_segment}_{episode_idx}_{suffix}.npy"
    
    # Load and compute statistics for each metric
    if torque_avg_file.exists():
        torque_avg_array = np.load(torque_avg_file)
        metrics["torque_average"] = compute_statistics(torque_avg_array, "torque_average")
    
    # Load current torque for statistics (life ratio calculation moved to compare_metrics.py)
    if torque_current_file.exists():
        torque_current_array = np.load(torque_current_file)
        metrics["torque_current"] = compute_statistics(torque_current_array, "torque_current")
    
    if velocity_file.exists():
        velocity_array = np.load(velocity_file)
        metrics["velocity"] = compute_statistics(velocity_array, "velocity")
    
    if acceleration_file.exists():
        acceleration_array = np.load(acceleration_file)
        metrics["acceleration"] = compute_statistics(acceleration_array, "acceleration")
    
    if kinematic_jerk_delta_file.exists():
        kinematic_jerk_delta_array = np.load(kinematic_jerk_delta_file)
        metrics["kinematic_jerk_delta"] = compute_statistics(kinematic_jerk_delta_array, "kinematic_jerk_delta")
    
    if kinematic_jerk_gradient_file.exists():
        kinematic_jerk_gradient_array = np.load(kinematic_jerk_gradient_file)
        metrics["kinematic_jerk_gradient"] = compute_statistics(kinematic_jerk_gradient_array, "kinematic_jerk_gradient")
    
    if actuator_jerk_delta_file.exists():
        actuator_jerk_delta_array = np.load(actuator_jerk_delta_file)
        metrics["actuator_jerk_delta"] = compute_statistics(actuator_jerk_delta_array, "actuator_jerk_delta")
    
    if actuator_jerk_gradient_file.exists():
        actuator_jerk_gradient_array = np.load(actuator_jerk_gradient_file)
        metrics["actuator_jerk_gradient"] = compute_statistics(actuator_jerk_gradient_array, "actuator_jerk_gradient")
    
    # Electric Energy - separate by draw/regen/net
    if electric_energy_file.exists():
        electric_energy_array = np.load(electric_energy_file)  # shape (T,)
        
        # Separate positive (draw) and negative (regen) energy
        draw_energy = np.maximum(electric_energy_array, 0.0)  # Only positive
        regen_energy = np.abs(np.minimum(electric_energy_array, 0.0))  # Absolute value of negative
        net_energy = electric_energy_array  # Keep sign
        
        metrics["energy_draw"] = compute_statistics(draw_energy, "energy_draw")
        metrics["energy_regen"] = compute_statistics(regen_energy, "energy_regen")
        metrics["energy_net"] = compute_statistics(net_energy, "energy_net")
        
        # Summary
        total_draw = float(np.sum(draw_energy))
        total_regen = float(np.sum(regen_energy))
        total_net = float(np.sum(net_energy))
        
        logging.info(f"Energy Summary: draw={total_draw:.6f}J, regen={total_regen:.6f}J, net={total_net:.6f}J")
    
    return metrics


def save_metrics_to_csv(
    all_metrics: Dict[str, Dict[str, float]],
    task_segment: str,
    episode_idx: int,
    suffix: str,
    output_dir: str,
    task_description: str = ""
) -> str:
    """
    Save metrics to CSV file in timeseries subdirectory.
    
    Args:
        all_metrics: dictionary of all metrics (from compute_extra_metrics)
        task_segment: task segment name
        episode_idx: episode index
        suffix: "success" or "failure"
        output_dir: output directory (parent directory)
        task_description: task description for logging
    
    Returns:
        Path to saved CSV file
    """
    output_dir = pathlib.Path(output_dir)
    timeseries_dir = output_dir / "timeseries"
    timeseries_dir.mkdir(parents=True, exist_ok=True)
    
    csv_file = timeseries_dir / f"metrics_{task_segment}_{episode_idx}_{suffix}.csv"
    
    try:
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(["metric_type", "metric_name", "value"])
            
            # Data
            for metric_type, metric_dict in all_metrics.items():
                for metric_name, value in metric_dict.items():
                    writer.writerow([metric_type, metric_name, value])
        
        logging.info(f"Saved metrics to {csv_file}")
        return str(csv_file)
    
    except Exception as e:
        logging.error(f"Failed to save metrics to CSV: {e}")
        return ""


def save_summary_csv(
    all_episode_metrics: List[Dict],
    output_file: str,
    task_suite_name: str,
    base_name: str
) -> str:
    """
    Save all episodes' metrics to a single summary CSV file.
    
    Args:
        all_episode_metrics: List of dicts, each containing:
            {
                "task": task_name,
                "episode": episode_idx,
                "success": 0/1,
                "dt": dt_value,
                "metrics": {all flattened metrics}
            }
        output_file: path to output CSV file
        task_suite_name: task suite name for logging
        base_name: base name for logging
    
    Returns:
        Path to saved CSV file
    """
    if not all_episode_metrics:
        logging.warning("No episode metrics to save")
        return ""
    
    output_file = pathlib.Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Collect all column names
        all_columns = set()
        all_columns.add("task")
        all_columns.add("episode")
        all_columns.add("success")
        all_columns.add("dt")
        
        for episode_dict in all_episode_metrics:
            if "metrics" in episode_dict:
                for metric_type, metric_dict in episode_dict["metrics"].items():
                    for metric_name, value in metric_dict.items():
                        # Create column name like "torque_average.mean" or "torque_average.joint_0_mean"
                        col_name = f"{metric_type}.{metric_name}"
                        all_columns.add(col_name)
        
        # Sort columns for consistent ordering
        sorted_columns = ["task", "episode", "success", "dt"] + sorted([c for c in all_columns if c not in ["task", "episode", "success", "dt"]])
        
        # Write CSV
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted_columns)
            writer.writeheader()
            
            for episode_dict in all_episode_metrics:
                row = {
                    "task": episode_dict.get("task", ""),
                    "episode": episode_dict.get("episode", ""),
                    "success": episode_dict.get("success", ""),
                    "dt": episode_dict.get("dt", ""),
                }
                
                # Flatten metrics
                if "metrics" in episode_dict:
                    for metric_type, metric_dict in episode_dict["metrics"].items():
                        for metric_name, value in metric_dict.items():
                            col_name = f"{metric_type}.{metric_name}"
                            row[col_name] = value
                
                writer.writerow(row)
        
        logging.info(f"✅ Summary CSV saved to {output_file}")
        logging.info(f"   Rows: {len(all_episode_metrics)}, Columns: {len(sorted_columns)}")
        return str(output_file)
    
    except Exception as e:
        logging.error(f"Failed to save summary CSV: {e}")
        return ""
