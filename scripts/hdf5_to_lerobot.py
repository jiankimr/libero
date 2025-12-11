#!/usr/bin/env python3
"""
HDF5 to LeRobot format converter.

Converts rollout data saved by eval.py (in LIBERO HDF5 format) to LeRobot format
compatible with GR00T training.

Usage:
    python hdf5_to_lerobot.py \
        --input_dir ./rollouts/rollout_libero_10_xxx \
        --output_dir ./data/rollout_libero_10_xxx \
        --fps 20

Input format (HDF5):
    rollout_task_description.hdf5
    └── data/
        ├── demo_0/
        │   ├── obs/
        │   │   ├── agentview_rgb (N, H, W, 3)
        │   │   ├── eye_in_hand_rgb (N, H, W, 3)
        │   │   ├── ee_states (N, 6)
        │   │   ├── gripper_states (N, 2)
        │   │   └── joint_states (N, n_joints)
        │   ├── actions (N, 7)
        │   ├── rewards (N,)
        │   └── dones (N,)
        └── demo_1/...

Output format (LeRobot):
    output_dir/
    ├── meta/
    │   ├── info.json
    │   ├── modality.json
    │   ├── tasks.jsonl
    │   └── episodes.jsonl
    ├── data/
    │   └── chunk-000/
    │       ├── episode_000000.parquet
    │       └── ...
    └── videos/
        └── chunk-000/
            ├── observation.images.image/
            │   ├── episode_000000.mp4
            │   └── ...
            └── observation.images.wrist_image/
                ├── episode_000000.mp4
                └── ...
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from glob import glob

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# Try to import video encoding libraries
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("Warning: imageio not available. Videos will not be encoded.")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def create_modality_json(output_dir: Path) -> None:
    """Create modality.json matching the LIBERO format used for GR00T training."""
    modality = {
        "state": {
            "x": {"start": 0, "end": 1},
            "y": {"start": 1, "end": 2},
            "z": {"start": 2, "end": 3},
            "axis_angle1": {"start": 3, "end": 4, "rotation_type": "axis_angle"},
            "axis_angle2": {"start": 4, "end": 5, "rotation_type": "axis_angle"},
            "axis_angle3": {"start": 5, "end": 6, "rotation_type": "axis_angle"},
            "gripper_left_finger": {"start": 6, "end": 7},
            "gripper_right_finger": {"start": 7, "end": 8}
        },
        "action": {
            "x": {"start": 0, "end": 1},
            "y": {"start": 1, "end": 2},
            "z": {"start": 2, "end": 3},
            "axis_angle1": {"start": 3, "end": 4, "rotation_type": "axis_angle"},
            "axis_angle2": {"start": 4, "end": 5, "rotation_type": "axis_angle"},
            "axis_angle3": {"start": 5, "end": 6, "rotation_type": "axis_angle"},
            "gripper": {"start": 6, "end": 7}
        },
        "video": {
            "image": {"original_key": "observation.images.image"},
            "wrist_image": {"original_key": "observation.images.wrist_image"}
        },
        "annotation": {
            "task_index": {},
            "human.action.task_description": {"original_key": "task_index"}
        }
    }
    
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=4)
    
    print(f"✅ Created modality.json")


def create_info_json(
    output_dir: Path,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    fps: int,
    image_size: int = 256,
) -> None:
    """Create info.json with dataset metadata."""
    info = {
        "codebase_version": "v2.1",
        "robot_type": "franka",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes * 2,  # agentview + wrist
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {
            "train": f"0:{total_episodes}"
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.wrist_image": {
                "dtype": "video",
                "shape": [image_size, image_size, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.height": image_size,
                    "video.width": image_size,
                    "video.codec": "mpeg4",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": fps,
                    "video.channels": 3,
                    "has_audio": False
                }
            },
            "observation.images.image": {
                "dtype": "video",
                "shape": [image_size, image_size, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.height": image_size,
                    "video.width": image_size,
                    "video.codec": "mpeg4",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": fps,
                    "video.channels": 3,
                    "has_audio": False
                }
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [8],
                "names": {
                    "motors": ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper", "gripper"]
                }
            },
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": {
                    "motors": ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper"]
                }
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None}
        }
    }
    
    meta_dir = output_dir / "meta"
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)
    
    print(f"✅ Created info.json")


def encode_video(images: np.ndarray, output_path: Path, fps: int) -> bool:
    """Encode images to MP4 video."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if HAS_IMAGEIO:
        try:
            # Use simpler options for compatibility
            imageio.mimwrite(str(output_path), images, fps=fps)
            return True
        except Exception as e:
            print(f"Warning: Failed to encode video with imageio: {e}")
    
    if HAS_CV2:
        try:
            h, w = images[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
            for img in images:
                out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            out.release()
            return True
        except Exception as e:
            print(f"Warning: Failed to encode video with cv2: {e}")
    
    return False


def convert_hdf5_to_lerobot(
    input_dir: str,
    output_dir: str,
    fps: int = 20,
    success_only: bool = False,
) -> None:
    """
    Convert HDF5 rollout files to LeRobot format.
    
    Args:
        input_dir: Directory containing HDF5 rollout files
        output_dir: Output directory for LeRobot dataset
        fps: Frames per second for video encoding
        success_only: If True, only include successful episodes
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find all HDF5 files
    hdf5_files = sorted(glob(str(input_path / "*.hdf5")))
    if not hdf5_files:
        print(f"❌ No HDF5 files found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(hdf5_files)} HDF5 files")
    
    # Collect all episodes and tasks
    all_episodes = []
    all_tasks = {}
    global_episode_idx = 0
    global_frame_idx = 0
    
    # Create output directories
    data_dir = output_path / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    video_agentview_dir = output_path / "videos" / "chunk-000" / "observation.images.image"
    video_wrist_dir = output_path / "videos" / "chunk-000" / "observation.images.wrist_image"
    video_agentview_dir.mkdir(parents=True, exist_ok=True)
    video_wrist_dir.mkdir(parents=True, exist_ok=True)
    
    print("Processing HDF5 files...")
    
    for hdf5_file in tqdm(hdf5_files):
        with h5py.File(hdf5_file, "r") as f:
            data_grp = f["data"]
            
            # Get list of demos
            demo_keys = sorted([k for k in data_grp.keys() if k.startswith("demo_")])
            
            for demo_key in demo_keys:
                demo_grp = data_grp[demo_key]
                
                # Check success filter
                success = demo_grp.attrs.get("success", True)
                if success_only and not success:
                    continue
                
                # Get task description
                task_description = demo_grp.attrs.get("task_description", "unknown_task")
                if isinstance(task_description, bytes):
                    task_description = task_description.decode("utf-8")
                
                # Register task
                if task_description not in all_tasks:
                    all_tasks[task_description] = len(all_tasks)
                task_idx = all_tasks[task_description]
                
                # Load data
                obs_grp = demo_grp["obs"]
                agentview_images = obs_grp["agentview_rgb"][:]
                eye_in_hand_images = obs_grp["eye_in_hand_rgb"][:]
                ee_states = obs_grp["ee_states"][:]  # (N, 6)
                gripper_states = obs_grp["gripper_states"][:]  # (N, 2)
                actions = demo_grp["actions"][:]  # (N, 7)
                
                num_frames = len(actions)
                
                # Create observation.state (8D: ee_pos (3) + axis_angle (3) + gripper (2))
                observation_states = np.concatenate([ee_states, gripper_states], axis=1).astype(np.float32)
                
                # Create parquet data
                parquet_data = []
                for frame_idx in range(num_frames):
                    row = {
                        "observation.state": observation_states[frame_idx],
                        "action": actions[frame_idx].astype(np.float32),
                        "timestamp": np.float32(frame_idx / fps),
                        "frame_index": np.int64(frame_idx),
                        "episode_index": np.int64(global_episode_idx),
                        "index": np.int64(global_frame_idx),
                        "task_index": np.int64(task_idx),
                    }
                    parquet_data.append(row)
                    global_frame_idx += 1
                
                # Save parquet
                df = pd.DataFrame(parquet_data)
                parquet_path = data_dir / f"episode_{global_episode_idx:06d}.parquet"
                df.to_parquet(parquet_path, engine="pyarrow")
                
                # Encode videos
                agentview_video_path = video_agentview_dir / f"episode_{global_episode_idx:06d}.mp4"
                wrist_video_path = video_wrist_dir / f"episode_{global_episode_idx:06d}.mp4"
                
                encode_video(agentview_images, agentview_video_path, fps)
                encode_video(eye_in_hand_images, wrist_video_path, fps)
                
                # Record episode info
                all_episodes.append({
                    "episode_index": global_episode_idx,
                    "tasks": [task_description],
                    "length": num_frames,
                })
                
                global_episode_idx += 1
    
    # Create metadata files
    create_modality_json(output_path)
    
    # Create tasks.jsonl
    meta_dir = output_path / "meta"
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for task, idx in sorted(all_tasks.items(), key=lambda x: x[1]):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")
    print(f"✅ Created tasks.jsonl ({len(all_tasks)} tasks)")
    
    # Create episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in all_episodes:
            f.write(json.dumps(ep) + "\n")
    print(f"✅ Created episodes.jsonl ({len(all_episodes)} episodes)")
    
    # Create info.json
    total_frames = sum(ep["length"] for ep in all_episodes)
    image_size = agentview_images.shape[1] if len(all_episodes) > 0 else 256
    create_info_json(
        output_path,
        total_episodes=len(all_episodes),
        total_frames=total_frames,
        total_tasks=len(all_tasks),
        fps=fps,
        image_size=image_size,
    )
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ Conversion complete!")
    print(f"{'='*60}")
    print(f"  Episodes: {len(all_episodes)}")
    print(f"  Tasks: {len(all_tasks)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Output: {output_path}")
    print(f"\nTo use this dataset for training:")
    print(f"  python scripts/gr00t_finetune.py \\")
    print(f"      --dataset_path {output_path} \\")
    print(f"      --data_config libero \\")
    print(f"      ...")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HDF5 rollout data to LeRobot format for GR00T training"
    )
    parser.add_argument(
        "--input_dir", "-i",
        type=str,
        required=True,
        help="Directory containing HDF5 rollout files"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        required=True,
        help="Output directory for LeRobot dataset"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frames per second for video encoding (default: 20)"
    )
    parser.add_argument(
        "--success_only",
        action="store_true",
        help="Only include successful episodes"
    )
    
    args = parser.parse_args()
    
    convert_hdf5_to_lerobot(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fps=args.fps,
        success_only=args.success_only,
    )


if __name__ == "__main__":
    main()

