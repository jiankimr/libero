# Rollout Dataset Collection Guide

Evaluation 중 rollout 데이터를 수집하여 GR00T 학습용 데이터셋을 생성하는 가이드입니다.

## 개요

```
eval.py (rollout 수집)
    ↓
HDF5 파일 (LIBERO 형식)
    ↓
hdf5_to_lerobot.py (변환)
    ↓
LeRobot 데이터셋 (GR00T 학습용)
```

## Step 1: Rollout 데이터 수집

### 옵션 설명

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--args.save-rollouts` | Rollout 데이터 수집 활성화 | False |
| `--args.rollout-save-path` | 저장 경로 (선택) | `./rollouts/rollout_{task_suite}_{timestamp}/` |

### 실행 예시

```bash
cd /workspace/repos/rsec/LIBERO
source /opt/miniforge3/etc/profile.d/conda.sh && conda activate libero

# EGL 문제 해결을 위한 환경 변수 (필수)
export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=osmesa

# Rollout 수집과 함께 eval 실행
python eval.py \
    --args.host localhost \
    --args.port 5555 \
    --args.task-suite-name libero_10 \
    --args.num-trials-per-task 20 \
    --args.save-rollouts \
    --args.output-suffix my_rollout
```

### 저장되는 데이터

```
./rollouts/rollout_libero_10_{timestamp}_{suffix}/
├── rollout_task00_{task_description}.hdf5
├── rollout_task01_{task_description}.hdf5
├── ...
└── rollout_task09_{task_description}.hdf5
```

각 HDF5 파일 구조:
```
data/
├── demo_0/
│   ├── obs/
│   │   ├── agentview_rgb (N, 256, 256, 3) - 메인 카메라 이미지
│   │   ├── eye_in_hand_rgb (N, 256, 256, 3) - 손목 카메라 이미지
│   │   ├── ee_states (N, 6) - end-effector 위치(3) + axis_angle(3)
│   │   ├── gripper_states (N, 2) - 그리퍼 상태
│   │   └── joint_states (N, 7) - 관절 상태
│   ├── actions (N, 7) - 액션
│   ├── rewards (N,) - 보상 (마지막 스텝에 1 if success)
│   └── dones (N,) - 종료 플래그
│   └── attrs: success, task_description, num_samples
├── demo_1/
└── ...
```

## Step 2: LeRobot 형식으로 변환

```bash
cd /workspace/repos/rsec/LIBERO
source /opt/miniforge3/etc/profile.d/conda.sh && conda activate gr00t

python scripts/hdf5_to_lerobot.py \
    --input_dir ./rollouts/rollout_libero_10_{timestamp}_{suffix} \
    --output_dir ./data/my_rollout_dataset \
    --fps 20 \
    --success_only  # (선택) 성공한 에피소드만 포함
```

### 옵션

| 옵션 | 설명 |
|------|------|
| `--input_dir`, `-i` | HDF5 파일이 있는 디렉토리 |
| `--output_dir`, `-o` | LeRobot 데이터셋 출력 경로 |
| `--fps` | 비디오 FPS (기본: 20) |
| `--success_only` | 성공한 에피소드만 포함 |

### 출력 구조

```
./data/my_rollout_dataset/
├── meta/
│   ├── info.json
│   ├── modality.json
│   ├── tasks.jsonl
│   └── episodes.jsonl
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
└── videos/
    └── chunk-000/
        ├── observation.images.image/
        │   ├── episode_000000.mp4
        │   └── ...
        └── observation.images.wrist_image/
            ├── episode_000000.mp4
            └── ...
```

## Step 3: GR00T 학습

```bash
cd /workspace/repos/rsec/Isaac-GR00T
source /opt/miniforge3/etc/profile.d/conda.sh && conda activate gr00t

python scripts/gr00t_finetune.py \
    --dataset_path /workspace/repos/rsec/LIBERO/data/my_rollout_dataset \
    --data_config libero \
    --output_dir /workspace/outputs/gr00t_rollout \
    --batch_size 32 \
    --max_steps 10000 \
    --save_steps 1000 \
    --num_gpus 1
```

## 데이터 형식 호환성

변환된 데이터셋은 기존 LIBERO 학습 데이터셋과 100% 호환됩니다:

| 필드 | 형식 | 설명 |
|------|------|------|
| `observation.state` | float32, (8,) | ee_pos(3) + axis_angle(3) + gripper(2) |
| `action` | float32, (7,) | ee_pos_delta(3) + rot_delta(3) + gripper(1) |
| `timestamp` | float32 | 시간 (초) |
| `frame_index` | int64 | 에피소드 내 프레임 인덱스 |
| `episode_index` | int64 | 에피소드 인덱스 |
| `task_index` | int64 | Task 인덱스 |

## 성공/실패 필터링

- HDF5 파일에 각 에피소드의 `success` 속성이 저장됨
- `--success_only` 옵션으로 성공한 에피소드만 학습 데이터셋에 포함 가능
- 실패 에피소드도 포함하여 robust한 모델 학습 가능

## 문제 해결

### EGL 오류
```bash
export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=osmesa
```

### 비디오 백엔드
- 학습 시 `--video_backend torchcodec` 사용 (기본값)
- 변환된 비디오는 mpeg4 코덱으로 인코딩됨

### 파일명 충돌
- 저장 경로에 timestamp가 자동 포함됨
- 각 task별 파일명에 task_id가 포함됨

## 검증 완료

| 항목 | 상태 |
|------|------|
| HDF5 저장 | ✅ |
| LeRobot 변환 | ✅ |
| 비디오 인코딩 | ✅ (mpeg4) |
| 데이터셋 로딩 | ✅ (torchcodec) |
| 학습 가능 여부 | ✅ |

