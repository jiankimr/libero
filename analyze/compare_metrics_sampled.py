"""
16스텝 간격으로 샘플링한 데이터로 비교 수행하는 스크립트

NPY 기반 피로 손상 분석 + CSV 기반 물리량(에너지, 토크, 운동학, Jerk 등) 비교를 함께 수행합니다.
※ NPY 데이터는 다운샘플링 후 비교, CSV 데이터는 원본 그대로 비교합니다.
"""
import argparse
import pathlib
import tempfile
import shutil
import numpy as np
from compare_metrics import compare_npy_metrics, compare_csv_metrics, find_csv_file

def downsample_npy_files(src_dir: pathlib.Path, dst_dir: pathlib.Path, step: int = 16, task_filter: str = None):
    """
    NPY 파일들을 다운샘플링하여 새 디렉토리에 저장
    
    Args:
        src_dir: 원본 NPY 파일이 있는 디렉토리
        dst_dir: 다운샘플링된 파일을 저장할 디렉토리
        step: 샘플링 간격 (기본값: 16)
        task_filter: 태스크 필터 문자열 (예: "pick_up" → 해당 문자열이 포함된 파일만 처리)
    
    Returns:
        다운샘플링된 파일 개수
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # torque_current_*.npy 파일만 다운샘플링
    npy_files = list(src_dir.glob("torque_current_*_success.npy"))
    
    # 태스크 필터 적용
    if task_filter:
        npy_files = [f for f in npy_files if task_filter in f.name]
        print(f"📁 발견된 torque_current_*_success.npy 파일 (태스크 필터 '{task_filter}'): {len(npy_files)}개")
    else:
        print(f"📁 발견된 torque_current_*_success.npy 파일: {len(npy_files)}개")
    
    processed_count = 0
    for src_file in npy_files:
        try:
            data = np.load(src_file)
            # 16스텝마다 샘플링 (첫번째 값부터)
            sampled_data = data[::step]
            
            dst_file = dst_dir / src_file.name
            np.save(dst_file, sampled_data)
            processed_count += 1
            
        except Exception as e:
            print(f"⚠️ {src_file.name} 처리 실패: {e}")
    
    print(f"✅ {processed_count}개 파일 다운샘플링 완료 → {dst_dir}")
    return processed_count

def main():
    parser = argparse.ArgumentParser(
        description="N스텝 간격 샘플링 후 비교 (NPY 피로 손상 + CSV 물리량)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python compare_metrics_sampled.py \\
    --before_dir ./analysis/analysis_libero_10_20251202_170035_noise_00000_clean \\
    --after_dir ./analysis/analysis_libero_10_20251206_052507_noise_01000_dim_action.x_rollout_noise_01_actionx \\
    --step 4 --m 3.0

※ NPY 데이터는 다운샘플링 후 피로 손상 분석
※ CSV 데이터는 원본 그대로 물리량(에너지, 토크, 운동학, Jerk 등) 비교
        """
    )
    parser.add_argument("--before_dir", type=str, required=True,
                        help="Before (baseline) 분석 디렉토리")
    parser.add_argument("--after_dir", type=str, required=True,
                        help="After (noise) 분석 디렉토리")
    parser.add_argument("--step", type=int, default=16, 
                        help="샘플링 간격 (기본: 16)")
    parser.add_argument("--m", type=float, default=3.0, 
                        help="Basquin 지수 (기본: 3.0)")
    parser.add_argument("--task_filter", type=str, default=None, 
                        help="태스크 필터 (예: 'pick_up' → 해당 문자열 포함 파일만 비교)")
    parser.add_argument("--skip_csv", action="store_true", default=False,
                        help="CSV 기반 물리량 비교 건너뛰기 (기본: False)")
    parser.add_argument("--output", type=str, default=None,
                        help="출력 파일 경로 (기본: 자동 생성)")
    
    args = parser.parse_args()
    
    before_dir = pathlib.Path(args.before_dir)
    after_dir = pathlib.Path(args.after_dir)
    
    # 디렉토리 존재 확인
    if not before_dir.exists():
        print(f"❌ Before 디렉토리를 찾을 수 없습니다: {before_dir}")
        exit(1)
    
    if not after_dir.exists():
        print(f"❌ After 디렉토리를 찾을 수 없습니다: {after_dir}")
        exit(1)
    
    print(f"\n{'='*80}")
    print(f"🔄 {args.step}스텝 간격 샘플링 비교 시작")
    print(f"{'='*80}")
    print(f"Before dir: {before_dir}")
    print(f"After dir:  {after_dir}")
    print(f"샘플링 간격: {args.step}스텝 (원본 데이터의 1/{args.step})")
    if args.task_filter:
        print(f"🎯 태스크 필터: '{args.task_filter}'")
    
    all_output = []
    
    # 임시 디렉토리에 다운샘플링된 데이터 저장
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        sampled_before = tmpdir / "before_sampled"
        sampled_after = tmpdir / "after_sampled"
        
        print(f"\n📦 Step 1/3: NPY 파일 다운샘플링...")
        print(f"   Before → {sampled_before}")
        downsample_npy_files(before_dir, sampled_before, step=args.step, task_filter=args.task_filter)
        
        print(f"\n   After → {sampled_after}")
        downsample_npy_files(after_dir, sampled_after, step=args.step, task_filter=args.task_filter)
        
        # NPY 비교 실행
        print(f"\n📊 Step 2/3: 다운샘플링된 데이터로 피로 손상 비교...")
        npy_output, npy_results = compare_npy_metrics(
            sampled_before, sampled_after,
            m=args.m,
            filter_unreliable=False,
        )
        
        all_output.append(npy_output)
        print(npy_output)
    
    # CSV 기반 물리량 비교 (원본 데이터 사용, 다운샘플링 없음)
    if not args.skip_csv:
        print(f"\n{'='*80}")
        print(f"📊 Step 3/3: CSV 기반 물리량 비교 (에너지, 토크, 운동학, Jerk 등)...")
        print(f"{'='*80}")
        print(f"※ CSV 비교는 원본 데이터를 사용합니다 (다운샘플링 없음)")
        
        before_csv = find_csv_file(before_dir)
        after_csv = find_csv_file(after_dir)
        
        if before_csv and after_csv:
            print(f"   Before CSV: {before_csv.name}")
            print(f"   After CSV:  {after_csv.name}")
            
            csv_output, csv_results = compare_csv_metrics(before_csv, after_csv)
            all_output.append(csv_output)
            print(csv_output)
        else:
            missing_msg = []
            if not before_csv:
                missing_msg.append(f"Before CSV not found in {before_dir}")
            if not after_csv:
                missing_msg.append(f"After CSV not found in {after_dir}")
            
            csv_warning = f"\n⚠️  CSV 파일을 찾을 수 없습니다: {', '.join(missing_msg)}"
            csv_warning += "\n   CSV 기반 물리량 비교를 건너뜁니다."
            all_output.append(csv_warning)
            print(csv_warning)
    else:
        print(f"\n⏭️  CSV 기반 물리량 비교 건너뜀 (--skip_csv 옵션)")
    
    # 결과 저장
    if args.output:
        output_file = pathlib.Path(args.output)
    elif args.task_filter:
        output_file = pathlib.Path("./results") / f"comparison_sampled_{args.step}step_{args.task_filter}.txt"
    else:
        output_file = pathlib.Path("./results") / f"comparison_sampled_{args.step}step.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    final_output = "\n".join(all_output)
    
    with open(output_file, "w") as f:
        f.write(f"# {args.step}스텝 간격 샘플링 비교 결과\n")
        f.write(f"# Before: {before_dir}\n")
        f.write(f"# After: {after_dir}\n")
        if args.task_filter:
            f.write(f"# Task Filter: {args.task_filter}\n")
        f.write(f"# Basquin 지수 (m): {args.m}\n")
        f.write(f"#\n")
        f.write(f"# ※ NPY 피로 손상: {args.step}스텝 다운샘플링 후 분석\n")
        f.write(f"# ※ CSV 물리량: 원본 데이터 그대로 비교\n")
        f.write("\n")
        f.write(final_output)
    
    file_size = output_file.stat().st_size
    print(f"\n{'='*80}")
    print(f"✅ 전체 비교 결과 저장 완료: {output_file}")
    print(f"   파일 크기: {file_size:,} bytes")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

