import os
import mne
import torch
import warnings
from preprocess import preprocess_base, apply_advanced_cleaning, build_master_epochs
from visualize import run_hypothesis_validation_tfr
from models import run_stage2_csp_svm, run_stage3_eegnet

warnings.filterwarnings("ignore")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"====== [본 실험 개시] 연산 디바이스: {device} ======")

    # --------------------------------------------------------------------------
    # 1. 본 실험 파라미터 (S001 ~ S050, 총 50명)
    # --------------------------------------------------------------------------
    subjects_to_test = list(range(1, 51))  # S001 ~ S050
    rm_runs = [3, 7, 11, 5, 9, 13]          # Real Movement 전체
    mi_runs = [4, 8, 12, 6, 10, 14]          # Motor Imagery 전체

    print(f" -> 총 {len(subjects_to_test)}명 피험자 데이터 로드 및 이원화 전처리 진행...")

    rm_simple_list, mi_simple_list = [], []
    rm_hard_list, mi_hard_list = [], []

    # --------------------------------------------------------------------------
    # 2. 전체 피험자 데이터 수집 및 전처리
    # --------------------------------------------------------------------------
    for sub in subjects_to_test:
        if sub % 5 == 0 or sub == 1:
            print(f"   [진행 상황] 피험자 S{sub:03d}/S050 처리 중...")

        # Real Movement (RM)
        for r in rm_runs:
            try:
                path = mne.datasets.eegbci.load_data(sub, r, verbose=False)[0]
                for hand in ['left', 'right']:
                    ep_base = preprocess_base(path, hand=hand)
                    rm_simple_list.append(ep_base)
                    rm_hard_list.append(apply_advanced_cleaning(ep_base))
            except Exception:
                continue

        # Motor Imagery (MI)
        for r in mi_runs:
            try:
                path = mne.datasets.eegbci.load_data(sub, r, verbose=False)[0]
                for hand in ['left', 'right']:
                    ep_base = preprocess_base(path, hand=hand)
                    mi_simple_list.append(ep_base)
                    mi_hard_list.append(apply_advanced_cleaning(ep_base))
            except Exception:
                continue

    # --------------------------------------------------------------------------
    # 3. Master Epochs 통합
    # --------------------------------------------------------------------------
    print("\n -> 전체 피험자 Master Epochs 생성 및 데이터 통합 중...")
    epochs_simple = build_master_epochs(rm_simple_list, mi_simple_list)
    epochs_hard = build_master_epochs(rm_hard_list, mi_hard_list)

    print(f" -> [완료] 간단 전처리 총 에포크: {len(epochs_simple)}개")
    print(f" -> [완료] 하드 전처리 총 에포크: {len(epochs_hard)}개\n")

    # --------------------------------------------------------------------------
    # 4. [실험 1] 간단 전처리 (Base Filtering) ML vs DL 평가
    # --------------------------------------------------------------------------
    print("="*75)
    print(" [실험 1] 간단 전처리 (Base Filtering) 데이터 평가 ")
    print("="*75)
    print("\n[ML] CSP + SVM Train/Test:")
    run_stage2_csp_svm(epochs_simple)
    
    print("\n[DL] PyTorch EEGNet Train/Test:")
    run_stage3_eegnet(epochs_simple)

    # --------------------------------------------------------------------------
    # 5. [실험 2] 하드 전처리 (ICA Artifact Cleaning) ML vs DL 평가
    # --------------------------------------------------------------------------
    print("\n\n" + "="*75)
    print(" [실험 2] 하드 전처리 (ICA Cleaning) 데이터 평가 ")
    print("="*75)
    print("\n[ML] CSP + SVM Train/Test:")
    run_stage2_csp_svm(epochs_hard)
    
    print("\n[DL] PyTorch EEGNet Train/Test:")
    run_stage3_eegnet(epochs_hard)

    # --------------------------------------------------------------------------
    # 6. [시각화] TFR 분석 검증
    # --------------------------------------------------------------------------
    print("\n -> 하드 전처리 데이터 기반 TFR 시각화 출력 중...")
    run_hypothesis_validation_tfr(epochs_hard)
    print("\n====== 본 실험 전체 과정 완료 ======")

if __name__ == "__main__":
    main()
