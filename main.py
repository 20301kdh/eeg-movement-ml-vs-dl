import mne
import warnings
from preprocess import preprocess_base, apply_advanced_cleaning, build_master_epochs
from visualize import run_hypothesis_validation_tfr
from models import run_stage2_csp_svm, run_stage3_eegnet

warnings.filterwarnings("ignore")

def main():
    # --------------------------------------------------------------------------
    # 1. 데이터 설정 (원본 코드의 Run 번호 유지)
    # --------------------------------------------------------------------------
    rm_runs = [3, 7, 11, 5, 9, 13]
    mi_runs = [4, 8, 12, 6, 10, 14]
    
    # 💡 실험할 피험자 수 설정 (ICA 연산 시간을 고려해 우선 3명으로 테스트, 이후 확장 가능)
    subjects_to_test = [1, 2, 3] 

    # AWS S3로 받은 경로가 있다면 지정해줍니다 (생략 시 기본 MNE 경로 사용)
    # mne.set_config('MNE_DATASETS_EEGBCI_PATH', './eeg_data/')

    print(f"====== EEGBCI 데이터 로드 및 전처리 시작 (피험자 수: {len(subjects_to_test)}명) ======")
    
    rm_simple_list, mi_simple_list = [], []
    rm_hard_list, mi_hard_list = [], []

    # --------------------------------------------------------------------------
    # 2. 데이터 순회 및 두 가지 전처리 동시 수행
    # --------------------------------------------------------------------------
    for sub in subjects_to_test:
        print(f" -> 피험자 S{sub:03d} 처리 중...")
        
        # Real Movement (RM) 처리
        for r in rm_runs:
            path = mne.datasets.eegbci.load_data(sub, r, verbose=False)[0]
            for hand in ['left', 'right']:
                # [간단 전처리]
                ep_base = preprocess_base(path, hand=hand)
                rm_simple_list.append(ep_base)
                # [하드 전처리 (ICA 포함)]
                rm_hard_list.append(apply_advanced_cleaning(ep_base))

        # Motor Imagery (MI) 처리
        for r in mi_runs:
            path = mne.datasets.eegbci.load_data(sub, r, verbose=False)[0]
            for hand in ['left', 'right']:
                # [간단 전처리]
                ep_base = preprocess_base(path, hand=hand)
                mi_simple_list.append(ep_base)
                # [하드 전처리 (ICA 포함)]
                mi_hard_list.append(apply_advanced_cleaning(ep_base))

    # 마스터 에포크 객체 생성
    print("\n -> 데이터를 통합하여 Master Epochs 생성 중...")
    epochs_simple = build_master_epochs(rm_simple_list, mi_simple_list)
    epochs_hard = build_master_epochs(rm_hard_list, mi_hard_list)

    print(f"완료! 총 에포크 개수: {len(epochs_simple)}개\n")

    # --------------------------------------------------------------------------
    # 3. [실험 1] 간단한 전처리 데이터 기반 모델 평가
    # --------------------------------------------------------------------------
    print("="*70)
    print("★★★ [실험 1] 간단한 전처리 (기본 필터링만 적용) 데이터 평가 ★★★")
    print("="*70)
    
    run_stage2_csp_svm(epochs_simple)
    run_stage3_eegnet(epochs_simple)

    # --------------------------------------------------------------------------
    # 4. [실험 2] 하드한 전처리 데이터 기반 모델 평가
    # --------------------------------------------------------------------------
    print("\n\n" + "="*70)
    print("★★★ [실험 2] 하드한 전처리 (ICA 불량 신호 제거 적용) 데이터 평가 ★★★")
    print("="*70)
    
    run_stage2_csp_svm(epochs_hard)
    run_stage3_eegnet(epochs_hard)

    # --------------------------------------------------------------------------
    # 5. [시각화] 노이즈가 제거된 하드 전처리 데이터로 TFR 시각화
    # --------------------------------------------------------------------------
    run_hypothesis_validation_tfr(epochs_hard)

if __name__ == "__main__":
    main()
