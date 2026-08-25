# 분리된 모듈에서 필요한 함수들 불러오기
from preprocess import preprocess_base, apply_advanced_cleaning, build_master_epochs
from visualize import run_hypothesis_validation_tfr
from models import run_stage2_csp_svm, run_stage3_eegnet

def main():
    # 1. 파일 경로 설정 (본인의 환경에 맞게 실제 파일 경로로 수정하세요)
    rm_files = ['data/subject1_RM.edf']
    mi_files = ['data/subject1_MI.edf']

    print("뇌파 데이터 파일 로딩 준비 완료...")

    # ==============================================================================
    # [실험 1] 간단한 전처리 (Base Preprocessing Only)
    # 목표: 1~35Hz 대역통과 필터링만 거친 원시적인 데이터로 ML과 DL의 성능 비교
    # ==============================================================================
    print("\n\n" + "#"*70)
    print("### [실험 1] 간단한 전처리 (기본 필터링만 적용) 데이터 세트 ###")
    print("#"*70)
    
    rm_simple_list = [preprocess_base(f, hand='left') for f in rm_files]
    mi_simple_list = [preprocess_base(f, hand='left') for f in mi_files]
    
    epochs_simple = build_master_epochs(rm_simple_list, mi_simple_list)

    print("\n[실험 1 모델 평가]")
    run_stage2_csp_svm(epochs_simple)
    run_stage3_eegnet(epochs_simple)


    # ==============================================================================
    # [실험 2] 하드한 전처리 (Advanced Cleaning)
    # 목표: ICA, EOG, 근육 노이즈(EMG) 등 불량 신호를 철저히 제거한 후 성능 비교
    # ==============================================================================
    print("\n\n" + "#"*70)
    print("### [실험 2] 하드한 전처리 (기본 필터링 + ICA 정밀 정화) 데이터 세트 ###")
    print("#"*70)

    rm_hard_list = []
    for f in rm_files:
        base = preprocess_base(f, hand='left')
        hard = apply_advanced_cleaning(base)
        rm_hard_list.append(hard)

    mi_hard_list = []
    for f in mi_files:
        base = preprocess_base(f, hand='left')
        hard = apply_advanced_cleaning(base)
        mi_hard_list.append(hard)

    epochs_hard = build_master_epochs(rm_hard_list, mi_hard_list)

    # 시각화(TFR)는 노이즈가 제거되어 가장 깔끔한 '하드 전처리' 데이터로 진행
    print("\n[하드 전처리 데이터 기반 TFR 시각화]")
    run_hypothesis_validation_tfr(epochs_hard)

    print("\n[실험 2 모델 평가]")
    run_stage2_csp_svm(epochs_hard)
    run_stage3_eegnet(epochs_hard)


if __name__ == "__main__":
    main()
