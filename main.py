# main.py
import warnings
import torch
from data_splitter import build_isolated_datasets
from models import run_stage2_csp_svm, run_stage3_eegnet
from visualize import run_hypothesis_validation_tfr

warnings.filterwarnings("ignore")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"====== [본 실험 개시] 연산 디바이스: {device} ======")
    print("====== 데이터 오염 방지: 신체 부위별 4분할 격리 실험 ======\n")

    # 실험용 세팅 (빠른 테스트를 위해 3명으로 제한, 본 실험 시 range(1, 51)로 변경)
    subjects_to_test = [1, 2, 3] 

    print(f" -> 총 {len(subjects_to_test)}명 피험자 데이터를 4개 부위로 격리 병합 중...")
    
    # 1. 신체 부위별로 4등분 된 데이터셋 생성
    isolated_master_datasets = build_isolated_datasets(subjects_to_test)
    
    print("\n[데이터 구축 완료] 본격적인 부위별 모델 평가를 시작합니다.")

    # 2. 부위별(왼손, 오른손, 양손, 양발) 4회 반복 순회 실험
    for body_part, datasets in isolated_master_datasets.items():
        epochs_simple = datasets['simple']
        epochs_hard = datasets['hard']
        
        print("\n" + "="*80)
        print(f"[ 타겟 신체 부위: {body_part} ] 실험 세션 시작")
        print("="*80)
        
        # [조건 A] 간단 전처리 성능 평가
        print(f"\n▶ [조건 A] {body_part} - 간단 전처리 데이터 평가 (Epochs: {len(epochs_simple)})")
        run_stage2_csp_svm(epochs_simple)
        run_stage3_eegnet(epochs_simple)
        
        # [조건 B] 하드 전처리(ICA) 성능 평가
        print(f"\n▶ [조건 B] {body_part} - 하드 전처리 데이터 평가 (Epochs: {len(epochs_hard)})")
        run_stage2_csp_svm(epochs_hard)
        run_stage3_eegnet(epochs_hard)
        
        # [시각화] 하드 전처리 데이터 기준 TFR 출력
        print(f"\n -> {body_part} TFR(시간-주파수 반응) 시각화 생성 중...")
        run_hypothesis_validation_tfr(epochs_hard)

    print("\n====== 모든 신체 부위별 교차 실험 및 시각화 완료 ======")

if __name__ == "__main__":
    main()
