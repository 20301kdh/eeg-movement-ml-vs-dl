import mne
from preprocess import preprocess_base, apply_advanced_cleaning, build_master_epochs

# ---------------------------------------------------------
# [Run 번호 정의] 문서 기준 엄격한 분리
# ---------------------------------------------------------
# 편측(Unilateral) 런: T1 = 왼손(Left Fist), T2 = 오른손(Right Fist)
RM_UNILATERAL = [3, 7, 11]
MI_UNILATERAL = [4, 8, 12]

# 양측(Bilateral) 런: T1 = 양손(Both Fists), T2 = 양발(Both Feet)
RM_BILATERAL = [5, 9, 13]
MI_BILATERAL = [6, 10, 14]

def build_isolated_datasets(subjects_to_test):
    """
    피험자 리스트를 받아 4개의 신체 부위별로 완전히 격리된 
    RM vs MI 마스터 에포크 딕셔너리를 반환합니다.
    """
    # 4개의 독립된 카테고리 데이터 컨테이너
    isolated_data = {
        'Left_Fist':  {'RM_simple': [], 'MI_simple': [], 'RM_hard': [], 'MI_hard': []},
        'Right_Fist': {'RM_simple': [], 'MI_simple': [], 'RM_hard': [], 'MI_hard': []},
        'Both_Fists': {'RM_simple': [], 'MI_simple': [], 'RM_hard': [], 'MI_hard': []},
        'Both_Feet':  {'RM_simple': [], 'MI_simple': [], 'RM_hard': [], 'MI_hard': []}
    }
    
    for sub in subjects_to_test:
        # 1. 왼손 (Left Fist) 추출 -> Unilateral Run + T1(hand='left')
        _extract_and_process(sub, RM_UNILATERAL, 'left', isolated_data['Left_Fist'], 'RM')
        _extract_and_process(sub, MI_UNILATERAL, 'left', isolated_data['Left_Fist'], 'MI')
        
        # 2. 오른손 (Right Fist) 추출 -> Unilateral Run + T2(hand='right')
        _extract_and_process(sub, RM_UNILATERAL, 'right', isolated_data['Right_Fist'], 'RM')
        _extract_and_process(sub, MI_UNILATERAL, 'right', isolated_data['Right_Fist'], 'MI')
        
        # 3. 양손 (Both Fists) 추출 -> Bilateral Run + T1(hand='left')
        _extract_and_process(sub, RM_BILATERAL, 'left', isolated_data['Both_Fists'], 'RM')
        _extract_and_process(sub, MI_BILATERAL, 'left', isolated_data['Both_Fists'], 'MI')
        
        # 4. 양발 (Both Feet) 추출 -> Bilateral Run + T2(hand='right')
        _extract_and_process(sub, RM_BILATERAL, 'right', isolated_data['Both_Feet'], 'RM')
        _extract_and_process(sub, MI_BILATERAL, 'right', isolated_data['Both_Feet'], 'MI')

    # 각 부위별로 Master Epochs 병합
    master_datasets = {}
    for part, data in isolated_data.items():
        if len(data['RM_simple']) > 0 and len(data['MI_simple']) > 0:
            master_datasets[part] = {
                'simple': build_master_epochs(data['RM_simple'], data['MI_simple']),
                'hard': build_master_epochs(data['RM_hard'], data['MI_hard'])
            }
            
    return master_datasets

def _extract_and_process(sub, runs, hand_param, target_dict, task_type):
    """
    내부 유틸리티 함수: 특정 조건의 데이터를 다운로드하고 전처리하여 딕셔너리에 추가합니다.
    """
    for r in runs:
        try:
            path = mne.datasets.eegbci.load_data(sub, r, verbose=False)[0]
            # preprocess.py의 기본 기능을 활용 (hand='left'면 T1, 'right'면 T2 추출)
            ep_base = preprocess_base(path, hand=hand_param)
            
            # Simple 및 Hard 전처리 병렬 적용
            target_dict[f'{task_type}_simple'].append(ep_base)
            target_dict[f'{task_type}_hard'].append(apply_advanced_cleaning(ep_base))
        except Exception:
            continue
