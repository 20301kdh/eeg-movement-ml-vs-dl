# data_splitter.py
import mne
from preprocess import preprocess_base, apply_advanced_cleaning, build_master_epochs

RM_UNILATERAL = [3, 7, 11]
MI_UNILATERAL = [4, 8, 12]
RM_BILATERAL = [5, 9, 13]
MI_BILATERAL = [6, 10, 14]

def build_isolated_datasets(subjects_to_test):
    isolated_data = {
        'Left_Fist':  {'RM_simple': [], 'MI_simple': [], 'RM_hard': [], 'MI_hard': []},
        'Right_Fist': {'RM_simple': [], 'MI_simple': [], 'RM_hard': [], 'MI_hard': []},
        'Both_Fists': {'RM_simple': [], 'MI_simple': [], 'RM_hard': [], 'MI_hard': []},
        'Both_Feet':  {'RM_simple': [], 'MI_simple': [], 'RM_hard': [], 'MI_hard': []}
    }
    
    for sub in subjects_to_test:
        if sub % 5 == 0 or sub == 1:
            print(f"   [진행 상황] 피험자 S{sub:03d} 데이터를 부위별로 격리 및 전처리 중...")
            
        _extract_and_process(sub, RM_UNILATERAL, 'left', isolated_data['Left_Fist'], 'RM')
        _extract_and_process(sub, MI_UNILATERAL, 'left', isolated_data['Left_Fist'], 'MI')
        
        _extract_and_process(sub, RM_UNILATERAL, 'right', isolated_data['Right_Fist'], 'RM')
        _extract_and_process(sub, MI_UNILATERAL, 'right', isolated_data['Right_Fist'], 'MI')
        
        _extract_and_process(sub, RM_BILATERAL, 'left', isolated_data['Both_Fists'], 'RM')
        _extract_and_process(sub, MI_BILATERAL, 'left', isolated_data['Both_Fists'], 'MI')
        
        _extract_and_process(sub, RM_BILATERAL, 'right', isolated_data['Both_Feet'], 'RM')
        _extract_and_process(sub, MI_BILATERAL, 'right', isolated_data['Both_Feet'], 'MI')

    master_datasets = {}
    for part, data in isolated_data.items():
        if len(data['RM_simple']) > 0 and len(data['MI_simple']) > 0:
            master_datasets[part] = {
                'simple': build_master_epochs(data['RM_simple'], data['MI_simple']),
                'hard': build_master_epochs(data['RM_hard'], data['MI_hard'])
            }
    return master_datasets

def _extract_and_process(sub, runs, hand_param, target_dict, task_type):
    for r in runs:
        try:
            path = mne.datasets.eegbci.load_data(sub, r, verbose=False)[0]
            ep_base = preprocess_base(path, hand=hand_param)
            target_dict[f'{task_type}_simple'].append(ep_base)
            target_dict[f'{task_type}_hard'].append(apply_advanced_cleaning(ep_base))
        except Exception:
            continue
