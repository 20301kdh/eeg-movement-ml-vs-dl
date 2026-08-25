import mne
import numpy as np
from scipy.stats import kurtosis

def preprocess_base(file_path, hand='left'):
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    rename_dict = {ch: ch.strip(".").upper() for ch in raw.ch_names}
    raw.rename_channels(rename_dict)
    
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing="ignore")
    raw.filter(l_freq=1.0, h_freq=35, fir_design='firwin', verbose=False)
    
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    target_marker, target_value = ('T1', 2) if hand == 'left' else ('T2', 3)
    
    target_event_id = None
    for key, value in event_id.items():
        if key == target_marker or value == target_value:
            target_event_id = {key: value}
            break
    if target_event_id is None:
        target_event_id = {target_marker: target_value}

    return mne.Epochs(raw, events, event_id=target_event_id,
                      tmin=-1.0, tmax=3.0, baseline=(None, 0.0),
                      preload=True, verbose=False)

def apply_advanced_cleaning(epochs_base):
    epochs_cleaned = epochs_base.copy()
    data_matrix = epochs_cleaned.get_data(copy=True)
    n_epochs, n_channels, n_times = data_matrix.shape
    flattened_data = data_matrix.transpose(1, 0, 2).reshape(n_channels, -1)
    ch_names = epochs_cleaned.ch_names

    variances = np.var(flattened_data, axis=1)
    z_scores = (variances - np.mean(variances)) / np.std(variances)
    bad_var_names = [ch_names[i] for i in np.where(np.abs(z_scores) > 3)[0]]

    corr_matrix = np.corrcoef(flattened_data)
    np.fill_diagonal(corr_matrix, 0)
    bad_corr_names = [ch_names[i] for i in np.where(np.max(np.abs(corr_matrix), axis=1) < 0.4)[0]]
    epochs_cleaned.info['bads'] = list(set(bad_var_names + bad_corr_names))

    ica = mne.preprocessing.ICA(n_components=0.95, method='infomax',
                                fit_params=dict(extended=True), random_state=42)
    ica.fit(epochs_cleaned, verbose=False)

    exclude_idx = []
    sources = ica.get_sources(epochs_cleaned).get_data()
    sources_flat = sources.transpose(1, 0, 2).reshape(sources.shape[1], -1)
    sfreq = epochs_cleaned.info['sfreq']

    eog_ch = next((ch for ch in ch_names if 'FP1' in ch.upper()), None)
    if eog_ch:
        eog_indices, _ = ica.find_bads_eog(epochs_cleaned, ch_name=eog_ch, threshold=2.5, verbose=False)
        exclude_idx.extend(eog_indices)

    for idx, src_signal in enumerate(sources_flat):
        fft_vals = np.abs(np.fft.rfft(src_signal))
        fft_freqs = np.fft.rfftfreq(len(src_signal), d=1/sfreq)
        total_power = np.sum(fft_vals[(fft_freqs >= 1.0) & (fft_freqs <= 40.0)])
        high_power = np.sum(fft_vals[(fft_freqs >= 30.0) & (fft_freqs <= 40.0)])
        
        if high_power / (total_power + 1e-10) > 0.35 or kurtosis(src_signal) > 4.0:
            exclude_idx.append(idx)

    ica.exclude = list(set(exclude_idx))
    ica.apply(epochs_cleaned, verbose=False)

    if epochs_cleaned.info['bads']:
        epochs_cleaned = epochs_cleaned.interpolate_bads(reset_bads=True, verbose=False)
    return epochs_cleaned

def build_master_epochs(rm_list, mi_list):
    X_rm = mne.concatenate_epochs(rm_list).get_data(copy=True)
    X_mi = mne.concatenate_epochs(mi_list).get_data(copy=True)
    y_final = np.concatenate([np.zeros(X_rm.shape[0], dtype=int), np.ones(X_mi.shape[0], dtype=int)])
    X_final = np.concatenate([X_rm, X_mi], axis=0)

    new_events = np.zeros((len(y_final), 3), dtype=int)
    new_events[:, 0] = np.arange(len(y_final)) * 100
    new_events[:, -1] = y_final

    return mne.EpochsArray(X_final, rm_list[0].info, events=new_events,
                           event_id={'Real_Movement': 0, 'Motor_Imagery': 1}, verbose=False)
