import numpy as np
import matplotlib.pyplot as plt
import mne

def run_hypothesis_validation_tfr(master_epochs_cleaned):
    print("\n" + "="*60)
    print("====== [가설 검증] 시간·강도·공간 3대 측면 통합 TFR 시각화 시작 ======")
    print("="*60)

    epochs_rm = master_epochs_cleaned['Real_Movement']
    epochs_mi = master_epochs_cleaned['Motor_Imagery']

    frequencies = np.arange(6, 36, 1)
    n_cycles = frequencies / 2.0

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True, sharey=True)
    states = [('Real Movement (RM)', epochs_rm, 0), ('Motor Imagery (MI)', epochs_mi, 1)]
    channels = [('Left Hemisphere (C3)', 'C3', 0), ('Right Hemisphere (C4)', 'C4', 1)]
    vmin, vmax = -3.0, 3.0

    for state_name, epochs_data, row in states:
        for ch_name, ch_code, col in channels:
            ax = axes[row, col]
            tfr = mne.time_frequency.tfr_morlet(
                epochs_data, freqs=frequencies, n_cycles=n_cycles,
                picks=[ch_code], return_itc=False, average=True, verbose=False
            )

            baseline_idx = np.where(tfr.times <= 0)[0]
            if len(baseline_idx) <= 1:
                baseline_idx = np.arange(len(tfr.times))

            raw_power = tfr.data[0]
            baseline_power = np.mean(raw_power[:, baseline_idx], axis=1, keepdims=True)
            data_matrix = 10 * np.log10((raw_power + 1e-10) / (baseline_power + 1e-10))
            data_matrix = np.nan_to_num(data_matrix, nan=0.0, posinf=vmax, neginf=vmin)

            im = ax.imshow(data_matrix, extent=[tfr.times[0], tfr.times[-1], frequencies[0], frequencies[-1]],
                           aspect='auto', origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
            ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.set_title(f"{state_name} - {ch_code} ({ch_name})", fontsize=12, fontweight='bold')
            if col == 0: ax.set_ylabel("Frequency (Hz)", fontsize=10)
            if row == 1: ax.set_xlabel("Time (Seconds)", fontsize=10)

    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Power Change (dB)', rotation=270, labelpad=15, fontweight='bold')
    plt.suptitle("BCI Hypothesis Validation: RM vs MI", fontsize=16, fontweight='bold', y=0.96)
    plt.subplots_adjust(right=0.90, hspace=0.2, wspace=0.15)
    plt.show()
