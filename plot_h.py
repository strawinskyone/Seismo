#!/usr/bin/env python3
"""
plot_h.py — строит графики Z, N, E, H, cft для одного npz.
Использование:
    python plot_h.py snapshots/20260920_105039_217_quake.npz
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, find_peaks
from obspy.signal.trigger import recursive_sta_lta

import config


def bandpass(data, fs, fmin, fmax, order=4):
    if len(data) < 100:
        return data.copy()
    sos = butter(order, [fmin, fmax], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, data)


def detrend_linear(data):
    if len(data) < 10:
        return data.copy()
    x = np.arange(len(data), dtype=np.float64)
    coef = np.polyfit(x, data, 1)
    return data - (coef[0] * x + coef[1])


def main():
    if len(sys.argv) < 2:
        print("Использование: python plot_h.py <файл.npz>")
        sys.exit(1)

    fname = sys.argv[1]
    if not os.path.exists(fname):
        print(f"Файл не найден: {fname}")
        sys.exit(1)

    d = np.load(fname, allow_pickle=False)
    z = d['z'].astype(np.float64)
    n = d['n'].astype(np.float64)
    e = d['e'].astype(np.float64)
    sr = float(d['sample_rate'])
    p_idx = int(d['p_idx'])
    p_end_idx = int(d['p_end_idx']) if 'p_end_idx' in d.files else None
    s_idx = int(d['s_idx']) if 's_idx' in d.files and not np.isnan(d['s_idx']) else None

    # --- H ---
    n_s = detrend_linear(n.copy())
    e_s = detrend_linear(e.copy())
    n_s = bandpass(n_s, sr, config.S_FILTER_FREQMIN, config.S_FILTER_FREQMAX)
    e_s = bandpass(e_s, sr, config.S_FILTER_FREQMIN, config.S_FILTER_FREQMAX)
    h_data = np.sqrt(n_s**2 + e_s**2)

    # --- S-зона ---
    s_start = p_idx + config.S_SEARCH_START_AFTER_P_N
    s_end = min(len(h_data), p_idx + config.MAX_P_S_SEARCH_N)
    guard_skip = int(0.2 * sr)
    s_start = min(s_start + guard_skip, s_end - 1)

    # --- cft ---
    h_zone = h_data[s_start:s_end]
    cft_s = recursive_sta_lta(h_zone, config.S_PICKER_STA_N, config.S_PICKER_LTA_N)
    cft_full = np.full(len(h_data), np.nan)
    cft_full[s_start:s_start+len(cft_s)] = cft_s

    # --- время в секундах от начала ---
    t = (np.arange(len(z)) - p_idx) / sr

    # --- графики ---
    fig, axes = plt.subplots(5, 1, figsize=(16, 12), sharex=True)

    axes[0].plot(t, z, color='tab:blue', lw=0.7)
    axes[0].set_ylabel('Z, V')
    axes[0].axvline(0, color='k', ls='--', lw=0.5, label='p_idx')
    if p_end_idx is not None:
        axes[0].axvline((p_end_idx - p_idx)/sr, color='g', ls='--', lw=0.5, label='p_end')
    if s_idx is not None:
        axes[0].axvline((s_idx - p_idx)/sr, color='r', ls='--', lw=0.5, label='s_idx')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, n, color='tab:green', lw=0.7)
    axes[1].set_ylabel('N, V')
    axes[1].axvline(0, color='k', ls='--', lw=0.5)
    if s_idx is not None:
        axes[1].axvline((s_idx - p_idx)/sr, color='r', ls='--', lw=0.5)
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, e, color='tab:orange', lw=0.7)
    axes[2].set_ylabel('E, V')
    axes[2].axvline(0, color='k', ls='--', lw=0.5)
    if s_idx is not None:
        axes[2].axvline((s_idx - p_idx)/sr, color='r', ls='--', lw=0.5)
    axes[2].grid(alpha=0.3)

    axes[3].plot(t, h_data, color='tab:purple', lw=0.7)
    axes[3].set_ylabel('H = sqrt(N²+E²), V')
    axes[3].axvline(0, color='k', ls='--', lw=0.5, label='p_idx')
    axes[3].axvline((s_start - p_idx)/sr, color='b', ls='--', lw=0.5, label='s_start')
    axes[3].axvline((s_end - p_idx)/sr, color='b', ls='--', lw=0.5, label='s_end')
    if s_idx is not None:
        axes[3].axvline((s_idx - p_idx)/sr, color='r', ls='--', lw=0.5, label='s_idx')
    axes[3].legend(loc='upper right', fontsize=8)
    axes[3].grid(alpha=0.3)

    axes[4].plot(t, cft_full, color='tab:red', lw=0.7)
    axes[4].set_ylabel('cft (STA/LTA)')
    axes[4].axhline(config.S_PICKER_TRIGGER, color='k', ls='--', lw=0.5,
                    label=f'trig={config.S_PICKER_TRIGGER}')
    if s_idx is not None:
        axes[4].axvline((s_idx - p_idx)/sr, color='r', ls='--', lw=0.5, label='s_idx')
    axes[4].legend(loc='upper right', fontsize=8)
    axes[4].grid(alpha=0.3)
    axes[4].set_xlabel('Время от p_idx, с')

    plt.tight_layout()
    out = os.path.splitext(fname)[0] + '_plot.png'
    plt.savefig(out, dpi=110)
    print(f"Сохранено: {out}")

    # --- анализ: где реальные пики cft ---
    print("=" * 60)
    print(f"p_idx={p_idx}, p_end_idx={p_end_idx}, s_idx={s_idx}")
    print(f"s_start={s_start}, s_end={s_end}")
    print(f"cft_max={np.nanmax(cft_s):.2f}")
    cft_clean = cft_full[~np.isnan(cft_full)]
    peaks, props = find_peaks(cft_clean, height=config.S_PICKER_TRIGGER,
                              distance=int(0.5 * sr))
    print(f"Пиков cft > {config.S_PICKER_TRIGGER}: {len(peaks)}")
    for pk in peaks[:10]:
        idx_global = s_start + pk
        t_rel = (idx_global - p_idx) / sr
        zone_pos = pk / max(1, len(cft_s))
        print(f"  peak @ t={t_rel:+.3f}s, cft={cft_clean[pk]:.2f}, "
              f"zone_pos={zone_pos:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()