"""
scan_evs.py
Строит графики N/E/Z для каждого snapshot в snapshots/.
"""
import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_one(fname):
    d = np.load(fname, allow_pickle=False)
    n = d['n']; e = d['e']; z = d['z']; t = d['t']
    t0 = t[0]
    p_idx = int(d['p_idx'])
    s_idx = int(d['s_idx']) if 's_idx' in d.files else 0
    tr = (t - t0) * 400

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(tr, n, 'g')
    axes[0].set_ylabel('N')
    axes[0].axvline(p_idx, color='k', ls='--', label='p_idx')
    axes[0].axvline(s_idx, color='r', ls='--', label='s_idx')
    axes[0].legend()

    axes[1].plot(tr, e, 'orange')
    axes[1].set_ylabel('E')
    axes[1].axvline(p_idx, color='k', ls='--')
    axes[1].axvline(s_idx, color='r', ls='--')

    axes[2].plot(tr, z, 'b')
    axes[2].set_ylabel('Z')
    axes[2].axvline(p_idx, color='k', ls='--')
    axes[2].axvline(s_idx, color='r', ls='--')

    plt.tight_layout()
    out = fname.replace('.npz', '_nez.png')
    plt.savefig(out, dpi=100)
    plt.close()
    print(f'OK: {out}')


def main():
    files = sorted(glob.glob('snapshots/*.npz'))
    print(f'Найдено {len(files)} файлов')
    for f in files:
        try:
            plot_one(f)
        except Exception as ex:
            print(f'ERROR {f}: {ex}')


if __name__ == '__main__':
    main()