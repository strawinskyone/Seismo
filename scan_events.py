#!/usr/bin/env python3
"""
scan_events.py v2
Сводка по snapshots/.
Использование:
    python scan_events.py                    # сводка по всем
    python scan_events.py --list             # список файлов
    python scan_events.py --with-s           # только с S-волной
    python scan_events.py --type quake       # только quake
    python scan_events.py --min-ml 2.0       # только Ml >= 2.0
    python scan_events.py --file <path>      # детальный отчёт по одному
"""
import os
import glob
import argparse
import numpy as np


SNAPSHOT_DIR = "snapshots"


def _f(v):
    try:
        v = float(v)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None


def print_summary(files):
    if not files:
        print("Нет файлов в snapshots/")
        return
    total = len(files)
    with_s = 0
    types = {}
    mls = []
    strongest = None
    strongest_ml = -999
    print("=" * 78)
    print(f"СВОДКА ПО SNAPSHOTS/ ({total} файлов)")
    print("=" * 78)
    for fname in files:
        try:
            d = np.load(fname, allow_pickle=False)
        except Exception as e:
            print(f"  ! {os.path.basename(fname)}: {e}")
            continue
        is_s = bool(d['is_s']) if 'is_s' in d.files else False
        etype = str(d['event_type']) if 'event_type' in d.files else '?'
        ml = _f(d['ml_magnitude']) if 'ml_magnitude' in d.files else None
        if is_s:
            with_s += 1
        types[etype] = types.get(etype, 0) + 1
        if ml is not None:
            mls.append(ml)
            if ml > strongest_ml:
                strongest_ml = ml
                strongest = fname
    print(f"  С S-волной:        {with_s} / {total}")
    print(f"  Типы событий:      {types}")
    if mls:
        print(f"  Ml: min={min(mls):.2f} max={max(mls):.2f} "
              f"mean={sum(mls)/len(mls):.2f}")
    if strongest:
        print(f"  Сильнейшее:        {os.path.basename(strongest)} "
              f"(Ml={strongest_ml:.2f})")
    print("=" * 78)


def print_list(files):
    if not files:
        print("Нет файлов.")
        return
    print("=" * 78)
    print(f"СПИСОК ФАЙЛОВ ({len(files)})")
    print("=" * 78)
    for fname in files:
        try:
            d = np.load(fname, allow_pickle=False)
        except Exception:
            print(f"  ! {os.path.basename(fname)}")
            continue
        etype = str(d['event_type']) if 'event_type' in d.files else '?'
        is_s = bool(d['is_s']) if 'is_s' in d.files else False
        ml = _f(d['ml_magnitude']) if 'ml_magnitude' in d.files else None
        dist = _f(d['distance']) if 'distance' in d.files else None
        cft = _f(d['cft_max']) if 'cft_max' in d.files else None
        ml_str = f"Ml={ml:.2f}" if ml is not None else "Ml=N/A"
        dist_str = f"D={dist:.1f}km" if dist is not None else "D=N/A"
        cft_str = f"cft={cft:.2f}" if cft is not None else "cft=N/A"
        s_str = "S" if is_s else "-"
        print(f"  {os.path.basename(fname)}  [{s_str}] {etype:9s} "
              f"{ml_str:9s} {dist_str:9s} {cft_str}")
    print("=" * 78)


def print_detail(fname):
    if not os.path.exists(fname):
        print(f"Файл не найден: {fname}")
        return
    d = np.load(fname, allow_pickle=False)
    print("=" * 78)
    print(f"ФАЙЛ: {fname}")
    print("=" * 78)
    all_keys = ['event_type', 'confidence', 'is_s', 'p_s_delta',
                'cft_max', 'cft_sharpness', 's_amp_ratio', 'zone_pos',
                'azimuth', 'rectilinearity',
                'distance', 'depth', 'ml_magnitude',
                'peak_amplitude_mv', 'velocity_mm_s', 'displacement_um',
                'f_char', 'dom_freq',
                'scope_id', 'version', 'sample_rate',
                'p_idx', 'p_end_idx', 's_idx']
    for key in all_keys:
        if key in d.files:
            try:
                v = d[key]
                if v.ndim == 0:
                    v = v.item()
                print(f"  {key:22s} = {v}")
            except Exception as ex:
                print(f"  {key:22s} = <unreadable: {ex}>")
    z = d['z']; n = d['n']; e = d['e']
    sr = float(d['sample_rate'])
    print(f"  --- Сигналы ---")
    print(f"  samples={len(z)}, duration={len(z)/sr:.2f}s")
    for name, arr in [('Z', z), ('N', n), ('E', e)]:
        print(f"  {name}: min={arr.min():+.4f}V max={arr.max():+.4f}V "
              f"rms={np.sqrt(np.mean(arr**2)):.4f}V "
              f"p2p={arr.max()-arr.min():.4f}V")

    # Прореженный ряд Z, N, E
    step = max(1, len(z) // 120)
    print(f"  Z_decimated (every {step}-th):")
    print("    " + " ".join(f"{v:+.4f}" for v in z[::step]))
    print(f"  N_decimated (every {step}-th):")
    print("    " + " ".join(f"{v:+.4f}" for v in n[::step]))
    print(f"  E_decimated (every {step}-th):")
    print("    " + " ".join(f"{v:+.4f}" for v in e[::step]))
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', type=str, default=None,
                    help='Детальный отчёт по конкретному файлу')
    ap.add_argument('--list', action='store_true',
                    help='Список файлов с краткой сводкой')
    ap.add_argument('--with-s', action='store_true',
                    help='Только события с S-волной')
    ap.add_argument('--type', type=str, default=None,
                    help='Фильтр по типу (quake/explosion)')
    ap.add_argument('--min-ml', type=float, default=None,
                    help='Фильтр по Ml >= X')
    args = ap.parse_args()

    if args.file:
        print_detail(args.file)
        return

    files = sorted(glob.glob(os.path.join(SNAPSHOT_DIR, "*.npz")))
    if not files:
        print("Нет файлов в snapshots/")
        return

    filtered = []
    for fname in files:
        try:
            d = np.load(fname, allow_pickle=False)
        except Exception:
            continue
        if args.with_s:
            if not (bool(d['is_s']) if 'is_s' in d.files else False):
                continue
        if args.type:
            etype = str(d['event_type']) if 'event_type' in d.files else ''
            if etype != args.type:
                continue
        if args.min_ml is not None:
            ml = _f(d['ml_magnitude']) if 'ml_magnitude' in d.files else None
            if ml is None or ml < args.min_ml:
                continue
        filtered.append(fname)

    if args.list:
        print_list(filtered)
    else:
        print_summary(filtered)


if __name__ == "__main__":
    main()