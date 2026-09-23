#!/usr/bin/env python3
"""
timer_20s.py
Бип-таймер на 20 с: бип в начале, отсчёт в консоли, 3 бипа в конце.

Использование:
  python3 timer_20s.py
"""
import sys
import time
import os
import struct

TIMER_SEC = 20
BEEP_START = True
BEEP_END = True


def play_beep(duration_sec=0.5, freq_hz=800):
    try:
        sr = 48000
        n = int(sr * duration_sec)
        amp = 32767
        data = bytearray()
        for i in range(n):
            s = int(amp * np.sin(2 * np.pi * freq_hz * i / sr))
            data.extend(struct.pack('<h', s))
        tmp = "/tmp/timer_beep.raw"
        with open(tmp, "wb") as f:
            f.write(data)
        os.system(f"aplay -t raw -f S16_LE -r {sr} -c 1 {tmp} > /dev/null 2>&1")
    except Exception:
        sys.stdout.write('\a')
        sys.stdout.flush()


def main():
    print("=" * 40)
    print(f"  ТАЙМЕР {TIMER_SEC} с")
    print("=" * 40)

    if BEEP_START:
        print("  🔔 СТАРТ")
        play_beep(duration_sec=0.5, freq_hz=800)

    t_start = time.perf_counter()
    t_end = t_start + TIMER_SEC

    while True:
        elapsed = time.perf_counter() - t_start
        remaining = TIMER_SEC - elapsed
        if remaining <= 0:
            break
        sys.stdout.write(f"\r  Осталось: {remaining:5.1f} с  ")
        sys.stdout.flush()
        time.sleep(0.1)

    print(f"\r  Осталось:   0.0 с  ")
    print()

    if BEEP_END:
        print("  🔔 СТОП")
        play_beep(duration_sec=0.15, freq_hz=1200)
        time.sleep(0.2)
        play_beep(duration_sec=0.15, freq_hz=1200)
        time.sleep(0.2)
        play_beep(duration_sec=0.5, freq_hz=1200)

    actual = time.perf_counter() - t_start
    print(f"  Фактически прошло: {actual:.2f} с")


if __name__ == "__main__":
    import numpy as np
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Прервано.")
        sys.exit(0)