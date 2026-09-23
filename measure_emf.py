#!/usr/bin/env python3
"""
measure_emf.py v1.1.0
Измерение ЭДС катушки геофона через ADC (без преампа).
Автотриггер по амплитуде — не нужно угадывать момент отпускания.
"""
import sys
import time
import os
import struct
import logging
import collections
import numpy as np
import pigpio

import config
from AD7606B import AD7606B

logger = logging.getLogger('seismic')

# ============================================================================
# ПАРАМЕТРЫ
# ============================================================================
CAL_CHANNEL = 3                     # 1=N (A1), 2=E (A2), 3=Z (A3)
CHANNEL_NAMES = {1: "A1 (CH2)", 2: "A2 (CH3)", 3: "A3 (CH4)",
                 5: "A5 (CH6)", 6: "A6 (CH7)", 7: "A7 (CH8)"}
CHANNEL_DIRECTIONS = {1: "СЕВЕР", 2: "ВОСТОК", 3: "ВВЕРХ",
                      5: "-", 6: "-", 7: "-"}

X_0_M = 0.005                       # амплитуда отклонения, м (5 мм)
GAIN_PREAMP = 1090.0                # R_os / R_in

# v1.1.0: запись по триггеру от прихода волны.
PRETRIGGER_SEC = 1.0                # сколько писать ДО триггера
POSTTRIGGER_SEC = 5.0               # сколько писать ПОСЛЕ
MAX_WAIT_SEC = 30.0                 # максимум ждать триггер
TRIGGER_ABS_MV = 5.0                # порог триггера, мВ (≈ 65 LSB)
BEEP_BEFORE_SEC = 3.0               # бип за N секунд до «слушаем»

# Проверка
if CAL_CHANNEL not in (1, 2, 3, 5, 6, 7):
    print("❌ CAL_CHANNEL должен быть 1, 2, 3, 5, 6 или 7")
    sys.exit(1)


def play_beep(duration_sec=0.5, freq_hz=800):
    """Короткий звуковой сигнал через aplay."""
    try:
        sr = 48000
        n = int(sr * duration_sec)
        amp = 32767
        data = bytearray()
        for i in range(n):
            s = int(amp * np.sin(2 * np.pi * freq_hz * i / sr))
            data.extend(struct.pack('<h', s))
        tmp = "/tmp/measure_emf_beep.raw"
        with open(tmp, "wb") as f:
            f.write(data)
        os.system(f"aplay -t raw -f S16_LE -r {sr} -c 1 {tmp} > /dev/null 2>&1")
    except Exception:
        sys.stdout.write('\a')
        sys.stdout.flush()


def analyze(samples_v, real_fs=None):
    """Возвращает E_max, E_rms, f_peak, V_max, G, знак 1-го импульса."""
    if real_fs is None or real_fs <= 0:
        real_fs = config.SAMPLE_RATE
    n = len(samples_v)
    sig = samples_v - np.mean(samples_v)

    E_max = float(np.max(np.abs(sig)))
    E_rms = float(np.std(sig))

    # --- Знак первого импульса ---
    threshold = 0.3 * E_max if E_max > 0 else 1e-9
    first_sign = 0
    first_idx = -1
    for i in range(len(sig)):
        if abs(sig[i]) > threshold:
            first_sign = 1 if sig[i] > 0 else -1
            first_idx = i
            break

    # --- f_peak: zero-crossing в активной части ---
    # v1.1.5: f_peak через FFT в окне 2–4 Гц. Zero-crossing на затухающем
    # сигнале даёт большой разброс.
    window = np.hanning(n)
    spec = np.abs(np.fft.rfft(sig * window))
    freqs = np.fft.rfftfreq(n, d=1.0 / real_fs)
    mask = (freqs > 1.8) & (freqs < 2.4)
    if np.any(mask):
        idx_local = int(np.argmax(spec[mask]))
        f_peak = float(freqs[mask][idx_local])
    else:
        idx_peak = int(np.argmax(spec[1:]) + 1)
        f_peak = float(freqs[idx_peak])
    logger.info(f"[FFT-DBG] f_peak={f_peak:.3f} (окно 2–4 Гц)")

    # --- Коэффициент G ---
    omega_0 = 2.0 * np.pi * f_peak
    V_max_theoretical = X_0_M * omega_0
    E_at_preamp_output = E_max * GAIN_PREAMP
    G = E_at_preamp_output / (V_max_theoretical + 1e-12)

    return {
        'E_max': E_max,
        'E_rms': E_rms,
        'f_peak': f_peak,
        'V_max_theoretical': V_max_theoretical,
        'E_at_preamp_output': E_at_preamp_output,
        'G': G,
        'n': n,
        'first_sign': first_sign,
        'first_idx': first_idx,
    }


def run_measurement():
    ch_name = CHANNEL_NAMES.get(CAL_CHANNEL, f"A{CAL_CHANNEL}")
    ch_dir = CHANNEL_DIRECTIONS.get(CAL_CHANNEL, "-")
    _ch_letter = {1: 'N', 2: 'E', 3: 'Z'}.get(CAL_CHANNEL,
                                               f'A{CAL_CHANNEL}')

    print("=" * 68)
    print(f" 📡  ИЗМЕРЕНИЕ ЭДС КАТУШКИ: КАНАЛ {ch_name}")
    print("=" * 68)
    print(f"  Отклонение:    {X_0_M*1000:.1f} мм в направлении «{ch_dir}»")
    print(f"  GAIN преампа:  {GAIN_PREAMP:.0f}")
    print(f"  Pre-trigger:   {PRETRIGGER_SEC:.1f} с")
    print(f"  Post-trigger:  {POSTTRIGGER_SEC:.1f} с")
    print(f"  Порог триггера: {TRIGGER_ABS_MV:.1f} мВ")
    print(f"  Макс. ожидание: {MAX_WAIT_SEC:.0f} с")
    print(f"  1 LSB =        {config.ADC_SCALE_V*1e6:.2f} мкВ")
    print(f"  Ожидание SEED: при отклонении в «{ch_dir}» — первый импульс −")
    print()

    # --- Инициализация ADC ---
    pi = pigpio.pi()
    if not pi.connected:
        print("  ❌ ОШИБКА: pigpiod не запущен. Выполните: sudo pigpiod")
        return None
    try:
        adc = AD7606B(pi=pi)
    except Exception as e:
        print(f"  ❌ ОШИБКА инициализации ADC: {e}")
        pi.stop()
        return None

    # --- Бип за N секунд до «слушаю» ---
    print(f"\n  [🔔 ПОДГОТОВКА] Через {BEEP_BEFORE_SEC:.0f} с начну ждать импульс.")
    print(f"  Отклоните катушку на {X_0_M*1000:.0f} мм "
          f"в направлении «{ch_dir}» и отпустите.")
    play_beep(duration_sec=0.5, freq_hz=800)
    time.sleep(BEEP_BEFORE_SEC)

    # --- Запись с триггером ---
    pre_len = int(PRETRIGGER_SEC * config.SAMPLE_RATE)
    post_len = int(POSTTRIGGER_SEC * config.SAMPLE_RATE)
    ring = collections.deque(maxlen=pre_len)
    samples = []
    t_start = time.perf_counter()
    trigger_found = False
    t_end_wait = t_start + MAX_WAIT_SEC
    threshold_v = TRIGGER_ABS_MV * 1e-3

    print(f"  [👂 СЛУШАЮ] Порог {TRIGGER_ABS_MV:.1f} мВ...")
    play_beep(duration_sec=1.0, freq_hz=1200)

    try:
        while True:
            _, raws = adc.read_4ch_fixed_delay()
            val = raws[CAL_CHANNEL]
            val_v = val * config.ADC_SCALE_V

            if not trigger_found:
                ring.append(val)
                if abs(val_v) > threshold_v:
                    trigger_found = True
                    samples.extend(ring)
                    print(f"  [🔔 ТРИГГЕР] |V|={abs(val_v)*1000:.2f} мВ, "
                          f"pre={len(ring)-1}")
                elif time.perf_counter() > t_end_wait:
                    print(f"  ⚠️  ТАЙМАУТ: триггер не сработал за "
                          f"{MAX_WAIT_SEC:.0f} с.")
                    break
            else:
                samples.append(val)
                if len(samples) >= pre_len + post_len:
                    break
    except Exception as e:
        print(f"  ❌ ОШИБКА записи: {e}")
    finally:
        try:
            adc.close()
        except Exception:
            pass
        pi.stop()

    t_end = time.perf_counter()
    total = len(samples)
    if total < 100:
        print(f"  ❌ Слишком мало отсчётов: {total}. Пропуск.")
        return None

    real_fs = total / (t_end - t_start)
    print(f"  [OK] Записано {total} отсчётов за {t_end - t_start:.2f} с "
          f"(Fs = {real_fs:.1f} Гц)")

    # --- Для G-анализа: только post-trigger часть ---
    samples_for_g = samples
    print(f"  [INFO] Для G-анализа: {len(samples_for_g)} отсчётов (pre+post)")

    # --- Анализ ---
    samples_v = np.array(samples_for_g, dtype=np.float64) * config.ADC_SCALE_V
    if len(samples_v) < 100:
        print("  ❌ Слишком мало отсчётов для анализа.")
        return None

    res = analyze(samples_v, real_fs=real_fs)

    # --- Проверки ---
    adc_max_v = 2.5
    clip = res['E_max'] > adc_max_v * 0.95
    weak = res['E_max'] < 5.0 * config.ADC_SCALE_V

    # --- Отчёт ---
    print()
    print("=" * 68)
    print(f" 📊  РЕЗУЛЬТАТ ИЗМЕРЕНИЯ (КАНАЛ {ch_name})")
    print("=" * 68)
    print(f"  Отсчётов:                 {res['n']}")
    print(f"  Fs реальная:              {real_fs:.2f} Гц")
    print()
    print(f"  E_max (катушка):          {res['E_max']*1000:.4f} мВ "
          f"({res['E_max']:.6f} В)")
    print(f"  E_rms (катушка):          {res['E_rms']*1000:.4f} мВ")
    print(f"  LSB_max:                  {res['E_max']/config.ADC_SCALE_V:.1f}")
    print(f"  LSB_rms:                  {res['E_rms']/config.ADC_SCALE_V:.1f}")
    print()
    print(f"  f_peak (маятник):         {res['f_peak']:.3f} Гц")
    print(f"  V_max теоретическая:      {res['V_max_theoretical']:.4f} м/с")
    print(f"  E на выходе преампа:      {res['E_at_preamp_output']*1000:.2f} мВ")
    print()

    sign_str = ('+' if res['first_sign'] > 0
                else ('-' if res['first_sign'] < 0 else '?'))
    print(f"  Знак 1-го импульса:       {sign_str} (idx={res['first_idx']})")
    print(f"  Ожидание (SEED):          - (минус) при отклонении "
          f"в «{ch_dir}»")
    if res['first_sign'] > 0:
        print(f"  ℹ️  Первый импульс +. Для инвертирующего преампа — норма.")
    elif res['first_sign'] < 0:
        print(f"  ✅ Первый импульс −. Прямое включение.")
    else:
        print(f"  ⚠️  Не удалось определить знак.")

    print()
    print(f"  🔥 КОЭФФИЦИЕНТ G:")
    print(f"     GEOPHONE_SENSITIVITY_{_ch_letter} = {res['G']:.2f} В/(м/с)")
    print()

    if clip:
        print(f"  ⚠️  КЛИППИНГ: E_max={res['E_max']:.4f} В ≥ "
              f"{adc_max_v*0.95:.3f} В. Уменьшите отклонение!")
    if weak:
        print(f"  ⚠️  СЛАБЫЙ СИГНАЛ: E_max={res['E_max']*1000:.3f} мВ "
              f"≈ шум. Увеличьте отклонение или используйте преамп.")

    print("=" * 68)
    print(f"  Внесите в config.py:")
    print(f"    GEOPHONE_SENSITIVITY_{_ch_letter} = {res['G']:.2f}")
    print("=" * 68)

    return res


def main():
    config.setup_logging()   # v1.1.1: логи в ddd.log
    try:
        res = run_measurement()
        if res is None:
            sys.exit(1)
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n  Прервано пользователем.")
        sys.exit(0)
    except Exception as e:
        print(f"\n  ❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()