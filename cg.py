#!/usr/bin/env python3
"""
calibrate_geophone.py v7.3.7
Калибратор геофона для AD7606B.
Канал задаётся константой CAL_CHANNEL (0=N, 1=E, 2=Z).
"""
import numpy as np
import time
import sys
import os
import pigpio

import config
from AD7606B import AD7606B

# ============================================================================
# ВЫБЕРИТЕ КАНАЛ ДЛЯ КАЛИБРОВКИ (перезапускайте скрипт для каждого геофона)
# ============================================================================
CAL_CHANNEL = 1          # 0 = N (А0 / CH1), 1 = E (А1 / CH2), 2 = Z (А2 / CH3)
CHANNEL_NAMES = {0: "N (А0 / CH1)", 1: "E (А1 / CH2)", 2: "Z (А2 / CH3)"}

# Проверка
if CAL_CHANNEL not in (0, 1, 2):
    print("❌ CAL_CHANNEL должен быть 0, 1 или 2")
    sys.exit(1)

def play_beep(duration_sec=0.5, freq_hz=800):
    """Генерирует звуковой тон через колонки RPi"""
    try:
        # Генерируем raw sine wave и воспроизводим через aplay
        # Это надёжнее speaker-test с фоновым kill
        import struct
        sample_rate = 48000
        n_samples = int(sample_rate * duration_sec)
        amplitude = 32767
        data = bytearray()
        for i in range(n_samples):
            sample = int(amplitude * np.sin(2 * np.pi * freq_hz * i / sample_rate))
            data.extend(struct.pack('<h', sample))
        
        # Записываем временный raw-файл и проигрываем
        tmp_path = "/tmp/cal_beep.raw"
        with open(tmp_path, "wb") as f:
            f.write(data)
        os.system(f"aplay -t raw -f S16_LE -r {sample_rate} -c 1 {tmp_path} > /dev/null 2>&1")
    except Exception:
        sys.stdout.write('\a')
        sys.stdout.flush()

def run_hardware_calibration():
    ch_name = CHANNEL_NAMES[CAL_CHANNEL]
    print("=" * 65)
    print(f" 🛠️  КАЛИБРОВКА ГЕОФОНА: КАНАЛ {ch_name}")
    print("=" * 65)
    
    # 1. Таймер ожидания
    START_DELAY = 5
    for i in range(START_DELAY, 0, -1):
        print(f"[ТАЙМЕР] До старта осталось: {i} сек... Спокойно идите к приямку!")
        if i <= 3:
            play_beep(duration_sec=0.1, freq_hz=600)
            time.sleep(0.9)
        else:
            time.sleep(1.0)
        
    # Сигнал старта
    print(f"\n[🔔 СТАРТ!] ОТПУСКАЙТЕ МАЯТНИК (Отклонение строго 0.5 см)! Канал: {ch_name}")
    play_beep(duration_sec=1.5, freq_hz=1000)

    # 2. Инициализация
    pi_hardware = pigpio.pi()
    if not pi_hardware.connected:
        print("❌ ОШИБКА: Не удалось подключиться к pigpiod! Запустите: sudo pigpiod")
        sys.exit(1)

    adc = AD7606B(pi=pi_hardware) 
    
    duration_sec = 10.0
    total_samples = int(duration_sec * config.SAMPLE_RATE)
    raw_samples = []
    sample_interval = 1.0 / config.SAMPLE_RATE
    
    print(f"Идет непрерывная скоростная запись канала {ch_name}...")
    t_start = time.perf_counter()          # ← perf_counter вместо time.time()
    for _ in range(total_samples):
        _, raw_codes = adc.read_4ch_fixed_delay()
        raw_samples.append(raw_codes[CAL_CHANNEL])
        time.sleep(sample_interval)
    t_end = time.perf_counter()
    
    real_fs = total_samples / (t_end - t_start)
    print(f"[OK] Запись завершена за {t_end - t_start:.2f} сек. Реальная Fs: {real_fs:.1f} Гц")
    
    adc.close()
    pi_hardware.stop()

    # Сигнал окончания
    print("[INFO] Сбор данных завершен. Можно подходить к консоли.")
    for _ in range(3):
        play_beep(duration_sec=0.2, freq_hz=800)
        time.sleep(0.15)

    # 3. Математический анализ
    n = len(raw_samples)
    sig_v = np.array(raw_samples, dtype=np.float64) * config.ADC_SCALE_V
    sig_v -= np.mean(sig_v)

    E_max = np.max(np.abs(sig_v))
    
    window = np.hanning(n)
    fft_data = np.abs(np.fft.rfft(sig_v * window))
    freqs = np.fft.rfftfreq(n, d=1.0/config.SAMPLE_RATE)
    
    idx_peak = np.argmax(fft_data[1:]) + 1
    f_0_exact = freqs[idx_peak]
    omega_0_exact = 2 * np.pi * f_0_exact

    X_0 = 0.005  # 0.5 см в метрах
    
    V_max_theoretical = X_0 * omega_0_exact
    G_exact = E_max / (V_max_theoretical + 1e-10)

    print("\n" + "=" * 65)
    print(f" 📊 ИТОГОВЫЙ ОТЧЕТ КАЛИБРОВКИ (КАНАЛ {ch_name}):")
    print("=" * 65)
    print(f"➡️ Замерено пиковое напряжение (E_max): {E_max:.4f} В")
    print(f"➡️ Точная частота маятника (f0):     {f_0_exact:.3f} Гц")
    print(f"➡️ Скорость катушки маятника:        {V_max_theoretical:.4f} м/с")
    print("-" * 65)
    print(f"🔥 РАССЧИТАННЫЙ КОЭФФИЦИЕНТ ГЕОФОНА:")
    print(f"👉 G_{['N','E','Z'][CAL_CHANNEL]} = {G_exact:.2f} В/(м/с) 👈")
    print("-" * 65)
    print("Внесите это число в config.py → GEOPHONE_SENSITIVITY")
    print("(если калибруете один общий коэффициент, усредните по трём каналам)")
    print("=" * 65)

if __name__ == "__main__":
    try:
        run_hardware_calibration()
    except KeyboardInterrupt:
        print("\nКалибровка прервана.")
        sys.exit(0)