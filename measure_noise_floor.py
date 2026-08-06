#!/usr/bin/env python3
"""
measure_noise_floor.py v8.1
Измерение уровня шума сейсмостанции в режиме полного покоя.
Точный тайминг: 1/SAMPLE_RATE между выборками.
ВСЕ сообщения → logger 'seismic' (ddd.log / error.log). Терминал чист.
"""
import sys
import time
import numpy as np
from scipy import signal
import pigpio
import logging

import config
from AD7606B import AD7606B

logger = logging.getLogger('seismic')

SAMPLE_RATE = 400
RECORD_SECONDS = 20.0
NUM_SAMPLES = int(SAMPLE_RATE * RECORD_SECONDS)
ADC_SCALE_V = 2.5 / 32768.0
NPERSEG = 1024
NOVERLAP = 512
WINDOW = 'hann'


def main():
    logger.info("=== Измерение уровня шума сейсмостанции ===")
    logger.info(f"Сбор данных в режиме полного покоя, {RECORD_SECONDS:.1f} секунд, {NUM_SAMPLES} сэмплов")

    pi = None
    adc = None
    try:
        pi = pigpio.pi()
        if not pi.connected:
            raise RuntimeError("Не удалось подключиться к pigpio daemon")

        # SPI 500 кГц для снижения наводок
        adc = AD7606B(pi=pi, sample_rate=SAMPLE_RATE, spi_speed_hz=1000000)

        raw_buffer = np.empty((NUM_SAMPLES, 3), dtype=np.int64)
        volt_buffer = np.empty((NUM_SAMPLES, 3), dtype=np.float64)

        interval = 1.0 / SAMPLE_RATE
        next_frame = time.perf_counter()
        start_time = time.perf_counter()

        for idx in range(NUM_SAMPLES):
            # Точный тайминг — как в acquisition.py
            current_time = time.perf_counter()
            if current_time < next_frame:
                remaining = next_frame - current_time
                if remaining > 0.0015:
                    time.sleep(remaining - 0.001)
                while time.perf_counter() < next_frame:
                    pass
            next_frame += interval

            volts, raw_codes = adc.read_4ch_fixed_delay()
            raw_buffer[idx, 0] = int(raw_codes[2])  # Z
            raw_buffer[idx, 1] = int(raw_codes[0])  # N
            raw_buffer[idx, 2] = int(raw_codes[1])  # E

            volt_buffer[idx, 0] = float(volts[2])   # Z
            volt_buffer[idx, 1] = float(volts[0])   # N
            volt_buffer[idx, 2] = float(volts[1])   # E

        elapsed_time = time.perf_counter() - start_time
        actual_fs = float(NUM_SAMPLES) / float(elapsed_time) if elapsed_time > 0.0 else float(SAMPLE_RATE)

        logger.info(f"Время записи: {elapsed_time:.3f} с")
        logger.info(f"Фактическая частота дискретизации: {actual_fs:.3f} Гц")

        header = "Ось | DC Offset, В | Peak-Peak, LSB | Peak-Peak, мВ | RMS, LSB | RMS, мВ | Dominant Freq, Гц"
        logger.info(header)
        logger.info("-" * 120)

        channel_names = ["Z", "N", "E"]
        summary = {}
        for ch_idx, ch_name in enumerate(channel_names):
            raw_codes = raw_buffer[:, ch_idx]
            volts = volt_buffer[:, ch_idx]

            raw_mean = float(np.mean(raw_codes))
            volt_mean = float(np.mean(volts))

            raw_detrend = raw_codes - raw_mean
            volt_detrend = volts - volt_mean

            pp_lsb = float(np.max(raw_detrend) - np.min(raw_detrend))
            pp_mv = float(np.max(volt_detrend) - np.min(volt_detrend)) * 1000.0

            rms_lsb = float(np.sqrt(np.mean(raw_detrend ** 2)))
            rms_mv = float(np.sqrt(np.mean(volt_detrend ** 2))) * 1000.0

            f, psd = signal.welch(
                volt_detrend,
                fs=actual_fs,
                nperseg=NPERSEG,
                noverlap=NOVERLAP,
                window=WINDOW,
            )
            peak_idx = int(np.argmax(psd))
            peak_freq = float(f[peak_idx])

            line = (
                f"{ch_name:>4s} | {volt_mean:>12.6f} | {pp_lsb:>12.3f} | "
                f"{pp_mv:>11.3f} | {rms_lsb:>8.3f} | {rms_mv:>8.3f} | {peak_freq:>15.3f}"
            )
            logger.info(line)

            summary[ch_name] = {"rms_lsb": rms_lsb}

        logger.info("")
        logger.info("=== Инженерное резюме ===")
        if summary["Z"]["rms_lsb"] > 2.0:
            logger.warning(
                "Высокий уровень наводок! Проверьте экранирование кабеля "
                "или гальваническую развязку ISO7662"
            )
        elif summary["Z"]["rms_lsb"] < 1.0:
            logger.info(
                "Отлично: Измерительный тракт идеален, уровень шума на пределе чувствительности чипа"
            )
        else:
            logger.info("Уровень шума находится в ожидаемом диапазоне для данного тракта")

        logger.info("")
        logger.info("Скрипт завершён.")

    except KeyboardInterrupt:
        logger.info("Прервано пользователем.")
    except Exception as exc:
        logger.error(f"Ошибка: {exc}")
        sys.exit(1)
    finally:
        if adc is not None:
            try:
                adc.close()
            except Exception as e:
                logger.warning(f"Ошибка закрытия ADC: {e}")
        elif pi is not None:
            try:
                pi.stop()
            except Exception as e:
                logger.warning(f"Ошибка остановки pigpio: {e}")


if __name__ == "__main__":
    config.setup_logging()
    main()
