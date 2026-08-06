#!/usr/bin/env python3
"""
acquisition.py v9.2
CPU affinity for DAQ thread, batch=20, optimized for single-core isolation.
ВСЕ сообщения → в логгер 'seismic' (ddd.log / error.log). Терминал чист.

ИСПРАВЛЕНИЕ v9.2:
  - Тайминг: next_time += interval с resync при drift > interval.
    Убран time.sleep для sub-ms ожиданий — заменён на busy-wait.
    Это устраняет накопление ошибок и "скачки" графика.
  - DIAG: time.monotonic() вместо time.time() — защита от скачков wall clock.
  - DIAG: логируем только если missed > 0 или sent отличается от expected > 5%.
"""
import time
import os
import logging
from PyQt5.QtCore import QThread, pyqtSignal

from AD7606B import AD7606B
import config

VERSION = "v9.2"

logger = logging.getLogger('seismic')


def debug(msg):
    """Пишет в ddd.log через логгер 'seismic'."""
    logger.debug(msg)


class DataAcquisitionThread(QThread):
    """
    Поток сбора данных с AD7606B.

    Сигнал data_ready испускает 5 аргументов:
        batch_volts, batch_raws, batch_ts, sample_counter, error_counter

    Для подключения к processor.py используйте lambda:
        daq.data_ready.connect(
            lambda volts, raws, ts, cnt, err: processor.process_batch(raws, ts)
        )
    """
    data_ready = pyqtSignal(list, list, list, int, int)

    def __init__(self, sample_rate=None, batch_size=20, parent=None):
        super().__init__(parent)
        self.sample_rate = sample_rate or config.SAMPLE_RATE
        self.batch_size = batch_size
        self._running = False
        self.pi = None
        self.adc = None
        self.sample_counter = 0
        self.error_counter = 0

    def run(self):
        self._running = True

        if config.DAQ_CPU_CORES:
            try:
                os.sched_setaffinity(0, config.DAQ_CPU_CORES)
                logger.info(f"[DAQ {VERSION}] CPU affinity set to {config.DAQ_CPU_CORES}")
            except Exception as e:
                logger.warning(f"[DAQ {VERSION}] CPU affinity failed (need root): {e}")

        try:
            os.nice(-10)
            logger.info(f"[DAQ {VERSION}] Priority increased (nice=-10)")
        except Exception as e:
            logger.warning(f"[DAQ {VERSION}] Priority increase failed: {e}")

        import pigpio
        self.pi = pigpio.pi()
        if not self.pi.connected:
            logger.error(f"[DAQ {VERSION}] FATAL: pigpio not connected. Ensure sudo pigpiod is running!")
            return

        try:
            self.adc = AD7606B(self.pi, sample_rate=self.sample_rate)
        except Exception as e:
            logger.error(f"[DAQ {VERSION}] FATAL: ADC init failed: {e}")
            self.pi.stop()
            return

        logger.info(f"[DAQ {VERSION}] Started at {self.sample_rate} SPS, batch={self.batch_size}")

        interval = 1.0 / self.sample_rate
        next_time = time.perf_counter()
        sample_idx = 0

        # monotonic() для точных интервалов DIAG
        last_diag_time = time.monotonic()
        last_diag_counter = 0

        batch_volts = []
        batch_raws = []
        batch_ts = []

        while self._running:
            next_time += interval
            sample_idx += 1

            # --- Точный тайминг ---
            now = time.perf_counter()
            if now < next_time:
                # Для интервалов < 2 мс time.sleep неточен из-за гранулярности планировщика.
                # Используем busy-wait на выделенном ядре.
                while time.perf_counter() < next_time:
                    pass
            elif now > next_time + interval:
                # Отстали больше чем на один период — сбрасываем синхронизацию,
                # чтобы не накапливать ошибку.
                drift = now - next_time
                logger.warning(
                    f"[DAQ {VERSION}] Timing drift {drift*1000:.1f} ms "
                    f"({int(drift/interval)} samples), resyncing"
                )
                next_time = now + interval

            sample_utc = time.time()

            try:
                volts, raws = self.adc.read_4ch_fixed_delay()
                self.sample_counter += 1

                batch_volts.append(volts)
                batch_raws.append(raws)
                batch_ts.append(sample_utc)

                if len(batch_volts) >= self.batch_size:
                    self.data_ready.emit(batch_volts, batch_raws, batch_ts,
                                        self.sample_counter, self.error_counter)
                    batch_volts = []
                    batch_raws = []
                    batch_ts = []

            except Exception as e:
                self.error_counter += 1
                logger.error(f"[DAQ {VERSION}] Error {self.error_counter}/{self.sample_counter}: {e}")

            # --- DIAG каждые 10 сек ---
            now_mono = time.monotonic()
            elapsed = now_mono - last_diag_time
            if elapsed >= 10.0:
                expected = int(elapsed * self.sample_rate)
                actual = self.sample_counter - last_diag_counter
                missed = max(0, expected - actual)
                loss_pct = (missed / expected * 100) if expected > 0 else 0

                if missed > 0:
                    logger.warning(
                        f"[DAQ {VERSION}] DIAG {elapsed:.1f}s: expected={expected}, sent={actual}, "
                        f"missed={missed} ({loss_pct:.1f}%), errors={self.error_counter}"
                    )
                else:
                    debug(
                        f"[DAQ {VERSION}] DIAG {elapsed:.1f}s: expected={expected}, sent={actual}, "
                        f"missed={missed}, errors={self.error_counter}"
                    )

                last_diag_counter = self.sample_counter
                last_diag_time = now_mono

        # Отправляем остаток при остановке
        if batch_volts:
            self.data_ready.emit(batch_volts, batch_raws, batch_ts,
                                self.sample_counter, self.error_counter)

    def stop(self):
        logger.info(f"[DAQ {VERSION}] Stopping acquisition thread...")
        self._running = False
        self.wait(2000)
        if self.adc:
            try:
                self.adc.close()
            except Exception:
                pass
        if self.pi and self.pi.connected:
            self.pi.stop()
        logger.info(f"[DAQ {VERSION}] Acquisition thread stopped.")
