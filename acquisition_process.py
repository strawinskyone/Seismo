"""
acquisition_process.py v9.4.0
DAQ в отдельном процессе (multiprocessing.Process).
Полностью избавляет от GIL contention между DAQ и GUI.
"""
import time
import os
import logging
import multiprocessing

from AD7606B import AD7606B
import config

logger = logging.getLogger('seismic')


class DataAcquisitionProcess(multiprocessing.Process):
    def __init__(self, data_queue, stop_event, sample_rate=None, batch_size=None):
        super().__init__()
        self.data_queue = data_queue
        self._stop_event = stop_event
        self.sample_rate = sample_rate or config.SAMPLE_RATE
        self.batch_size = batch_size or config.BATCH_SIZE
        self.sample_counter = 0
        self.error_counter = 0
        self._last_drift_log = 0.0

    def run(self):
        if config.DAQ_CPU_CORES:
            try:
                os.sched_setaffinity(0, config.DAQ_CPU_CORES)
                logger.info(f"[DAQ {config.VERSION}] CPU affinity set to {config.DAQ_CPU_CORES}")
            except Exception as e:
                logger.warning(f"[DAQ {config.VERSION}] CPU affinity failed (need root): {e}")

        try:
            os.nice(-10)
            logger.info(f"[DAQ {config.VERSION}] Priority increased (nice=-10)")
        except Exception as e:
            logger.warning(f"[DAQ {config.VERSION}] Priority increase failed: {e}")

        import pigpio
        pi = pigpio.pi()
        if not pi.connected:
            logger.error(f"[DAQ {config.VERSION}] FATAL: pigpio not connected. Ensure sudo pigpiod is running!")
            return

        try:
            adc = AD7606B(pi, sample_rate=self.sample_rate)
        except Exception as e:
            logger.error(f"[DAQ {config.VERSION}] FATAL: ADC init failed: {e}")
            pi.stop()
            return

        logger.info(f"[DAQ {config.VERSION}] Started at {self.sample_rate} SPS, batch={self.batch_size}")

        interval = 1.0 / self.sample_rate
        next_time = time.perf_counter()
        last_diag_time = time.monotonic()
        last_diag_counter = 0

        batch_volts = []
        batch_raws = []
        batch_ts = []

        while not self._stop_event.is_set():
            next_time += interval
            now = time.perf_counter()
            if now < next_time:
                sleep_time = next_time - now - 0.0002
                if sleep_time > 0:
                    time.sleep(sleep_time)
                while time.perf_counter() < next_time:
                    pass
            elif now > next_time + interval:
                drift = now - next_time
                drift_ms = drift * 1000.0
                samples_missed = int(drift / interval)
                now_t = time.time()
                if now_t - self._last_drift_log >= 5.0:
                    if drift_ms > 20.0:
                        logger.warning(
                            f"[DAQ {config.VERSION}] Timing drift {drift_ms:.1f} ms "
                            f"({samples_missed} samples), resyncing"
                        )
                        self._last_drift_log = now_t
                    elif drift_ms > 5.0:
                        logger.debug(
                            f"[DAQ {config.VERSION}] Timing drift {drift_ms:.1f} ms "
                            f"({samples_missed} samples), resyncing"
                        )
                        self._last_drift_log = now_t
                next_time = now + interval

            try:
                volts, raws = adc.read_4ch_fixed_delay()
                sample_utc = time.time()
                self.sample_counter += 1
                batch_volts.append(volts)
                batch_raws.append(raws)
                batch_ts.append(sample_utc)

                if len(batch_volts) >= self.batch_size:
                    try:
                        self.data_queue.put_nowait(
                            (batch_volts, batch_raws, batch_ts,
                             self.sample_counter, self.error_counter)
                        )
                    except Exception:
                        logger.warning(
                            f"[DAQ {config.VERSION}] Queue full, dropping batch "
                            f"(samples={self.sample_counter})"
                        )
                    batch_volts = []
                    batch_raws = []
                    batch_ts = []
            except Exception as e:
                self.error_counter += 1
                logger.error(f"[DAQ {config.VERSION}] Error {self.error_counter}/{self.sample_counter}: {e}")

            now_mono = time.monotonic()
            elapsed = now_mono - last_diag_time
            if elapsed >= 300.0:
                expected = int(elapsed * self.sample_rate)
                actual = self.sample_counter - last_diag_counter
                missed = max(0, expected - actual)
                loss_pct = (missed / expected * 100) if expected > 0 else 0
                if missed > 0 or loss_pct > 5.0:
                    logger.warning(
                        f"[DAQ {config.VERSION}] DIAG {elapsed:.1f}s: expected={expected}, sent={actual}, "
                        f"missed={missed} ({loss_pct:.1f}%), errors={self.error_counter}"
                    )
                else:
                    logger.debug(
                        f"[DAQ {config.VERSION}] DIAG {elapsed:.1f}s: expected={expected}, sent={actual}, "
                        f"missed={missed}, errors={self.error_counter}"
                    )
                last_diag_counter = self.sample_counter
                last_diag_time = now_mono

        if batch_volts:
            try:
                self.data_queue.put_nowait(
                    (batch_volts, batch_raws, batch_ts,
                     self.sample_counter, self.error_counter)
                )
            except Exception:
                pass

        if adc:
            try:
                adc.close()
            except Exception:
                pass
        if pi and pi.connected:
            pi.stop()
        logger.info(f"[DAQ {config.VERSION}] Acquisition process stopped.")
