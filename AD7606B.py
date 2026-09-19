"""
AD7606B driver v9.4.0
SPI 1MHz Mode 2 — verified working with ISO7662 isolator.
ВСЕ каналы: ±2.5V (регистр 0x04 = 0x00).
ВСЕ сообщения → логгер 'seismic' (ddd.log / error.log). Терминал чист.
"""
import pigpio
import time
import numpy as np
import logging

import config

logger = logging.getLogger('seismic')


class AD7606B:
#    CONV_TIME_US = {
#        0x00: 5, 0x01: 10, 0x02: 20, 0x03: 35,
#        0x04: 70, 0x05: 140, 0x06: 280,
#    }

    def __init__(self, pi, convst=18, reset=27, busy=25, cs=17,
                 v_range=2.5, sample_rate=400, spi_speed_hz=1000000):
        self.pi = pi
        self.CONVST = convst
        self.RESET = reset
        self.BUSY = busy
        self.CS = cs
        self.v_range = v_range
        self.sample_rate = sample_rate
        self.spi_speed = spi_speed_hz
        self.osr = self._choose_oversampling(sample_rate)
#       self.conv_us = self.CONV_TIME_US.get(self.osr, 50)
        
        self.pi.set_mode(self.CONVST, pigpio.OUTPUT)
        self.pi.set_mode(self.RESET, pigpio.OUTPUT)
        self.pi.set_mode(self.CS, pigpio.OUTPUT)
        self.pi.set_mode(self.BUSY, pigpio.INPUT)
        self.pi.set_pull_up_down(self.BUSY, pigpio.PUD_OFF)

        self.pi.write(self.CONVST, 0)
        self.pi.write(self.RESET, 0)
        self.pi.write(self.CS, 1)

        if self.CS != 8:
            self.pi.set_mode(8, pigpio.INPUT)
            self.pi.set_pull_up_down(8, pigpio.PUD_OFF)

        self.spi_handle = self.pi.spi_open(0, self.spi_speed, 0x02)
        if self.spi_handle < 0:
            raise RuntimeError("pigpio SPI open failed")

        self._init_adc()

    def _init_adc(self):
        logger.info(f"[AD7606B {config.VERSION}] Init...")
        self._reset_adc()
        time.sleep(0.5)
        self._reg_read(0x02)
        self._configure_adc_registers()      # сначала настраиваем OSR и диапазон
        self._skip_dummy_frame()
        self.conv_us = self._measure_conversion_time()  # потом измеряем
        logger.info(f"[AD7606B {config.VERSION}] Ready (+-{self.v_range}V, SPS={self.sample_rate}, OSR=0x{self.osr:02X}, SPI={self.spi_speed/1000:.0f}kHz Mode2)")

    def _reset_adc(self):
        self.pi.write(self.RESET, 1)
        time.sleep(0.01)
        self.pi.write(self.RESET, 0)

    def _configure_adc_registers(self):
        self._reg_write(0x02, 0x08)
        self._reg_write(0x03, 0x00)
        self._reg_write(0x04, 0x00)
        self._reg_write(0x08, self.osr)
        self._reg_write(0x00, 0x00)

    def _choose_oversampling(self, sample_rate):
        if sample_rate <= 100: return 0x08   # 256×
        if sample_rate <= 200: return 0x07   # 128×
        if sample_rate <= 400: return 0x07   # 64×  ← или 0x08, если CPU тянет
        if sample_rate <= 600: return 0x04   # 16×
        return 0x02                          # 4×

    def _reg_write(self, addr, data):
        word = ((addr & 0x3F) << 8) | (data & 0xFF)
        self.pi.write(self.CS, 0)
        time.sleep(0.001)
        self.pi.spi_xfer(self.spi_handle, [word >> 8, word & 0xFF])
        time.sleep(0.001)
        self.pi.write(self.CS, 1)
        time.sleep(0.001)

    def _reg_read(self, addr):
        word = 0x4000 | ((addr & 0x3F) << 8)
        self.pi.write(self.CS, 0)
        time.sleep(0.001)
        self.pi.spi_xfer(self.spi_handle, [word >> 8, word & 0xFF])
        count, rx = self.pi.spi_xfer(self.spi_handle, [0x00, 0x00])
        time.sleep(0.001)
        self.pi.write(self.CS, 1)
        return rx[1] if count == 2 else 0

    def _skip_dummy_frame(self):
        self._trigger_conversion()
        time.sleep(0.005)
        self._read_raw_bytes()

    def _trigger_conversion(self, pulse_us=10):
        self.pi.gpio_trigger(self.CONVST, pulse_us, 1)

    def _read_raw_bytes(self):
        self.pi.write(self.CS, 0)
        count, raw_bytes = self.pi.spi_xfer(self.spi_handle, b'\x00' * 8)
        self.pi.write(self.CS, 1)
        if count != 8:
            raise RuntimeError(f"SPI read mismatch: expected 8, got {count}")
        return raw_bytes

    def _measure_conversion_time(self, n=25):
        """Однократное измерение реального времени конверсии при старте."""
        # Грубая оценка сверху по OSR (чтобы не ждать целую миллисекунду)
        rough_us = {
            0x00: 30,
            0x01: 40,
            0x02: 50,
            0x03: 70,
            0x04: 100,
            0x05: 180,
            0x06: 320,
        }.get(self.osr, 100)

        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            self._trigger_conversion()

            # Ждём немного больше rough-оценки
            deadline = time.perf_counter() + (rough_us + 30) * 1e-6
            while time.perf_counter() < deadline:
                pass

            self._read_raw_bytes()
            elapsed_us = (time.perf_counter() - t0) * 1e6
            times.append(elapsed_us)

        times = np.array(times)
        median = float(np.median(times))
        p95 = float(np.percentile(times, 95))
        maximum = float(np.max(times))

        used = p95 + max(6.0, p95 * 0.12)

        logger.info(
            f"[AD7606B {config.VERSION}] Conversion timing measured ({n} samples):\n"
            f"         median = {median:.1f} µs | 95% = {p95:.1f} µs | max = {maximum:.1f} µs\n"
            f"         → using {used:.1f} µs (includes safety margin)"
        )
        return used
        
    def _decode_raw_fast(self, raw_bytes):
        raw_codes = np.frombuffer(raw_bytes, dtype='>i2')
        scales = np.array([
            2.5 / 32768.0,
            2.5 / 32768.0,
            2.5 / 32768.0,
            2.5 / 32768.0,
        ], dtype=np.float64)
        volts = np.round(raw_codes * scales, 4)
        return volts.tolist(), raw_codes.tolist()

    def read_4ch_fixed_delay(self):
        self._trigger_conversion()
        deadline = time.perf_counter() + self.conv_us * 1e-6
        while time.perf_counter() < deadline:
            pass
        raw_bytes = self._read_raw_bytes()
        return self._decode_raw_fast(raw_bytes)

    def close(self):
        if hasattr(self, 'spi_handle') and self.spi_handle >= 0:
            self.pi.spi_close(self.spi_handle)
        logger.info(f"[AD7606B {config.VERSION}] Stopped.")
