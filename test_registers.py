import logging
import pigpio
import config          # <-- добавлено
config.setup_logging() # <-- добавлено
from AD7606B import AD7606B

logger = logging.getLogger('seismic')

pi = pigpio.pi()
adc = AD7606B(pi, sample_rate=400)

logger.info("")
logger.info("=== ТЕСТ ЗАПИСИ/ЧТЕНИЯ РЕГИСТРОВ ===")
EXPECTED = {
    0x02: 0x08,
    0x03: 0x00,
    0x04: 0x00,   # <-- все каналы ±2.5V
    0x08: 0x02,
    0x00: 0x00,
}
for addr, expected in EXPECTED.items():
    v = adc._reg_read(addr)
    status = 'OK' if v == expected else 'FAIL'
    logger.info(f"REG 0x{addr:02X}: ожидали 0x{expected:02X}, прочитали 0x{v:02X}  {status}")

logger.info("")
logger.info("=== ДАННЫЕ 4 КАНАЛОВ (single-shot) ===")
volts, codes = adc.read_4ch_fixed_delay()
logger.info(f"CH1:{codes[0]:+6d} CH2:{codes[1]:+6d} CH3:{codes[2]:+6d} CH4:{codes[3]:+6d}")
logger.info(f"VOLTS: X={volts[0]:+.4f} Y={volts[1]:+.4f} Z={volts[2]:+.4f} W={volts[3]:+.4f}")

adc.close()
pi.stop()
