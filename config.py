# Требуется: pip install obspy scipy numpy PyQt5 pyqtgraph
#
# Полный конфигурационный файл сейсмостанции.
# ВСЕ значения задаются здесь. Нигде в коде нет "магических чисел".
# РАЗДЕЛЫ СГРУППИРОВАНЫ ПО СМЫСЛУ.
#
# v9.6.x: часть констант вычисляется автоматически в разделе
# «ВЫЧИСЛЯЕМЫЕ КОНСТАНТЫ». Меняйте только БАЗОВЫЕ параметры
# (секунды, Гц, мВ, км, В) — производные пересчитаются сами.

import logging
from logging.handlers import RotatingFileHandler

# ==================== ОБЩИЕ / СИСТЕМА ====================
VERSION = "v9.6.11"
SAMPLE_RATE = 400                     # Гц. Допустимые: 100, 200, 300, 400, 600.
                                      # 400 Гц — оптимум для локальной сейсмики.
                                      # ВАЖНО: с OSR 128x (0x07) реальная частота = 400.03 Гц,
                                      # проверено 2026-09-18. С OSR 256x — падала до 327 Гц.

# === CPU AFFINITY (Raspberry Pi 4: ядра 0-3) ===
DAQ_CPU_CORES = {3}
GUI_CPU_CORES = {1}
HEAVY_PROCESS_CPU_CORES = {2}

# === ПРОЦЕССОР ===
HEAVY_PROCESS_INTERVAL_SEC = 0.35     # интервал тяжёлой обработки, сек.

# === ADC (AD7606B) ===
ADC_SCALE_V = 2.5 / 32768.0           # 1 LSB = 76.29 мкВ при ±2.5 В

# ==================== СТАРТ / ПРОГРЕВ ====================
ADC_WARMUP_SEC = 3.0

# === ДАТЧИК ВОДЫ ===
WATER_ALARM_THRESHOLD_V = 1.5

# ==================== SNAPSHOT / QUEUE ====================
SNAPSHOT_PRE_P_SEC = 5.0
SNAPSHOT_POST_P_SEC = 15.0
QUEUE_MAXSIZE = 3

# ==================== ОСЦИЛЛОГРАФ / КАРУСЕЛЬ ====================
TIME_SCALE = 90
CAROUSEL_PANELS = 4
MAP_GRAPH_RATIO = 5
GRAPH_SENSITIVITY_MV = 500.0
CAROUSEL_DOWNSAMPLE = 120
LINE_WIDTH = 1.0
GRAPH_POINT_SIZE = 4
GRAPH_BACKGROUND_COLOR = "#0a0a0a"
GRAPH_LINE_COLOR = "#00ff64"
GRAPH_POINT_COLOR = "#ff3232"

# ==================== КАРТА ====================
MAP_IMAGE_FILE = "map_20.jpg"
MAP_MAX_DISTANCE_KM = 20              # радиус основной карты (км)
MAP_ARROW_MAX_DISTANCE_KM = 50        # максимальная дистанция для стрелок (км)
MAP_STEP_KM = 2
MAP_SCALE_KM_PER_PIXEL = 40.0 / 1484
MAP_MIN_SCALE = 0.2
MAP_MAX_SCALE = 2.0
STATION_OFFSET_X = 0
STATION_OFFSET_Y = 0
FADE_OUT_SECONDS = 60
PULSE_FADE_SECONDS = 4.0
MAX_EVENTS_ON_MAP = 10
DEAD_ZONE_KM = 1.0                    # мёртвая зона (км). События ближе — игнорируются.
MAP_LABEL_FONT_SIZE = 12
MAP_ARROW_WIDTH = 3

# === ПУЛЬС СОБЫТИЯ (v9.6.x) ===
PULSE_USE_MAGNITUDE = False
PULSE_BASE_RADIUS_KM = 0.5
PULSE_MAG_SCALE = 0.5
PULSE_MAX_RADIUS_KM = 2.0

# ==================== ПОРОГИ ДЕТЕКЦИИ ====================
EVENT_THRESHOLD_MV = 0.5              # абсолютный порог для карты, мВ.
EVENT_DEBOUNCE_MS = 1000

# ==================== ОРИЕНТАЦИЯ ====================
AZIMUTH_OFFSET = 0.0

# ==================== ФИЛЬТРЫ ====================
# Полосовой фильтр для P-волны (refinement в processor.py).
FILTER_FREQMIN = 4.0
FILTER_FREQMAX = 28.0

# Полосовой фильтр для S-волны (S-пикер в heavy_worker.py).
# v9.6.0: полоса S вынесена из резонансной зоны геофона (2.5-2.8 Гц).
S_FILTER_FREQMIN = 1.5
S_FILTER_FREQMAX = 8.0

# ==================== ИНТЕГРИРОВАНИЕ СИГНАЛА ====================
# v9.6.0: геофон в рабочей полосе (выше f0) выдаёт СКОРОСТЬ.
# Для физически корректной амплитуды и формы S-волны — интегрируем в СМЕЩЕНИЕ.
INTEGRATE_FOR_S = False               # True = интегрировать (скорость → смещение)
INTEGRATE_TREND_REMOVE_SEC = 5.0      # период удаления линейного тренда при интегрировании.

# ==================== STA/LTA (P-волна) ====================
P_STA_SEC = 0.1
P_LTA_SEC = 4.0
P_TRIGGER_RATIO = 1.5
P_DETRIGGER_RATIO = 1.05

# === P-ДЕТЕКЦИЯ: АБСОЛЮТНЫЕ ПОРОГИ (v9.6.0) ===
# v9.6.0: детекция работает В ВОЛЬТАХ, не в SNR. Все пороги — абсолютные.
# Это устраняет проблему «заморозки RMS» и несогласованности порогов.
P_DETECT_ABS_MIN_V = 0.05             # В. Минимальная |Z| для P-детекции.
                                      # Отсекает фоновый шум (RMS ~4 мВ).
P_DETECT_ABS_MIN_D = 0.02             # В. Минимальная производная |Z[i]-Z[i-N]| за 10 мс.
                                      # Отсекает медленные дрейфы и одиночные спайки.
P_DETECT_DIFF_N = 4                   # окно производной в отсчётах (10 мс при 400 SPS).

# === P_END: ОТНОСИТЕЛЬНЫЙ ПОРОГ (v9.6.0) ===
# Абсолютный порог P_END_ABS_V вычисляется ниже из P_DETECT_ABS_MIN_V.
P_END_REL_BASE = 0.30                 # базовая доля от пика (30%).
P_END_REL_SNR_FACTOR = 0.20           # коррекция по SNR (legacy, не используется).

# ==================== STA/LTA (S-волна) ====================
S_STA_SEC = 0.25
S_LTA_SEC = 12.0
S_TRIGGER_RATIO = 4.5
S_DETRIGGER_RATIO = 1.0

# ==================== ВРЕМЕННЫЕ ОКНА P-S ====================
# MIN_P_S_TIME_SEC и MAX_P_S_TIME_SEC вычисляются автоматически
# (в разделе «ВЫЧИСЛЯЕМЫЕ КОНСТАНТЫ») из DEAD_ZONE_KM, MAP_ARROW_MAX_DISTANCE_KM
# и VP_VS_FACTOR (зависит от VP и VS).

# v9.6.0: S-поиск стартует сразу после P + короткая задержка,
# а не после p_end + 0.5 с (p_end часто затягивается).
S_SEARCH_START_AFTER_P_SEC = 0.2      # старт S-поиска через 0.2 с после p_idx.
S_SEARCH_POST_P_END_SEC = 0.5         # legacy, оставлен для совместимости.

# ==================== СКОРОСТИ ВОЛН ====================
VP = 5.8                              # P-волна, км/с. Влияет на VP_VS_FACTOR.
VS = 3.3                              # S-волна, км/с.

# ==================== МОДЕЛЬ ГЛУБИНЫ ====================
DEPTH_FACTOR = 0.05
DEPTH_OFFSET = 0.0

# ==================== ХАРАКТЕРИСТИКИ ДАТЧИКА ====================
GEOPHONE_SENSITIVITY = 29.2
GEOPHONE_SENSITIVITY_N = 28.48
GEOPHONE_SENSITIVITY_E = 29.07
GEOPHONE_SENSITIVITY_Z = 32.13

GAIN_CORRECTION_N = 1.0
GAIN_CORRECTION_E = 1.0
GAIN_CORRECTION_Z = 1.0

# === АЗИМУТ: ВЗВЕШИВАНИЕ КАНАЛОВ (v9.6.0) ===
AZIMUTH_WEIGHT_BY_RMS = True          # True = взвешивать N и E по 1/RMS
AZIMUTH_WIN_SEC = 0.5                 # окно PCA для азимута, сек.

# ==================== FFT / WELCH ====================
FFT_NPERSEG = 512
FFT_NOVERLAP = 256
FFT_WINDOW = 'hann'

EXPLOSION_SPECTRAL_THRESHOLD = 2.0
EXPLOSION_DOMINANT_FREQ_MIN = 6.0

# ==================== HEAVY WORKER S-PICKER ====================
S_PICKER_STA_SEC = 0.5
S_PICKER_LTA_SEC = 4.0
S_PICKER_TRIGGER = 1.3                # v9.6.0: снижено с 4.0 — S на смещении слабее.
S_PICKER_DETRIGGER = 0.8
S_USE_RMS_WEIGHTING = True           # v9.6.x: отключено (тест).
H_DISPLAY_STA_SEC = 0.15
H_DISPLAY_LTA_SEC = 4.0

# ==================== АРХИВ ====================
ARCHIVE_FOLDER = "archive"

# ==================== ТАЙМИНГИ ИНТЕРФЕЙСА ====================
MAP_UPDATE_MS = 1000
RESULT_TIMER_MS = 200
JOIN_TIMEOUT_SEC = 3.0
REFRESH_TIMER_MS = 400
BLINK_TIMER_MS = 500

# ==================== DAQ ПАРАМЕТРЫ ====================
DAQ_QUEUE_MAXSIZE = 200
BATCH_SIZE = 40

# ==================== ADAPTIVE ENVELOPE ====================
ENVELOPE_WIN_SEC = 0.15
NOISE_ESTIMATE_SEC = 10.0
NOISE_PERCENTILE = 50.0
SNR_THRESHOLD = 2.0                   # legacy, не используется в v9.6.0
ONSET_DERIVATIVE_FACTOR = 5.0         # legacy, заменено на P_DETECT_ABS_MIN_D
ONSET_GUARD_SEC = 0.5
MAX_TRACKERS = 3

P_MIN_DURATION_SEC = 0.3
P_MAX_DURATION_SEC = 5.0
NOISE_UPDATE_INTERVAL_SEC = 5.0
PROC_DIAG_INTERVAL_SEC = 0.0          # v9.6.x: интервал диагностики processor, сек.
                                      # 0 = отключено. 5 = отладка. 30+ = продакшн.


# ============================================================================
# ==================== ВЫЧИСЛЯЕМЫЕ КОНСТАНТЫ (НЕ РЕДАКТИРОВАТЬ) ==============
# ============================================================================
# Все значения ниже вычисляются автоматически из базовых параметров выше.
# При изменении базовых (секунды, Гц, км, В) — пересчитываются сами.

# --- Физические константы из VP/VS ---
# Коэффициент перевода P-S времени в дистанцию: distance = ΔP-S · VP_VS_FACTOR
VP_VS_FACTOR = (VP * VS) / (VP - VS)   # = 7.656 при VP=5.8, VS=3.3

# --- P-S временные окна (в секундах) ---
# Привязаны к DEAD_ZONE_KM и MAP_ARROW_MAX_DISTANCE_KM через VP_VS_FACTOR.
# Жёсткий минимум 0.2 с — ниже физически невозможно для реального S.
MIN_P_S_TIME_SEC = max(0.2, DEAD_ZONE_KM / VP_VS_FACTOR)
MAX_P_S_TIME_SEC = MAP_ARROW_MAX_DISTANCE_KM / VP_VS_FACTOR

# --- Порог окончания P (в вольтах) ---
# Половина от порога начала, минимум 5 мВ.
P_END_ABS_V = max(P_DETECT_ABS_MIN_V * 0.5, 0.005)

# --- Экран / визуализация ---
SAMPLES_PER_SCREEN = int(TIME_SCALE * SAMPLE_RATE)

# --- Буферы processor.py ---
BUFFER_SIZE = int(max(80.0, SNAPSHOT_PRE_P_SEC + SNAPSHOT_POST_P_SEC + 20.0) * SAMPLE_RATE)
ENVELOPE_WIN_N = max(1, int(ENVELOPE_WIN_SEC * SAMPLE_RATE))
NOISE_ESTIMATE_N = int(NOISE_ESTIMATE_SEC * SAMPLE_RATE)
DERIV_HISTORY_N = max(100, NOISE_ESTIMATE_N)
ENVELOPE_DELAY_N = max(1, int(0.1 * SAMPLE_RATE))
HEAVY_PROCESS_INTERVAL_N = int(HEAVY_PROCESS_INTERVAL_SEC * SAMPLE_RATE)
NOISE_UPDATE_INTERVAL_N = int(NOISE_UPDATE_INTERVAL_SEC * SAMPLE_RATE)

# --- STA/LTA в сэмплах (P-уточнение в processor) ---
P_STA_N = max(1, int(P_STA_SEC * SAMPLE_RATE))
P_LTA_N = max(1, int(P_LTA_SEC * SAMPLE_RATE))
P_DETECT_DIFF_N = max(1, int(P_DETECT_DIFF_N))

# --- STA/LTA в сэмплах (S-пикер в heavy_worker) ---
S_PICKER_STA_N = max(1, int(S_PICKER_STA_SEC * SAMPLE_RATE))
S_PICKER_LTA_N = max(1, int(S_PICKER_LTA_SEC * SAMPLE_RATE))

# --- H-STA/LTA для визуализации (в сэмплах) ---
H_DISPLAY_STA_N = max(1, int(H_DISPLAY_STA_SEC * SAMPLE_RATE))
H_DISPLAY_LTA_N = max(1, int(H_DISPLAY_LTA_SEC * SAMPLE_RATE))

# --- Snapshot границы ---
SNAPSHOT_PRE_P_N = int(SNAPSHOT_PRE_P_SEC * SAMPLE_RATE)
SNAPSHOT_POST_P_N = int(SNAPSHOT_POST_P_SEC * SAMPLE_RATE)

# --- P-S временные окна в сэмплах ---
MIN_P_S_TIME_N = int(MIN_P_S_TIME_SEC * SAMPLE_RATE)
MAX_P_S_TIME_N = int(MAX_P_S_TIME_SEC * SAMPLE_RATE)
S_SEARCH_START_AFTER_P_N = int(S_SEARCH_START_AFTER_P_SEC * SAMPLE_RATE)
S_SEARCH_POST_P_END_N = int(S_SEARCH_POST_P_END_SEC * SAMPLE_RATE)

# --- Длительности фаз ---
P_MIN_DURATION_N = int(P_MIN_DURATION_SEC * SAMPLE_RATE)
P_MAX_DURATION_N = int(P_MAX_DURATION_SEC * SAMPLE_RATE)
ONSET_GUARD_N = int(ONSET_GUARD_SEC * SAMPLE_RATE)

# --- Прочее ---
ADC_WARMUP_N = int(ADC_WARMUP_SEC * SAMPLE_RATE)
EVENT_DEBOUNCE_N = int(EVENT_DEBOUNCE_MS / 1000.0 * SAMPLE_RATE)

# --- heavy_worker специфичные ---
AZIMUTH_WIN_N = max(1, int(AZIMUTH_WIN_SEC * SAMPLE_RATE))
RSAM_WIN_N = max(1, int(2.0 * SAMPLE_RATE))
MAX_P_S_SEARCH_N = int((MAX_P_S_TIME_SEC + 2.0) * SAMPLE_RATE)

INTEGRATE_TREND_REMOVE_N = int(INTEGRATE_TREND_REMOVE_SEC * SAMPLE_RATE)



# ============================================================================
# ==================== ВАЛИДАЦИЯ КОНФИГУРАЦИИ ================================
# ============================================================================

def validate_config():
    """
    Проверяет корректность базовых параметров при старте.
    Вызывать один раз в main.py перед запуском процессов.
    При ошибке — AssertionError с понятным сообщением.
    """
    assert SNAPSHOT_PRE_P_SEC >= P_LTA_SEC + 1.0, \
        f"SNAPSHOT_PRE_P_SEC ({SNAPSHOT_PRE_P_SEC}) must cover P_LTA_SEC ({P_LTA_SEC}) + 1s"
    assert SNAPSHOT_POST_P_SEC >= MAX_P_S_TIME_SEC + 3.0, \
        f"SNAPSHOT_POST_P_SEC ({SNAPSHOT_POST_P_SEC}) must cover MAX_P_S_TIME_SEC ({MAX_P_S_TIME_SEC}) + 3s buffer"
    assert P_LTA_N > P_STA_N, \
        f"P_LTA_N ({P_LTA_N}) must be > P_STA_N ({P_STA_N})"
    assert S_PICKER_LTA_N > S_PICKER_STA_N, \
        f"S_PICKER_LTA_N ({S_PICKER_LTA_N}) must be > S_PICKER_STA_N ({S_PICKER_STA_N})"
    assert H_DISPLAY_LTA_N > H_DISPLAY_STA_N, \
        f"H_DISPLAY_LTA_N ({H_DISPLAY_LTA_N}) must be > H_DISPLAY_STA_N ({H_DISPLAY_STA_N})"
    assert BUFFER_SIZE > SNAPSHOT_PRE_P_N + SNAPSHOT_POST_P_N, \
        f"BUFFER_SIZE ({BUFFER_SIZE}) must fit full snapshot ({SNAPSHOT_PRE_P_N + SNAPSHOT_POST_P_N})"
    assert FFT_NPERSEG <= SNAPSHOT_PRE_P_N + SNAPSHOT_POST_P_N, \
        f"FFT_NPERSEG ({FFT_NPERSEG}) must fit inside snapshot ({SNAPSHOT_PRE_P_N + SNAPSHOT_POST_P_N})"
    assert SAMPLE_RATE in (100, 200, 300, 400, 600), \
        f"Unsupported sample rate: {SAMPLE_RATE}. Allowed: 100, 200, 300, 400, 600"
    assert P_TRIGGER_RATIO > P_DETRIGGER_RATIO, \
        f"P_TRIGGER_RATIO ({P_TRIGGER_RATIO}) must be > P_DETRIGGER_RATIO ({P_DETRIGGER_RATIO})"
    assert S_TRIGGER_RATIO > S_DETRIGGER_RATIO, \
        f"S_TRIGGER_RATIO ({S_TRIGGER_RATIO}) must be > S_DETRIGGER_RATIO ({S_DETRIGGER_RATIO})"
    assert S_PICKER_TRIGGER > S_PICKER_DETRIGGER, \
        f"S_PICKER_TRIGGER ({S_PICKER_TRIGGER}) must be > S_PICKER_DETRIGGER ({S_PICKER_DETRIGGER})"
    assert MAP_MIN_SCALE > 0 and MAP_MAX_SCALE > MAP_MIN_SCALE, \
        "MAP_MIN_SCALE must be > 0 and MAP_MAX_SCALE > MAP_MIN_SCALE"
    assert ENVELOPE_WIN_SEC > 0, "ENVELOPE_WIN_SEC must be > 0"
    assert NOISE_ESTIMATE_SEC >= 5.0, \
        f"NOISE_ESTIMATE_SEC ({NOISE_ESTIMATE_SEC}) too small — минимум 5 секунд"
    assert CAROUSEL_DOWNSAMPLE >= 1, "CAROUSEL_DOWNSAMPLE must be >= 1"
    assert BATCH_SIZE >= 1, "BATCH_SIZE must be >= 1"
    assert DAQ_QUEUE_MAXSIZE >= 10, "DAQ_QUEUE_MAXSIZE must be >= 10"
    assert QUEUE_MAXSIZE >= 1, "QUEUE_MAXSIZE must be >= 1"
    assert MAX_TRACKERS >= 1, "MAX_TRACKERS must be >= 1"
    assert DEAD_ZONE_KM >= 0, "DEAD_ZONE_KM must be >= 0"
    assert VP > VS > 0, \
        f"VP ({VP}) must be > VS ({VS}) > 0"
    assert HEAVY_PROCESS_INTERVAL_SEC >= 0.05, \
        f"HEAVY_PROCESS_INTERVAL_SEC ({HEAVY_PROCESS_INTERVAL_SEC}) too small — минимум 0.05 с"
    assert HEAVY_PROCESS_INTERVAL_SEC <= 2.0, \
        f"HEAVY_PROCESS_INTERVAL_SEC ({HEAVY_PROCESS_INTERVAL_SEC}) too large — максимум 2.0 с"
    assert EVENT_DEBOUNCE_MS >= 100, \
        f"EVENT_DEBOUNCE_MS ({EVENT_DEBOUNCE_MS}) too small — минимум 100 мс"
    assert FFT_NPERSEG >= 64, \
        f"FFT_NPERSEG ({FFT_NPERSEG}) too small — минимум 64 отсчёта"
    assert FFT_NOVERLAP < FFT_NPERSEG, \
        f"FFT_NOVERLAP ({FFT_NOVERLAP}) must be < FFT_NPERSEG ({FFT_NPERSEG})"
    assert FILTER_FREQMIN < FILTER_FREQMAX, \
        f"FILTER_FREQMIN ({FILTER_FREQMIN}) must be < FILTER_FREQMAX ({FILTER_FREQMAX})"
    assert FILTER_FREQMIN > 0, "FILTER_FREQMIN must be > 0"
    assert FILTER_FREQMAX <= SAMPLE_RATE / 2.0, \
        f"FILTER_FREQMAX ({FILTER_FREQMAX}) must be <= Nyquist ({SAMPLE_RATE/2.0} Гц)"
    assert S_FILTER_FREQMIN < S_FILTER_FREQMAX, \
        f"S_FILTER_FREQMIN ({S_FILTER_FREQMIN}) must be < S_FILTER_FREQMAX ({S_FILTER_FREQMAX})"
    assert S_FILTER_FREQMAX <= SAMPLE_RATE / 2.0, \
        f"S_FILTER_FREQMAX ({S_FILTER_FREQMAX}) must be <= Nyquist ({SAMPLE_RATE/2.0} Гц)"
    assert GRAPH_SENSITIVITY_MV > 0, "GRAPH_SENSITIVITY_MV must be > 0"
    assert TIME_SCALE > 0, "TIME_SCALE must be > 0"
    assert ADC_WARMUP_SEC >= 0, "ADC_WARMUP_SEC must be >= 0"
    assert P_MIN_DURATION_SEC > 0, "P_MIN_DURATION_SEC must be > 0"
    assert P_MAX_DURATION_SEC > P_MIN_DURATION_SEC, \
        f"P_MAX_DURATION_SEC ({P_MAX_DURATION_SEC}) must be > P_MIN_DURATION_SEC ({P_MIN_DURATION_SEC})"
    assert ONSET_GUARD_SEC >= 0, "ONSET_GUARD_SEC must be >= 0"
    assert SNR_THRESHOLD > 0, "SNR_THRESHOLD must be > 0"
    assert 0 < P_END_REL_BASE <= 1.0, "P_END_REL_BASE must be in (0, 1.0]"
    assert P_END_REL_SNR_FACTOR >= 0, "P_END_REL_SNR_FACTOR must be >= 0"
    assert P_DETECT_ABS_MIN_V > 0, "P_DETECT_ABS_MIN_V must be > 0"
    assert P_DETECT_ABS_MIN_D > 0, "P_DETECT_ABS_MIN_D must be > 0"
    assert P_DETECT_DIFF_N >= 1, "P_DETECT_DIFF_N must be >= 1"
    assert P_END_ABS_V > 0, "P_END_ABS_V must be > 0"
    assert S_SEARCH_START_AFTER_P_SEC >= 0, "S_SEARCH_START_AFTER_P_SEC must be >= 0"
    assert AZIMUTH_WIN_SEC > 0, "AZIMUTH_WIN_SEC must be > 0"
    assert INTEGRATE_TREND_REMOVE_SEC > 0, "INTEGRATE_TREND_REMOVE_SEC must be > 0"
    assert WATER_ALARM_THRESHOLD_V >= 0, "WATER_ALARM_THRESHOLD_V must be >= 0"
    assert MAP_MAX_DISTANCE_KM > 0, "MAP_MAX_DISTANCE_KM must be > 0"
    assert MAP_ARROW_MAX_DISTANCE_KM >= MAP_MAX_DISTANCE_KM, \
        f"MAP_ARROW_MAX_DISTANCE_KM ({MAP_ARROW_MAX_DISTANCE_KM}) must be >= MAP_MAX_DISTANCE_KM ({MAP_MAX_DISTANCE_KM})"
    assert FADE_OUT_SECONDS > 0, "FADE_OUT_SECONDS must be > 0"
    assert PULSE_FADE_SECONDS > 0, "PULSE_FADE_SECONDS must be > 0"
    assert MAX_EVENTS_ON_MAP >= 1, "MAX_EVENTS_ON_MAP must be >= 1"
    assert MAP_STEP_KM > 0, "MAP_STEP_KM must be > 0"
    assert MAP_SCALE_KM_PER_PIXEL > 0, "MAP_SCALE_KM_PER_PIXEL must be > 0"
    assert CAROUSEL_PANELS >= 1, "CAROUSEL_PANELS must be >= 1"
    assert MAP_GRAPH_RATIO >= 1, "MAP_GRAPH_RATIO must be >= 1"
    assert GEOPHONE_SENSITIVITY_N > 0, "GEOPHONE_SENSITIVITY_N must be > 0"
    assert GEOPHONE_SENSITIVITY_E > 0, "GEOPHONE_SENSITIVITY_E must be > 0"
    assert GEOPHONE_SENSITIVITY_Z > 0, "GEOPHONE_SENSITIVITY_Z must be > 0"
    assert GAIN_CORRECTION_N > 0, "GAIN_CORRECTION_N must be > 0"
    assert GAIN_CORRECTION_E > 0, "GAIN_CORRECTION_E must be > 0"
    assert GAIN_CORRECTION_Z > 0, "GAIN_CORRECTION_Z must be > 0"

class MaxLevelFilter(logging.Filter):
    """
    Фильтр уровня: пропускает записи с levelno <= max_level.
    Используется для разделения потоков:
    ddd_handler (DEBUG+INFO) vs err_handler (WARNING+).
    """
    def __init__(self, max_level):
        self.max_level = max_level
        super().__init__()

    def filter(self, record):
        return record.levelno <= self.max_level


def setup_logging():
    """
    Настраивает логгер 'seismic' с ротацией и разделением по уровням.

    ddd.log   — DEBUG и INFO (без WARNING и выше).
    error.log — WARNING, ERROR, CRITICAL (без дублирования).
    events.log — создаётся отдельно в main.py (только события на карту).

    v9.6.x: логгер 'seismic.flush' отключён по умолчанию.
    Для отладки flush — раскомментировать строку с setLevel(logging.DEBUG).
    """
    logger = logging.getLogger('seismic')
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # v9.6.17: mode='w' — при старте main.py старые логи обнуляются.
    # events.log не трогаем — он пишется отдельно, в append-режиме.
    ddd_handler = RotatingFileHandler(
        'ddd.log', mode='w', maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
    )
    ddd_handler.setLevel(logging.DEBUG)
    ddd_handler.addFilter(MaxLevelFilter(logging.INFO))
    ddd_handler.setFormatter(formatter)
    logger.addHandler(ddd_handler)

    err_handler = RotatingFileHandler(
        'error.log', mode='w', maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
    )
    err_handler.setLevel(logging.WARNING)
    err_handler.setFormatter(formatter)
    logger.addHandler(err_handler)

    # --- v9.6.x: отдельный логгер для flush-сообщений ---
    # По умолчанию отключён, чтобы не забивать ddd.log.
    # Для отладки: раскомментировать setLevel(logging.DEBUG).
    flush_logger = logging.getLogger('seismic.flush')
    flush_logger.setLevel(logging.CRITICAL + 1)   # полностью выключен
    # flush_logger.setLevel(logging.DEBUG)         # ← раскомментировать при отладке
    if not flush_logger.handlers:
        flush_logger.addHandler(ddd_handler)
        flush_logger.propagate = False

    return logger