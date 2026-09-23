# Требуется: pip install obspy scipy numpy PyQt5 pyqtgraph
#
# config.py v11.0.0 — оптимизированная версия.
# ВСЕ «магические числа» вынесены сюда. Базовые константы — физические
# (секунды, Гц, мВ, км, В). Производные вычисляются автоматически.

import os
import glob
import logging
from logging.handlers import RotatingFileHandler

# ==================== ОБЩИЕ / СИСТЕМА ====================
VERSION = "v11.0.0"
SAMPLE_RATE = 400                     # Гц. 100/200/300/400/600.

DAQ_CPU_CORES = {3}
GUI_CPU_CORES = {1}
HEAVY_PROCESS_CPU_CORES = {2}

HEAVY_PROCESS_INTERVAL_SEC = 0.35

# === ADC (AD7606B) ===
ADC_SCALE_V = 2.5 / 32768.0           # 1 LSB = 76.29 мкВ при ±2.5 В
ADC_WARMUP_SEC = 3.0

# === ДАТЧИК ВОДЫ ===
WATER_ALARM_THRESHOLD_V = 1.5

# ==================== SNAPSHOT / QUEUE ====================
SNAPSHOT_PRE_P_SEC = 5.0
SNAPSHOT_POST_P_SEC = 15.0            # верхняя граница, реальная — адаптивная.
SNAPSHOT_MIN_POST_SEC = 3.0           # v11.0.0: минимум после P_END.
QUEUE_MAXSIZE = 3
DAQ_QUEUE_MAXSIZE = 200
BATCH_SIZE = 40

# ==================== ОСЦИЛЛОГРАФ / КАРУСЕЛЬ ====================
TIME_SCALE = 90
CAROUSEL_PANELS = 4
CAROUSEL_DRAW_MODE = 'FULL' # или CURVE - для огибающей
MAP_GRAPH_RATIO = 5
GRAPH_SENSITIVITY_MV = 500.0
CAROUSEL_DOWNSAMPLE = 120
LINE_WIDTH = 1.0
GRAPH_POINT_SIZE = 4
GRAPH_BACKGROUND_COLOR = "#0a0a0a"
GRAPH_LINE_COLOR = "#00ff64"
GRAPH_POINT_COLOR = "#ff3232"

OSC_H_CENTER = -0.07
OSC_Z_CENTER = -0.03
OSC_Y_MAX = +0.15
OSC_Y_MIN = -0.20
OSC_CFT_CENTER = -0.16
OSC_CFT_SCALE = 3.0
OSC_MARKER_PAD_FRAC = 0.1

# ==================== КАРТА ====================
MAP_IMAGE_FILE = "map_20.jpg"
MAP_MAX_DISTANCE_KM = 20
MAP_ARROW_MAX_DISTANCE_KM = 50
MAP_STEP_KM = 2
MAP_SCALE_KM_PER_PIXEL = 40.0 / 1484
MAP_MIN_SCALE = 0.2
MAP_MAX_SCALE = 2.0
STATION_OFFSET_X = 0
STATION_OFFSET_Y = 0
FADE_OUT_SECONDS = 60
PULSE_FADE_SECONDS = 4.0
MAX_EVENTS_ON_MAP = 10
DEAD_ZONE_KM = 1.0
MAP_LABEL_FONT_SIZE = 12
MAP_ARROW_WIDTH = 3

# === ПУЛЬС СОБЫТИЯ ===
PULSE_USE_MAGNITUDE = False
PULSE_BASE_RADIUS_KM = 0.5
PULSE_MAG_SCALE = 0.5
PULSE_MAX_RADIUS_KM = 2.0

# ==================== ПОРОГИ ДЕТЕКЦИИ ====================
EVENT_THRESHOLD_MV = 20
EVENT_DEBOUNCE_MS = 10000

# ==================== ОРИЕНТАЦИЯ ====================
AZIMUTH_OFFSET = 0.0
AZIMUTH_WIN_SEC = 0.5

# ==================== ФИЛЬТРЫ ====================
FILTER_FREQMIN = 4.0
FILTER_FREQMAX = 28.0
S_FILTER_FREQMIN = 1.5
S_FILTER_FREQMAX = 8.0

# ==================== ИНТЕГРИРОВАНИЕ СИГНАЛА ====================
INTEGRATE_FOR_S = False
INTEGRATE_TREND_REMOVE_SEC = 5.0

# ==================== STA/LTA (P-волна) ====================
P_STA_SEC = 0.05
P_LTA_SEC = 4.0
P_TRIGGER_RATIO = 1.2
P_DETRIGGER_RATIO = 1.0
STA_LTA_RESET_TIMEOUT_SEC = 30.0      # v11.0.0: вынесено из processor.py
DETECT_WARMUP_LTA_MULT = 5.0          # v11.0.0: множитель P_LTA_SEC.

# === P-ДЕТЕКЦИЯ: АБСОЛЮТНЫЕ ПОРОГИ ===
P_DETECT_ABS_MIN_V = 0.02
P_DETECT_ABS_MIN_D = 0.01
P_DETECT_DIFF_N = 4

# === P_END: относительный порог ===
P_END_REL_BASE = 0.30
P_DROP_FROM_PEAK_FRAC = 0.90          # v11.0.0: было «магическое 0.90».
P_AFTER_PEAK_SEC = 0.1                # v11.0.0: было «магическое 0.1».
P_PEAKED_MAX_DURATION_FRAC = 0.5      # v11.0.0: было «магическое 0.5».

# ==================== S-ПИКЕР (heavy_worker) ====================
S_PICKER_STA_SEC = 0.5
S_PICKER_LTA_SEC = 4.0
S_PICKER_TRIGGER = 1.3
S_PICKER_DETRIGGER = 0.8
S_USE_RMS_WEIGHTING = False

# --- Адаптивная горизонтальная рамка ---
S_NOISE_PRE_WIN_SEC = 2.5
S_TRIGGER_FACTOR = 6.0
S_MIN_THRESHOLD_MV = 0.005
S_AMP_RATIO_MIN = 0.5

# --- Адаптивная вертикальная рамка (AIC) ---
AIC_WIN_POST_P_SEC = 12.0
AIC_MIN_POST_P_SEC = 0.05
AIC_USE_HILBERT = True
AIC_CFT_WEIGHT = 0.5
AIC_SHARPNESS_MIN = 0.5               # v11.0.0: было «магическое 0.5».
AIC_ZONE_POS_MIN = 0.02               # v11.0.0: границы zone_pos.
AIC_ZONE_POS_MAX = 0.98
AIC_CFT_TRIGGER_FRAC = 0.5            # cft_at_pick > S_PICKER_TRIGGER * frac.

# --- Окна амплитуд ---
AMP_WIN_SEC = 0.25                    # v11.0.0: было «магическое 0.25».

# --- Адаптивное завершение события (processor.py) ---
TAIL_DECAY_FACTOR = 1.8
TAIL_HOLD_SEC = 1.5
TAIL_MIN_POST_SEC = 3.0

# --- H-STA/LTA для визуализации ---
H_DISPLAY_STA_SEC = 0.15
H_DISPLAY_LTA_SEC = 4.0
H_ENV_WIN_SEC = 0.3                   # v11.0.0: было «магическое 0.3».
H_MIN_SUBWIN_SEC = 0.25               # v11.0.0: под-окно RMS в _noise_floor_h.

# ==================== КЛАССИФИКАЦИЯ ====================
EXPLOSION_SPECTRAL_THRESHOLD = 2.0
EXPLOSION_DOMINANT_FREQ_MIN = 6.0
RECTILINEARITY_MIN = 0.3              # v11.0.0: было «магическое 0.3».
CLASSIFY_BASE_CONFIDENCE = 0.75       # v11.0.0: было «магическое 0.75».
CLASSIFY_BONUS_RECT = 0.05
CLASSIFY_BONUS_DIST = 0.05
CLASSIFY_BONUS_S = 0.05
CLASSIFY_EXPLOSION_BONUS = 0.10
CLASSIFY_MAX_CONFIDENCE = 0.99

# ==================== СКОРОСТИ ВОЛН ====================
VP = 5.8
VS = 3.3

# ==================== МОДЕЛЬ ГЛУБИНЫ ====================
DEPTH_FACTOR = 0.05
DEPTH_OFFSET = 0.0

# ==================== ХАРАКТЕРИСТИКИ ДАТЧИКА ====================
GEOPHONE_SENSITIVITY_N = 2200
GEOPHONE_SENSITIVITY_E = 2000
GEOPHONE_SENSITIVITY_Z = 3200

GAIN_CORRECTION_N = 1.0
GAIN_CORRECTION_E = 1.0
GAIN_CORRECTION_Z = 1.0

# ==================== FFT / WELCH ====================
FFT_NPERSEG = 512
FFT_NOVERLAP = 256
FFT_WINDOW = 'hann'

# ==================== АРХИВ ====================
ARCHIVE_FOLDER = "archive"

# ==================== ТАЙМИНГИ ИНТЕРФЕЙСА ====================
MAP_UPDATE_MS = 1000
RESULT_TIMER_MS = 200
JOIN_TIMEOUT_SEC = 3.0
REFRESH_TIMER_MS = 400
BLINK_TIMER_MS = 500

# ==================== ADAPTIVE ENVELOPE / PROCESSOR ====================
ENVELOPE_WIN_SEC = 0.15
NOISE_ESTIMATE_SEC = 10.0
NOISE_PERCENTILE = 50.0
MAX_TRACKERS = 3
P_MIN_DURATION_SEC = 0.3
P_MAX_DURATION_SEC = 15.0
NOISE_UPDATE_INTERVAL_SEC = 5.0
PROC_DIAG_INTERVAL_SEC = 120.0
DC_REMOVE_TAU_SEC = 10.0              # v11.0.0: было «магическое 10.0».
DIAG_RATIO_INTERESTING = 2.0          # v11.0.0: было «магическое 2.0».


# ============================================================================
# ==================== ВЫЧИСЛЯЕМЫЕ КОНСТАНТЫ ================================
# ============================================================================

# --- Физические константы ---
VP_VS_FACTOR = (VP * VS) / (VP - VS)

# --- Сэмплы: общие ---
SAMPLES_PER_SCREEN = int(TIME_SCALE * SAMPLE_RATE)
ADC_WARMUP_N = int(ADC_WARMUP_SEC * SAMPLE_RATE)
EVENT_DEBOUNCE_N = int(EVENT_DEBOUNCE_MS / 1000.0 * SAMPLE_RATE)
ONSET_GUARD_SEC = EVENT_DEBOUNCE_MS / 1000.0
ONSET_GUARD_N = int(ONSET_GUARD_SEC * SAMPLE_RATE)

# --- Буферы processor.py ---
BUFFER_SIZE = int(max(80.0, SNAPSHOT_PRE_P_SEC + SNAPSHOT_POST_P_SEC + 20.0)
                  * SAMPLE_RATE)
ENVELOPE_WIN_N = max(1, int(ENVELOPE_WIN_SEC * SAMPLE_RATE))
NOISE_ESTIMATE_N = int(NOISE_ESTIMATE_SEC * SAMPLE_RATE)
DERIV_HISTORY_N = max(100, NOISE_ESTIMATE_N)
ENVELOPE_DELAY_N = max(1, int(0.1 * SAMPLE_RATE))
HEAVY_PROCESS_INTERVAL_N = int(HEAVY_PROCESS_INTERVAL_SEC * SAMPLE_RATE)
NOISE_UPDATE_INTERVAL_N = int(NOISE_UPDATE_INTERVAL_SEC * SAMPLE_RATE)

# --- STA/LTA (P) ---
P_STA_N = max(1, int(P_STA_SEC * SAMPLE_RATE))
P_LTA_N = max(1, int(P_LTA_SEC * SAMPLE_RATE))

# --- STA/LTA (S) ---
S_PICKER_STA_N = max(1, int(S_PICKER_STA_SEC * SAMPLE_RATE))
S_PICKER_LTA_N = max(1, int(S_PICKER_LTA_SEC * SAMPLE_RATE))

# --- H-визуализация ---
H_DISPLAY_STA_N = max(1, int(H_DISPLAY_STA_SEC * SAMPLE_RATE))
H_DISPLAY_LTA_N = max(1, int(H_DISPLAY_LTA_SEC * SAMPLE_RATE))
H_ENV_ALPHA_N = max(1, int(H_ENV_WIN_SEC * SAMPLE_RATE))
H_MIN_SUBWIN_N = max(4, int(H_MIN_SUBWIN_SEC * SAMPLE_RATE))

# --- Snapshot ---
SNAPSHOT_PRE_P_N = int(SNAPSHOT_PRE_P_SEC * SAMPLE_RATE)
SNAPSHOT_POST_P_N = int(SNAPSHOT_POST_P_SEC * SAMPLE_RATE)
SNAPSHOT_MIN_POST_N = int(SNAPSHOT_MIN_POST_SEC * SAMPLE_RATE)

# --- Длительности фаз ---
P_MIN_DURATION_N = int(P_MIN_DURATION_SEC * SAMPLE_RATE)

# --- P_END порог ---
P_END_ABS_V = max(P_DETECT_ABS_MIN_V * 0.5, 0.005)

# --- S-пикер (адаптивный) ---
S_NOISE_PRE_WIN_N = max(1, int(S_NOISE_PRE_WIN_SEC * SAMPLE_RATE))
AIC_WIN_POST_P_N = int(AIC_WIN_POST_P_SEC * SAMPLE_RATE)
AIC_MIN_POST_P_N = max(1, int(AIC_MIN_POST_P_SEC * SAMPLE_RATE))
AMP_WIN_N = max(1, int(AMP_WIN_SEC * SAMPLE_RATE))

# --- TAIL ---
TAIL_HOLD_N = int(TAIL_HOLD_SEC * SAMPLE_RATE)
TAIL_MIN_POST_N = int(TAIL_MIN_POST_SEC * SAMPLE_RATE)
# ==================== ЛОГИРОВАНИЕ ====================
# v11.0.1: отдельный логгер для TAIL-режима.
# По умолчанию — полностью выключен. Включать только при отладке адаптивного
# завершения события: раскомментировать setLevel(DEBUG).
TAIL_LOG_ENABLED = False

# --- Прочее ---
AZIMUTH_WIN_N = max(1, int(AZIMUTH_WIN_SEC * SAMPLE_RATE))
RSAM_WIN_N = max(1, int(2.0 * SAMPLE_RATE))
INTEGRATE_TREND_REMOVE_N = int(INTEGRATE_TREND_REMOVE_SEC * SAMPLE_RATE)


# ============================================================================
# ==================== ВАЛИДАЦИЯ КОНФИГУРАЦИИ ================================
# ============================================================================

def validate_config():
    """Проверяет корректность базовых параметров. Вызывать один раз в main.py."""
    # --- Секция snapshot / окна ---
    assert SNAPSHOT_PRE_P_SEC >= P_LTA_SEC + 1.0, \
        f"SNAPSHOT_PRE_P_SEC ({SNAPSHOT_PRE_P_SEC}) must cover P_LTA_SEC ({P_LTA_SEC}) + 1s"
    assert SNAPSHOT_POST_P_SEC >= SNAPSHOT_MIN_POST_SEC, \
        f"SNAPSHOT_POST_P_SEC ({SNAPSHOT_POST_P_SEC}) must be >= SNAPSHOT_MIN_POST_SEC ({SNAPSHOT_MIN_POST_SEC})"
    assert SNAPSHOT_POST_P_SEC >= AIC_WIN_POST_P_SEC, \
        f"SNAPSHOT_POST_P_SEC ({SNAPSHOT_POST_P_SEC}) must cover AIC_WIN_POST_P_SEC ({AIC_WIN_POST_P_SEC})"

    # --- STA/LTA ---
    assert P_LTA_N > P_STA_N, \
        f"P_LTA_N ({P_LTA_N}) must be > P_STA_N ({P_STA_N})"
    assert S_PICKER_LTA_N > S_PICKER_STA_N, \
        f"S_PICKER_LTA_N ({S_PICKER_LTA_N}) must be > S_PICKER_STA_N ({S_PICKER_STA_N})"
    assert H_DISPLAY_LTA_N > H_DISPLAY_STA_N, \
        f"H_DISPLAY_LTA_N ({H_DISPLAY_LTA_N}) must be > H_DISPLAY_STA_N ({H_DISPLAY_STA_N})"
    assert P_TRIGGER_RATIO > P_DETRIGGER_RATIO, \
        f"P_TRIGGER_RATIO ({P_TRIGGER_RATIO}) must be > P_DETRIGGER_RATIO ({P_DETRIGGER_RATIO})"
    assert S_PICKER_TRIGGER > S_PICKER_DETRIGGER, \
        f"S_PICKER_TRIGGER ({S_PICKER_TRIGGER}) must be > S_PICKER_DETRIGGER ({S_PICKER_DETRIGGER})"

    # --- Буферы ---
    assert BUFFER_SIZE > SNAPSHOT_PRE_P_N + SNAPSHOT_POST_P_N, \
        f"BUFFER_SIZE ({BUFFER_SIZE}) must fit full snapshot ({SNAPSHOT_PRE_P_N + SNAPSHOT_POST_P_N})"
    assert FFT_NPERSEG <= SNAPSHOT_PRE_P_N + SNAPSHOT_POST_P_N, \
        f"FFT_NPERSEG ({FFT_NPERSEG}) must fit inside snapshot"
    assert FFT_NPERSEG >= 64, f"FFT_NPERSEG ({FFT_NPERSEG}) too small"
    assert FFT_NOVERLAP < FFT_NPERSEG, \
        f"FFT_NOVERLAP ({FFT_NOVERLAP}) must be < FFT_NPERSEG ({FFT_NPERSEG})"

    # --- SAMPLE_RATE ---
    assert SAMPLE_RATE in (100, 200, 300, 400, 600), \
        f"Unsupported sample rate: {SAMPLE_RATE}"

    # --- Карта ---
    assert MAP_MIN_SCALE > 0 and MAP_MAX_SCALE > MAP_MIN_SCALE, \
        "MAP_MIN_SCALE must be > 0 and MAP_MAX_SCALE > MAP_MIN_SCALE"
    assert MAP_MAX_DISTANCE_KM > 0, "MAP_MAX_DISTANCE_KM must be > 0"
    assert MAP_ARROW_MAX_DISTANCE_KM >= MAP_MAX_DISTANCE_KM, \
        f"MAP_ARROW_MAX_DISTANCE_KM must be >= MAP_MAX_DISTANCE_KM"
    assert MAP_STEP_KM > 0, "MAP_STEP_KM must be > 0"
    assert MAP_SCALE_KM_PER_PIXEL > 0, "MAP_SCALE_KM_PER_PIXEL must be > 0"
    assert FADE_OUT_SECONDS > 0, "FADE_OUT_SECONDS must be > 0"
    assert PULSE_FADE_SECONDS > 0, "PULSE_FADE_SECONDS must be > 0"
    assert MAX_EVENTS_ON_MAP >= 1, "MAX_EVENTS_ON_MAP must be >= 1"
    assert DEAD_ZONE_KM >= 0, "DEAD_ZONE_KM must be >= 0"

    # --- Физика ---
    assert VP > VS > 0, f"VP ({VP}) must be > VS ({VS}) > 0"
    assert DEPTH_FACTOR >= 0, "DEPTH_FACTOR must be >= 0"

    # --- Фильтры ---
    assert FILTER_FREQMIN > 0 and FILTER_FREQMIN < FILTER_FREQMAX, \
        f"FILTER_FREQMIN ({FILTER_FREQMIN}) must be in (0, {FILTER_FREQMAX})"
    assert FILTER_FREQMAX <= SAMPLE_RATE / 2.0, \
        f"FILTER_FREQMAX ({FILTER_FREQMAX}) must be <= Nyquist"
    assert S_FILTER_FREQMIN > 0 and S_FILTER_FREQMIN < S_FILTER_FREQMAX, \
        f"S_FILTER_FREQMIN ({S_FILTER_FREQMIN}) must be in (0, {S_FILTER_FREQMAX})"
    assert S_FILTER_FREQMAX <= SAMPLE_RATE / 2.0, \
        f"S_FILTER_FREQMAX ({S_FILTER_FREQMAX}) must be <= Nyquist"

    # --- P-детекция ---
    assert P_DETECT_ABS_MIN_V > 0, "P_DETECT_ABS_MIN_V must be > 0"
    assert P_DETECT_ABS_MIN_D > 0, "P_DETECT_ABS_MIN_D must be > 0"
    assert P_DETECT_DIFF_N >= 1, "P_DETECT_DIFF_N must be >= 1"
    assert P_END_ABS_V > 0, "P_END_ABS_V must be > 0"
    assert 0 < P_END_REL_BASE <= 1.0, "P_END_REL_BASE must be in (0, 1.0]"
    assert 0 < P_DROP_FROM_PEAK_FRAC <= 1.0, \
        "P_DROP_FROM_PEAK_FRAC must be in (0, 1.0]"
    assert P_AFTER_PEAK_SEC >= 0, "P_AFTER_PEAK_SEC must be >= 0"
    assert 0 < P_PEAKED_MAX_DURATION_FRAC <= 1.0, \
        "P_PEAKED_MAX_DURATION_FRAC must be in (0, 1.0]"
    assert P_MIN_DURATION_SEC > 0, "P_MIN_DURATION_SEC must be > 0"
    assert P_MAX_DURATION_SEC > P_MIN_DURATION_SEC, \
        f"P_MAX_DURATION_SEC must be > P_MIN_DURATION_SEC"
    assert STA_LTA_RESET_TIMEOUT_SEC > 0, "STA_LTA_RESET_TIMEOUT_SEC must be > 0"
    assert DETECT_WARMUP_LTA_MULT >= 1.0, "DETECT_WARMUP_LTA_MULT must be >= 1.0"

    # --- S-пикер адаптивный ---
    assert S_NOISE_PRE_WIN_SEC > 0, "S_NOISE_PRE_WIN_SEC must be > 0"
    assert S_TRIGGER_FACTOR > 0, "S_TRIGGER_FACTOR must be > 0"
    assert S_MIN_THRESHOLD_MV > 0, "S_MIN_THRESHOLD_MV must be > 0"
    assert 0 < S_AMP_RATIO_MIN < 2.0, "S_AMP_RATIO_MIN must be in (0, 2.0)"
    assert AIC_WIN_POST_P_SEC > AIC_MIN_POST_P_SEC > 0, \
        f"AIC_WIN_POST_P_SEC ({AIC_WIN_POST_P_SEC}) must be > AIC_MIN_POST_P_SEC ({AIC_MIN_POST_P_SEC})"
    assert AIC_SHARPNESS_MIN >= 0, "AIC_SHARPNESS_MIN must be >= 0"
    assert 0 <= AIC_ZONE_POS_MIN < AIC_ZONE_POS_MAX <= 1.0, \
        f"AIC_ZONE_POS_MIN ({AIC_ZONE_POS_MIN}) must be in [0, AIC_ZONE_POS_MAX)"
    assert 0 <= AIC_CFT_TRIGGER_FRAC <= 1.0, "AIC_CFT_TRIGGER_FRAC must be in [0, 1]"
    assert AMP_WIN_SEC > 0, "AMP_WIN_SEC must be > 0"

    # --- TAIL ---
    assert TAIL_DECAY_FACTOR >= 1.0, \
        "TAIL_DECAY_FACTOR must be >= 1.0 (иначе порог ниже LTA)"
    assert TAIL_HOLD_SEC > 0, "TAIL_HOLD_SEC must be > 0"
    assert TAIL_MIN_POST_SEC >= 0, "TAIL_MIN_POST_SEC must be >= 0"
    assert SNAPSHOT_MIN_POST_SEC <= SNAPSHOT_POST_P_SEC, \
        f"SNAPSHOT_MIN_POST_SEC ({SNAPSHOT_MIN_POST_SEC}) must be <= SNAPSHOT_POST_P_SEC"

    # --- Классификация ---
    assert 0 <= RECTILINEARITY_MIN <= 1.0, "RECTILINEARITY_MIN must be in [0, 1]"
    assert 0 < CLASSIFY_BASE_CONFIDENCE < 1.0, "CLASSIFY_BASE_CONFIDENCE must be in (0, 1)"
    assert CLASSIFY_MAX_CONFIDENCE <= 1.0, "CLASSIFY_MAX_CONFIDENCE must be <= 1.0"
    assert EXPLOSION_DOMINANT_FREQ_MIN > 0, "EXPLOSION_DOMINANT_FREQ_MIN must be > 0"
    assert EXPLOSION_SPECTRAL_THRESHOLD > 0, "EXPLOSION_SPECTRAL_THRESHOLD must be > 0"

    # --- Датчики ---
    for name, val in [("GEOPHONE_SENSITIVITY_N", GEOPHONE_SENSITIVITY_N),
                      ("GEOPHONE_SENSITIVITY_E", GEOPHONE_SENSITIVITY_E),
                      ("GEOPHONE_SENSITIVITY_Z", GEOPHONE_SENSITIVITY_Z),
                      ("GAIN_CORRECTION_N", GAIN_CORRECTION_N),
                      ("GAIN_CORRECTION_E", GAIN_CORRECTION_E),
                      ("GAIN_CORRECTION_Z", GAIN_CORRECTION_Z)]:
        assert val > 0, f"{name} ({val}) must be > 0"

    # --- Остальное ---
    assert ENVELOPE_WIN_SEC > 0, "ENVELOPE_WIN_SEC must be > 0"
    assert NOISE_ESTIMATE_SEC >= 5.0, "NOISE_ESTIMATE_SEC must be >= 5.0"
    assert MAX_TRACKERS >= 1, "MAX_TRACKERS must be >= 1"
    assert CAROUSEL_PANELS >= 1, "CAROUSEL_PANELS must be >= 1"
    assert MAP_GRAPH_RATIO >= 1, "MAP_GRAPH_RATIO must be >= 1"
    assert CAROUSEL_DOWNSAMPLE >= 1, "CAROUSEL_DOWNSAMPLE must be >= 1"
    assert BATCH_SIZE >= 1, "BATCH_SIZE must be >= 1"
    assert DAQ_QUEUE_MAXSIZE >= 10, "DAQ_QUEUE_MAXSIZE must be >= 10"
    assert QUEUE_MAXSIZE >= 1, "QUEUE_MAXSIZE must be >= 1"
    assert HEAVY_PROCESS_INTERVAL_SEC >= 0.05 and HEAVY_PROCESS_INTERVAL_SEC <= 2.0, \
        f"HEAVY_PROCESS_INTERVAL_SEC ({HEAVY_PROCESS_INTERVAL_SEC}) must be in [0.05, 2.0]"
    assert EVENT_DEBOUNCE_MS >= 100, "EVENT_DEBOUNCE_MS must be >= 100"
    assert GRAPH_SENSITIVITY_MV > 0, "GRAPH_SENSITIVITY_MV must be > 0"
    assert TIME_SCALE > 0, "TIME_SCALE must be > 0"
    assert ADC_WARMUP_SEC >= 0, "ADC_WARMUP_SEC must be >= 0"
    assert AZIMUTH_WIN_SEC > 0, "AZIMUTH_WIN_SEC must be > 0"
    assert INTEGRATE_TREND_REMOVE_SEC > 0, "INTEGRATE_TREND_REMOVE_SEC must be > 0"
    assert WATER_ALARM_THRESHOLD_V >= 0, "WATER_ALARM_THRESHOLD_V must be >= 0"
    assert DC_REMOVE_TAU_SEC > 0, "DC_REMOVE_TAU_SEC must be > 0"
    assert DIAG_RATIO_INTERESTING > 0, "DIAG_RATIO_INTERESTING must be > 0"
    assert H_ENV_WIN_SEC > 0, "H_ENV_WIN_SEC must be > 0"
    assert H_MIN_SUBWIN_SEC > 0, "H_MIN_SUBWIN_SEC must be > 0"
    assert CAROUSEL_DRAW_MODE in ('FULL', 'CURVE'), \
    f"CAROUSEL_DRAW_MODE must be 'FULL' or 'CURVE', got {CAROUSEL_DRAW_MODE}"
    
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
    logger = logging.getLogger('seismic')
    if logger.handlers:
        return logger

    # 1. Очистка логов
    for pattern in ('ddd.log', 'ddd.log.*', 'error.log', 'error.log.*'):
        for f in glob.glob(pattern):
            try:
                os.remove(f)
            except Exception:
                pass

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 2. DDD-хэндлер
    ddd_handler = RotatingFileHandler(
        'ddd.log', mode='w', maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
    )
    ddd_handler.setLevel(logging.DEBUG)
    ddd_handler.addFilter(MaxLevelFilter(logging.INFO))
    ddd_handler.setFormatter(formatter)
    logger.addHandler(ddd_handler)

    # 3. ERR-хэндлер
    err_handler = RotatingFileHandler(
        'error.log', mode='w', maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
    )
    err_handler.setLevel(logging.WARNING)
    err_handler.setFormatter(formatter)
    logger.addHandler(err_handler)

    # 4. FLUSH-логгер (legacy)
    flush_logger = logging.getLogger('seismic.flush')
    flush_logger.setLevel(logging.CRITICAL + 1)
    if not flush_logger.handlers:
        flush_logger.addHandler(ddd_handler)
        flush_logger.propagate = False

    # 5. TAIL-логгер (v11.0.1) — ПОСЛЕ ddd_handler
    tail_logger = logging.getLogger('seismic.tail')
    tail_logger.setLevel(logging.CRITICAL + 1)   # выключено; DEBUG при отладке
    if not tail_logger.handlers:
        tail_logger.addHandler(ddd_handler)
        tail_logger.propagate = False

    return logger 