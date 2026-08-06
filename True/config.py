# config.py v9.2
# Требуется: pip install obspy scipy numpy PyQt5 pyqtgraph
#
# Полный конфигурационный файл сейсмостанции.
# ВСЕ значения задаются здесь. Нигде в коде нет "магических чисел".
# РАЗДЕЛЫ СГРУППИРОВАНЫ ПО СМЫСЛУ.

import logging
from logging.handlers import RotatingFileHandler

# ==================== ОБЩИЕ / СИСТЕМА ====================
VERSION = "v9.2"
SAMPLE_RATE = 400                     # Гц. Допустимые: 100, 200, 300, 400, 600

# === CPU AFFINITY (Raspberry Pi 4: ядра 0-3) ===
# Привязка потоков к ядрам для изоляции от планировщика Linux.
# Требует root (sudo). Пустое множество = авто.
DAQ_CPU_CORES = {0, 3}                # ядро для сбора данных (ADC + pigpio)
GUI_CPU_CORES = {1}                   # ядро для GUI (PyQt5)
HEAVY_PROCESS_CPU_CORES = {2}         # ядра для STA/LTA, Welch, FFT

# === ПРОЦЕССОР ===
HEAVY_PROCESS_INTERVAL_SEC = 0.2      # интервал тяжёлой обработки, сек

# === ADC (AD7606B) ===
# ±2.5V на 16-bit signed = 32768 LSB
# 1 LSB = 2.5 / 32768 = 76.3 мкВ
ADC_SCALE_V = 2.5 / 32768.0

# === ДАТЧИК ВОДЫ ===
WATER_ALARM_THRESHOLD_V = 1.5         # V. Выше = потоп
WATER_CHECK_INTERVAL_SEC = 30         # сек

# ==================== SNAPSHOT / QUEUE ====================
# Окно snapshot вокруг P-волны для передачи в heavy worker.
# Pre-P должно покрывать LTA-окно + запас.
# Post-P должно покрывать MAX_P_S_TIME + S-STA окно + запас.
SNAPSHOT_PRE_P_SEC = 5.0              # секунд ДО P в snapshot
SNAPSHOT_POST_P_SEC = 25.0            # секунд ПОСЛЕ P в snapshot
QUEUE_MAXSIZE = 3                     # макс. событий в очереди

# ==================== ОСЦИЛЛОГРАФ / КАРУСЕЛЬ ====================
# Время отображения на экране осциллографа
TIME_SCALE = 50                       # секунд на ВЕСЬ экран
DIVISIONS_X = 10                      # число делений по горизонтали (только для сетки)

CAROUSEL_PANELS = 4                   # число панелей карусели
MAP_GRAPH_RATIO = 5                   # соотношение ширины карты и карусели (stretch factor)

GRAPH_SENSITIVITY_MV = 100.0          # полный размах шкалы ±200 мВ

# === DOWNSAMPLING КАРУСЕЛИ (оптимизация CPU) ===
# При 4000 отсчётов на экране и ширине ~250px — 16 точек на пиксель!
# Достаточно 1-2 точек на пиксель для визуального качества.
CAROUSEL_DOWNSAMPLE = 8               # оставляем каждую 8-ю точку (500 вместо 4000)
                                      # 1 = макс. детализация, высокая нагрузка CPU
                                      # 4 = оптимально для Raspberry Pi (250 точек на экран)
                                      # 8 = минимальная нагрузка, но может теряться мелочь

LINE_WIDTH = 1.0                      # толщина линии, пиксели
GRAPH_POINT_SIZE = 4                  # размер точки текущей позиции

GRAPH_BACKGROUND_COLOR = "#0a0a0a"    # фон панели
GRAPH_LINE_COLOR = "#00ff64"          # цвет линии (лайм)
GRAPH_POINT_COLOR = "#ff3232"         # цвет точки текущей позиции
GRAPH_TEXT_COLOR = "#c8c8c8"          # цвет текста

GRAPH_UPDATE_RATE = 0.10              # мин. интервал перерисовки, сек
GRAPH_THROTTLE_RATE = 0.05            # троттлинг, сек
GRAPH_Y_RANGE_FACTOR = 1.0            # множитель диапазона Y
GRAPH_MIN_Y_RANGE = 0.001             # мин. диапазон Y, В
GRAPH_MAX_Y_RANGE = 0.5               # макс. диапазон Y, В (должен быть > GRAPH_SENSITIVITY_MV/1000)
GRAPH_CLIP_FACTOR = 0.45              # коэффициент клиппирования
GRAPH_BACKEND = "pyqtgraph"

# ==================== КАРТА ====================
MAP_IMAGE_FILE = "map_100.jpg"        # формат: PNG или JPG (JPG в 3x меньше)
MAP_IMAGE_MODE = "center"             # режим: "center", "fit", "stretch"

MAP_MAX_DISTANCE_KM = 100             # макс. дистанция отображения, км
MAP_STEP_KM = 10                      # шаг кругов расстояния, км
MAP_SCALE_KM_PER_PIXEL = 0.1          # базовый масштаб карты, км/пиксель
MAP_MIN_SCALE = 0.03                  # мин. масштаб (вся карта видна)
MAP_MAX_SCALE = 2.0                   # макс. масштаб (детали)

STATION_OFFSET_X = -240               # начальное смещение станции, пиксели
STATION_OFFSET_Y = -340

FADE_OUT_SECONDS = 60                 # время исчезновения события, сек
PULSE_FADE_SECONDS = 4.0              # время затухания импульса, сек
MAX_EVENTS_ON_MAP = 30                # макс. событий на карте
DEAD_ZONE_KM = 1.5                    # игнорировать события ближе, км

MAP_LABEL_FONT_SIZE = 12              # размер шрифта подписи события
MAP_LABEL_INFO_FONT_SIZE = 10         # размер инфо-шрифта
MAP_ARROW_WIDTH = 3                   # толщина стрелки

# ==================== ПОРОГИ ДЕТЕКЦИИ ====================
EVENT_THRESHOLD_MV = 9.0               # порог начала обработки, мВ
EVENT_DEBOUNCE_MS = 1000               # мин. интервал между событиями, мс

# ==================== ОРИЕНТАЦИЯ ====================
X_IS_NORTH = True                     # True: X канал = Север
Y_IS_EAST = True                      # True: Y канал = Восток
INVERT_X = False                      # инвертировать X
INVERT_Y = False                      # инвертировать Y
AZIMUTH_OFFSET = 0.0                  # поправка азимута, градусы
AZIMUTH_GROUP_TOLERANCE = 15.0        # допуск группировки событий, °

# ==================== OBSPY ФИЛЬТР ====================
FILTER_FREQMIN = 10.0                 # Гц
FILTER_FREQMAX = 45.0                 # Гц

# ==================== STA/LTA ====================
P_STA_SEC = 0.1                       # короткое окно P-волны, сек
P_LTA_SEC = 4.0                       # длинное окно P-волны, сек
P_TRIGGER_RATIO = 1.5                 # отношение для триггера P (чувствительность)
P_DETRIGGER_RATIO = 1.2               # отношение для сброса P (чем ниже, тем раньше сброс)

S_STA_SEC = 0.5                       # короткое окно S-волны, сек
S_LTA_SEC = 8.0                       # длинное окно S-волны, сек
S_TRIGGER_RATIO = 4.5                 # отношение для триггера S
S_DETRIGGER_RATIO = 1.0               # отношение для сброса S

MIN_P_S_TIME_SEC = 2.0                # мин. P-S время, сек
MAX_P_S_TIME_SEC = 10.0               # макс. P-S время, сек

# ==================== СКОРОСТИ ВОЛН ====================
VP = 5.5                              # P-волна, км/с
VS = 3.2                              # S-волна, км/с

# ==================== МОДЕЛЬ ГЛУБИНЫ ====================
# depth = max(0, distance * DEPTH_FACTOR - DEPTH_OFFSET)
DEPTH_FACTOR = 0.3
DEPTH_OFFSET = 4.0

# ==================== ХАРАКТЕРИСТИКИ ДАТЧИКА ====================
GEOPHONE_SENSITIVITY = 18.02          # В/(м/с). Коэффициент преобразования геофона

# ==================== FFT / WELCH ====================
FFT_NPERSEG = 256                     # длина окна Welch
FFT_NOVERLAP = 128                    # перекрытие
FFT_WINDOW = 'hamming'                # тип окна
# === ТИП ОКНА WELCH ===
# Доступные опции scipy.signal.welch:
#   'hann'     — Ханна (по умолчанию). Хорошее разрешение по частоте,
#                боковые лепестки -31 дБ. Оптимально для общего анализа.
#   'hamming'  — Хэмминга. Боковые лепестки -42 дБ (лучше подавление шума),
#                но хуже разрешение. Хорошо для сейсмики с сильным шумом.
#   'blackman' — Блэкмана. Боковые лепестки -58 дБ (лучшее подавление),
#                но самое широкое главное лобовое плечо. Для чистых сигналов.
#   'bartlett' — Бартлетта. Треугольное окно. Проще, хуже подавление.
#   'flattop'  — Flat Top. Идеально для точной амплитудной калибровки,
#                но плохое разрешение по частоте.
#   'tukey'    — Тьюки (cosine-tapered). Гибрид: контролируемое затухание краёв.
#   'parzen'   — Парзена. Хорошее разрешение, низкие боковые лепестки.
#
# РЕКОМЕНДАЦИИ для вашей установки:
#   Шумная среда + нужно различать взрывы/землетрясения → 'hamming'
#   Чистый сигнал + нужно точное разрешение частоты → 'blackman'
#   Универсальный вариант (как сейчас) → 'hann'
FREQ_BAND_LOW = [0.5, 2.0]            # низкая частота, Гц
FREQ_BAND_MID = [2.0, 8.0]            # средняя частота, Гц
FREQ_BAND_HIGH = [8.0, 20.0]          # высокая частота, Гц
FREQ_BAND_VERY_HIGH = [20.0, 50.0]    # очень высокая частота, Гц

EXPLOSION_SPECTRAL_THRESHOLD = 2.0
EXPLOSION_DOMINANT_FREQ_MIN = 8.0
SPECTRUM_UPDATE_INTERVAL = 2.0        # интервал обновления спектра, сек

# ==================== АРХИВ ====================
MINISEED_STATION = "ZPGEOST"
MINISEED_NETWORK = "AM"
MINISEED_FILE_DURATION_HOURS = 3
ARCHIVE_FOLDER = "archive"

# ==================== РАСЧЁТНЫЕ ====================
SAMPLES_PER_SCREEN = int(TIME_SCALE * SAMPLE_RATE)

# ==================== ADAPTIVE ENVELOPE v9.2 ====================
# Все пороги вычисляются адаптивно по истории огибающей.
# Ниже — только стартовые коэффициенты, не абсолютные пороги.

ENVELOPE_WIN_SEC = 0.15               # окно peak-hold огибающей, сек
NOISE_ESTIMATE_SEC = 30.0             # длина истории для оценки шума, сек
NOISE_PERCENTILE = 50.0               # перцентиль для noise_floor (медиана)
SNR_THRESHOLD = 3.0                   # env > noise_floor * SNR для onset
ONSET_DERIVATIVE_FACTOR = 5.0         # множитель MAD для порога производной
ONSET_GUARD_SEC = 0.5                 # guard zone между onset'ами, сек
MAX_TRACKERS = 3                      # макс. одновременных трекеров

P_MIN_DURATION_SEC = 0.3              # мин. длительность P перед PEAKED
P_MAX_DURATION_SEC = 5.0              # жёсткий таймаут P
P_END_ABS_FACTOR = 1.5                # abs_thr = noise_floor * factor
P_END_REL_BASE = 0.30                 # базовый относительный порог (30% пика)
P_END_REL_SNR_FACTOR = 0.20           # коррекция rel_thr по SNR

S_SEARCH_POST_P_END_SEC = 0.5       # запас после p_end_idx перед поиском S

def setup_logging():
    """
    Настраивает логгер 'seismic' с ротацией.
    Все модули получают его через logging.getLogger('seismic').
    В терминал ничего не выводится.
    """
    logger = logging.getLogger('seismic')
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Общий лог — всё от DEBUG
    ddd_handler = RotatingFileHandler(
        'ddd.log', maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
    )
    ddd_handler.setLevel(logging.DEBUG)
    ddd_handler.setFormatter(formatter)
    logger.addHandler(ddd_handler)

    # Лог ошибок — WARNING и выше
    err_handler = RotatingFileHandler(
        'error.log', maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
    )
    err_handler.setLevel(logging.WARNING)
    err_handler.setFormatter(formatter)
    logger.addHandler(err_handler)

    return logger
