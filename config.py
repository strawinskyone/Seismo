# Требуется: pip install obspy scipy numpy PyQt5 pyqtgraph
#
# Полный конфигурационный файл сейсмостанции.
# ВСЕ значения задаются здесь. Нигде в коде нет "магических чисел".
# РАЗДЕЛЫ СГРУППИРОВАНЫ ПО СМЫСЛУ.

import logging
from logging.handlers import RotatingFileHandler

# ==================== ОБЩИЕ / СИСТЕМА ====================
VERSION = "v9.5.3"
SAMPLE_RATE = 400                     # Гц. Допустимые: 100, 200, 300, 400, 600.
                                      # 400 Гц — оптимум для локальной сейсмики
                                      # (взрывы, удары, близкие землетрясения).
                                      # Выше 600 Гц — перегрузка CPU на Raspberry Pi.

# === CPU AFFINITY (Raspberry Pi 4: ядра 0-3) ===
# Привязка потоков к ядрам для изоляции от планировщика Linux.
# Требует root (sudo). Пустое множество = авто.
# Рекомендация: DAQ на одном ядре с изоляцией (isolcpus в cmdline.txt),
# GUI на отдельном, heavy worker на третьем.
DAQ_CPU_CORES = {3}                   # ядро для сбора данных (ADC + pigpio).
                                      # Ядро 3 часто свободно от системных задач.
GUI_CPU_CORES = {1}                   # ядро для GUI (PyQt5 + pyqtgraph).
                                      # PyQt5 event loop не любит миграции между ядрами.
HEAVY_PROCESS_CPU_CORES = {2}         # ядро для STA/LTA, Welch, FFT в отдельном процессе.
                                      # Обработка тяжёлая, но редкая — одного ядра достаточно.

# === ПРОЦЕССОР ===
HEAVY_PROCESS_INTERVAL_SEC = 0.35     # интервал тяжёлой обработки, сек.
                                      # STA/LTA, refinement P-времени, flush трекеров.
                                      # Меньше 0.1 с — перегрузка CPU. Больше 0.5 с —
                                      # задержка между P_END и отправкой snapshot.

# === ADC (AD7606B) ===
# ±2.5V на 16-bit signed = 32768 LSB (two's complement).
# 1 LSB = 2.5 / 32768 = 76.2939 мкВ.
# Этот масштаб используется для перевода raw LSB → вольты везде в коде.
ADC_SCALE_V = 2.5 / 32768.0

# ==================== СТАРТ / ПРОГРЕВ ====================
ADC_WARMUP_SEC = 3.0                   # Игнорировать данные первые N секунд после старта АЦП.
                                       # Убирает ложные onset'ы на переходном процессе.

# === ДАТЧИК ВОДЫ ===
WATER_ALARM_THRESHOLD_V = 1.5         # V. Выше этого напряжения = сработка "потоп".
                                      # Датчик подключён на CH4 (4-й канал АЦП).
                                      # При сухом состоянии обычно 0.0–0.3 В.

# ==================== SNAPSHOT / QUEUE ====================
# Окно snapshot вокруг P-волны для передачи в heavy worker.
# Pre-P должно покрывать LTA-окно (4 с) + запас.
# Post-P должно покрывать MAX_P_S_TIME (10 с) + S-STA окно + запас.
SNAPSHOT_PRE_P_SEC = 5.0              # секунд до P в snapshot.
                                      # Должно быть >= P_LTA_SEC + 1 с.
SNAPSHOT_POST_P_SEC = 15.0            # секунд после P в snapshot.
                                      # Должно быть >= MAX_P_S_TIME_SEC + 3 с (запас после S-поиска).
QUEUE_MAXSIZE = 3                     # макс. событий в очереди multiprocessing.
                                      # Если heavy worker не успевает — старые события
                                      # отбрасываются (лучше потерять старое, чем
                                      # забить очередь и зависнуть).

# ==================== ОСЦИЛЛОГРАФ / КАРУСЕЛЬ ====================
# Время отображения на экране осциллографа.
TIME_SCALE = 90                       # секунд на ВЕСЬ экран.
                                      # При 400 SPS = 36000 отсчётов на экран.
CAROUSEL_PANELS = 4                   # число панелей карусели.
                                      # [1/4] — текущие данные (активная, с метками P/S).
                                      # [2/4]–[4/4] — история (кардиограмма, без меток).
MAP_GRAPH_RATIO = 5                   # соотношение ширины карты и карусели (stretch factor).
                                      # 5:1 — карта занимает ~83% ширины окна.

GRAPH_SENSITIVITY_MV = 40.0           # половина полного размаха Y: ±40 мВ = 80 мВ total.
                                      # Это НЕ порог детекции, а только масштаб осциллографа.
                                      # Для слабых сигналов можно уменьшить до 20–50 мВ.

# === DOWNSAMPLING КАРУСЕЛИ (оптимизация CPU) ===
# При 400 SPS × 90 с = 36000 точек на экран.
# При ширине панели ~250 px — 144 точек на пиксель! Это бессмысленная нагрузка.
# Достаточно 1–2 точки на пиксель для визуального качества.
CAROUSEL_DOWNSAMPLE = 16              # оставляем каждую 16-ю точку (2250 вместо 36000).
                                      # 1 = макс. детализация, высокая нагрузка CPU.
                                      # 4 = оптимально для Raspberry Pi (~500 точек на экран).
                                      # 8 = минимальная нагрузка, мелочь может теряться.

LINE_WIDTH = 1.0                      # толщина линии на графике, пиксели.
GRAPH_POINT_SIZE = 4                  # размер точки текущей позиции (красная точка).

GRAPH_BACKGROUND_COLOR = "#0a0a0a"    # фон панели (почти чёрный).
GRAPH_LINE_COLOR = "#00ff64"          # цвет линии H (горизонтальная) — лайм.
GRAPH_POINT_COLOR = "#ff3232"         # цвет точки текущей позиции — красный.

# ==================== КАРТА ====================
MAP_IMAGE_FILE = "map_20.jpg"         # Фоновая карта. Формат: PNG или JPG.
                                      # JPG в 3× меньше весит — рекомендуется для Pi.
                                      # Разрешение должно быть достаточным для
                                      # MAP_MAX_DISTANCE_KM / MAP_SCALE_KM_PER_PIXEL.

MAP_MAX_DISTANCE_KM = 20              # макс. дистанция отображения событий, км.
                                      # События дальше показываются стрелками на краю.
MAP_ARROW_MAX_DISTANCE_KM = 50        # Не показывать стрелки для событий > 50 км
MAP_STEP_KM = 2                       # шаг кругов расстояния (концентрические круги), км.
MAP_SCALE_KM_PER_PIXEL = 40.0 / 1484  # = диаметр карты/разрешение карты=базовый масштаб карты, км/пиксель.
                                      # Используется для auto-scale при загрузке.
MAP_MIN_SCALE = 0.2                   # мин. масштаб (вся карта видна).
MAP_MAX_SCALE = 2.0                   # макс. масштаб (детали крупным планом).

STATION_OFFSET_X = 0                  # начальное смещение станции от центра, пиксели.
STATION_OFFSET_Y = 0                  # отрицательные = вверх-влево.

FADE_OUT_SECONDS = 60                 # время исчезновения события с карты, сек.
PULSE_FADE_SECONDS = 4.0              # время затухания импульсных кругов (радиальные волны), сек.
MAX_EVENTS_ON_MAP = 10                # макс. одновременных событий на карте.
                                      # При переполнении — удаляются самые старые.
DEAD_ZONE_KM = 1.0                    # игнорировать события ближе, км.
                                      # Фильтрует шум от проходящих машин, шагов и т.д.

MAP_LABEL_FONT_SIZE = 12              # размер шрифта подписи события (магнитуда, дистанция).
MAP_ARROW_WIDTH = 3                   # толщина стрелки для out-of-bounds событий.

# ==================== ПОРОГИ ДЕТЕКЦИИ ====================
EVENT_THRESHOLD_MV = 25.0             # абсолютный порог начала обработки, мВ.
                                      # Ниже этого — событие на карту не попадает.
                                      # Адаптивный onset использует SNR относительно
                                      # noise_floor, но этот порог — жёсткий floor.
EVENT_DEBOUNCE_MS = 1000              # мин. интервал между событиями, мс.
                                      # Защита от множественных срабатываний на одну волну.

# ==================== ОРИЕНТАЦИЯ ====================
AZIMUTH_OFFSET = 0.0                  # поправка азимута, градусы.
                                      # Добавляется к вычисленному азимуту перед выводом.
                                      # Используется в heavy_worker.py.

# ==================== OBSPY ФИЛЬТР ====================
# Полосовой фильтр для P-волны (STA/LTA refinement и heavy worker).
# Частоты подобраны для локальной сейсмики (взрывы, удары, близкие толчки).
FILTER_FREQMIN = 4.0                  # нижняя граница полосы, Гц.
                                      # Ниже 10 Гц — много культурного шума (дороги, ветер).
FILTER_FREQMAX = 40.0                 # верхняя граница полосы, Гц.
                                      # Выше 45 Гц — затухание в грунте, мало энергии.

# ==================== STA/LTA (P-волна) ====================
# Параметры для уточнения P-времени в heavy processor.
# Короткое окно = мгновенная энергия, длинное = фоновый уровень.
P_STA_SEC = 0.1                       # короткое окно P-волны, сек.
                                      # 0.1 с при 400 SPS = 40 отсчётов.
P_LTA_SEC = 4.0                       # длинное окно P-волны, сек.
                                      # Должно быть >= длительности тихого фона перед событием.
P_TRIGGER_RATIO = 1.5                 # отношение STA/LTA для триггера P (чувствительность).
                                      # Меньше = чувствительнее, но больше ложных срабатываний.
P_DETRIGGER_RATIO = 1.2               # отношение для сброса P (чем ниже, тем раньше сброс).
                                      # Должно быть < P_TRIGGER_RATIO.

# ==================== STA/LTA (S-волна) ====================
# Параметры для поиска S-волны в heavy worker (на горизонтальных H = N,E).
S_STA_SEC = 0.5                       # короткое окно S-волны, сек.
                                      # S-волна длиннее P — короткое окно больше.
S_LTA_SEC = 8.0                       # длинное окно S-волны, сек.
S_TRIGGER_RATIO = 4.5                 # отношение для триггера S.
                                      # S-волна сильнее P — порог выше.
S_DETRIGGER_RATIO = 1.0               # отношение для сброса S.
                                      # 1.0 = сброс при возврате к фону (жёсткий).

# ==================== ВРЕМЕННЫЕ ОКНА P-S ====================
MIN_P_S_TIME_SEC = 1.5                # мин. P-S время, сек.
                                      # Меньше — физически невозможно (S не успевает отстать).
MAX_P_S_TIME_SEC = 10.0               # макс. P-S время, сек.
                                      # Больше — событие слишком далеко для локальной модели.

# ==================== СКОРОСТИ ВОЛН ====================
# Используются для расчёта дистанции по формуле:
# distance = p_s_delta * VP * VS / (VP - VS)
# Значения для среднего грунта (сухой песок/грунт).
VP = 5.8                              # P-волна, км/с.
VS = 3.3                              # S-волна, км/с.
                                      # VP/VS ≈ 1.72 — типично для осадочных пород.

# ==================== МОДЕЛЬ ГЛУБИНЫ ====================
# Простая эмпирическая модель:
# depth = max(0, distance * DEPTH_FACTOR - DEPTH_OFFSET)
# Для локальных взрывов/ударов глубина часто мала или 0.
DEPTH_FACTOR = 0.05                    # доля дистанции, идущая в глубину.
DEPTH_OFFSET = 0.0                     # смещение, км.

# ==================== ХАРАКТЕРИСТИКИ ДАТЧИКА ====================
GEOPHONE_SENSITIVITY = 29.2           # В/(м/с). Коэффициент преобразования геофона.
                                      # Используется для перевода амплитуды (В) → скорость грунта (м/с).
                                      # Затем скорость → магнитуду ML.

# ==================== FFT / WELCH ====================
# Параметры спектрального анализа в heavy worker.
# Welch — усреднение по периодограммам для снижения дисперсии.
FFT_NPERSEG = 512                     # длина окна Welch, отсчётов.
                                      # При 400 SPS = 1.28 с окна, разрешение ~0.78 Гц.
                                      # Достаточно для классификации взрыв/толчок.
FFT_NOVERLAP = 256                    # перекрытие окон, отсчётов.
                                      # 50% — стандартный компромисс.
FFT_WINDOW = 'hann'                   # тип оконной функции.

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
#   Универсальный вариант → 'hann'

EXPLOSION_SPECTRAL_THRESHOLD = 2.0    # порог ratio high/low power для взрыва.
EXPLOSION_DOMINANT_FREQ_MIN = 6.0     # мин. доминирующая частота взрыва, Гц.
                                      # Взрывы обычно имеют высокочастотный спектр.

# ==================== HEAVY WORKER S-PICKER ====================
# Параметры STA/LTA для поиска S-волны на горизонтальных компонентах.
S_PICKER_STA_SEC = 0.3                # STA для S-пикера, сек.
S_PICKER_LTA_SEC = 3.0                # LTA для S-пикера, сек.
S_PICKER_TRIGGER = 4.0                # trigger ratio для S.
                                      # Должен быть > detrigger (как у P).
                                      # 4.0 — надёжный захват S на горизонталях.
S_PICKER_DETRIGGER = 1.0              # detrigger ratio для S.
                                      # 1.0 = сброс при возврате к фону.

# ==================== АРХИВ (РЕЗЕРВ — не используется в v9.3.2) ====================
ARCHIVE_FOLDER = "archive"

# ==================== ТАЙМИНГИ ИНТЕРФЕЙСА ====================
MAP_UPDATE_MS = 500                   # период обновления карты, мс.
RESULT_TIMER_MS = 200                 # период проверки результатов heavy worker, мс.
JOIN_TIMEOUT_SEC = 3.0                # таймаут ожидания heavy worker при выходе, с.
REFRESH_TIMER_MS = 300                # период перерисовки осциллографа, мс.
BLINK_TIMER_MS = 500                  # период мигания датчика воды, мс.

# ==================== DAQ ПАРАМЕТРЫ ====================
DAQ_QUEUE_MAXSIZE = 200               # макс. размер очереди DAQ→GUI (батчей).
                                      # 50 батчей × 20 сэмплов ≈ 2.5 сек буфера при 200 SPS.

BATCH_SIZE = 40                       # размер батча для передачи данных GUI.
                                      # Меньше — меньше задержка, больше накладных расходов.
                                      # Больше — эффективнее, но GUI обновляется реже.

# ==================== ADAPTIVE ENVELOPE ====================
# Все пороги вычисляются адаптивно по истории огибающей.
# Ниже — только стартовые коэффициенты, не абсолютные пороги.

ENVELOPE_WIN_SEC = 0.15               # окно peak-hold огибающей, сек.
                                      # 0.15 с при 400 SPS = 60 отсчётов.
                                      # Меньше — быстрее реакция, но шумнее.
                                      # Больше — плавнее, но запаздывает.
NOISE_ESTIMATE_SEC = 10.0             # длина истории для оценки шума, сек.
                                      # 10 с = 4000 точек при 400 SPS.
                                      # Должно быть >> длительности типичного события.
NOISE_PERCENTILE = 50.0               # перцентиль для noise_floor (медиана = 50%).
                                      # 50% — робастная оценка фона.
                                      # Меньше — чувствительнее к тихим событиям.
SNR_THRESHOLD = 2.0                   # env > noise_floor * SNR для onset.
                                      # Меньше — чувствительнее, но больше ложных.
ONSET_DERIVATIVE_FACTOR = 5.0         # множитель MAD для порога производной.
                                      # Производная огибающей должна превысить
                                      # median + factor * MAD.
ONSET_GUARD_SEC = 0.5                 # guard zone между onset'ами, сек.
                                      # Защита от дублирования трекеров на одном всплеске.
MAX_TRACKERS = 3                      # макс. одновременных трекеров.
                                      # При переполнении — удаляется самый старый P_ENDED.

P_MIN_DURATION_SEC = 0.3              # мин. длительность P перед переходом в PEAKED.
                                      # Защита от спайков: короткий всплеск не считается P.
P_MAX_DURATION_SEC = 5.0              # жёсткий таймаут P, сек.
                                      # Если P не закончилась сама — принудительный P_END.
P_END_ABS_FACTOR = 1.5                # abs_thr = noise_floor * factor.
                                      # Абсолютный порог окончания P.
P_END_REL_BASE = 0.30                 # базовый относительный порог (30% пика).
                                      # P заканчивается, когда амплитуда упала до 30% пика.
P_END_REL_SNR_FACTOR = 0.20           # коррекция rel_thr по SNR.
                                      # Для сильных сигналов (высокий SNR) порог снижается,
                                      # чтобы не затягивать P.

NOISE_UPDATE_INTERVAL_SEC = 5.0       # интервал обновления статистики шума, сек.
S_SEARCH_POST_P_END_SEC = 0.5         # запас после p_end_idx перед поиском S, сек.
                                      # S не ищется сразу после P — ждём затухания P-звена.


# ============================================================================
# ==================== ВЫЧИСЛЯЕМЫЕ КОНСТАНТЫ (НЕ РЕДАКТИРОВАТЬ) ==============
# ============================================================================
# Все значения ниже вычисляются автоматически из базовых параметров выше.
# При изменении SAMPLE_RATE или временных окон они пересчитаются сами.
# Редактировать только базовые параметры (секунды, Гц, мВ, км)!

# --- Экран / визуализация ---
SAMPLES_PER_SCREEN = int(TIME_SCALE * SAMPLE_RATE)

# --- Буферы processor.py ---
# Основной кольцевой буфер (должен вместить snapshot + запас)
BUFFER_SIZE = int(max(80.0, SNAPSHOT_PRE_P_SEC + SNAPSHOT_POST_P_SEC + 20.0) * SAMPLE_RATE)

# Окно огибающей (peak-hold)
ENVELOPE_WIN_N = max(1, int(ENVELOPE_WIN_SEC * SAMPLE_RATE))

# История шума
NOISE_ESTIMATE_N = int(NOISE_ESTIMATE_SEC * SAMPLE_RATE)

# История производной (для MAD)
DERIV_HISTORY_N = max(100, NOISE_ESTIMATE_N)

# Задержка огибающей для d_env
ENVELOPE_DELAY_N = max(1, int(0.1 * SAMPLE_RATE))

# --- Интервалы обработки ---
HEAVY_PROCESS_INTERVAL_N = int(HEAVY_PROCESS_INTERVAL_SEC * SAMPLE_RATE)
NOISE_UPDATE_INTERVAL_N = int(NOISE_UPDATE_INTERVAL_SEC * SAMPLE_RATE)

# --- STA/LTA в сэмплах (P-уточнение в processor) ---
P_STA_N = max(1, int(P_STA_SEC * SAMPLE_RATE))
P_LTA_N = max(1, int(P_LTA_SEC * SAMPLE_RATE))

# --- STA/LTA в сэмплах (S-пикер в heavy_worker) ---
S_PICKER_STA_N = max(1, int(S_PICKER_STA_SEC * SAMPLE_RATE))
S_PICKER_LTA_N = max(1, int(S_PICKER_LTA_SEC * SAMPLE_RATE))

# --- Snapshot границы (для get_snapshot) ---
SNAPSHOT_PRE_P_N = int(SNAPSHOT_PRE_P_SEC * SAMPLE_RATE)
SNAPSHOT_POST_P_N = int(SNAPSHOT_POST_P_SEC * SAMPLE_RATE)

# --- P-S временные окна в сэмплах ---
MIN_P_S_TIME_N = int(MIN_P_S_TIME_SEC * SAMPLE_RATE)
MAX_P_S_TIME_N = int(MAX_P_S_TIME_SEC * SAMPLE_RATE)
S_SEARCH_POST_P_END_N = int(S_SEARCH_POST_P_END_SEC * SAMPLE_RATE)

# --- Длительности фаз ---
P_MIN_DURATION_N = int(P_MIN_DURATION_SEC * SAMPLE_RATE)
P_MAX_DURATION_N = int(P_MAX_DURATION_SEC * SAMPLE_RATE)
ONSET_GUARD_N = int(ONSET_GUARD_SEC * SAMPLE_RATE)

# --- Прочее ---
ADC_WARMUP_N = int(ADC_WARMUP_SEC * SAMPLE_RATE)
EVENT_DEBOUNCE_N = int(EVENT_DEBOUNCE_MS / 1000.0 * SAMPLE_RATE)

# --- heavy_worker специфичные ---
AZIMUTH_WIN_N = max(1, int(1.0 * SAMPLE_RATE))
RSAM_WIN_N = max(1, int(2.0 * SAMPLE_RATE))
MAX_P_S_SEARCH_N = int((MAX_P_S_TIME_SEC + 2.0) * SAMPLE_RATE)


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
    assert GRAPH_SENSITIVITY_MV > 0, "GRAPH_SENSITIVITY_MV must be > 0"
    assert TIME_SCALE > 0, "TIME_SCALE must be > 0"
    assert ADC_WARMUP_SEC >= 0, "ADC_WARMUP_SEC must be >= 0"
    assert P_MIN_DURATION_SEC > 0, "P_MIN_DURATION_SEC must be > 0"
    assert P_MAX_DURATION_SEC > P_MIN_DURATION_SEC, \
        f"P_MAX_DURATION_SEC ({P_MAX_DURATION_SEC}) must be > P_MIN_DURATION_SEC ({P_MIN_DURATION_SEC})"
    assert ONSET_GUARD_SEC >= 0, "ONSET_GUARD_SEC must be >= 0"
    assert SNR_THRESHOLD > 0, "SNR_THRESHOLD must be > 0"
    assert P_END_ABS_FACTOR > 0, "P_END_ABS_FACTOR must be > 0"
    assert 0 < P_END_REL_BASE <= 1.0, "P_END_REL_BASE must be in (0, 1.0]"
    assert P_END_REL_SNR_FACTOR >= 0, "P_END_REL_SNR_FACTOR must be >= 0"
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


class MaxLevelFilter(logging.Filter):
    """
    Фильтр уровня: пропускает записи с levelno <= max_level.
    Используется для разделения потоков:
    ddd_handler (DEBUG+INFO) vs err_handler (WARNING+).
    Без этого фильтра WARNING дублировались в оба файла.
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

    В терминал ничего не выводится — всё идёт в файлы.
    """
    logger = logging.getLogger('seismic')
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # ddd.log — всё от DEBUG до INFO (WARNING и выше — НЕТ)
    ddd_handler = RotatingFileHandler(
        'ddd.log', maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
    )
    ddd_handler.setLevel(logging.DEBUG)
    ddd_handler.addFilter(MaxLevelFilter(logging.INFO))
    ddd_handler.setFormatter(formatter)
    logger.addHandler(ddd_handler)

    # error.log — только WARNING и выше
    err_handler = RotatingFileHandler(
        'error.log', maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
    )
    err_handler.setLevel(logging.WARNING)
    err_handler.setFormatter(formatter)
    logger.addHandler(err_handler)

    return logger
