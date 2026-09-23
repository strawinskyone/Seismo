#!/usr/bin/env python3
"""
oscilloscope_standalone.py v7.4
Полная интеграция: AD7606B → acquisition → processor → heavy_worker.
+ Автофиксация по производной Z (адаптивный порог, K=18).
+ Запись N, E, Z (детекция — только по Z).
+ Ручная кнопка SAVE EVENT (3 с до + 6 с после).
ВСЕ print() → logger 'seismic' (ddd.log / error.log). Терминал чист.
"""
import sys
import json
import time
import os
import numpy as np
from multiprocessing import Process, Queue, Event
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QHBoxLayout, QLabel
)
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg
import logging
import config
from config import SAMPLE_RATE
from acquisition_process import DataAcquisitionProcess
from processor import SeismicProcessor
from heavy_worker import worker_loop

logger = logging.getLogger('seismic')

pg.setConfigOptions(useOpenGL=False, enableExperimental=False)
pg.setConfigOptions(imageAxisOrder='row-major')
pg.setConfigOptions(antialias=False)

CONFIG_FILE = "oscilloscope_config.json"
TIME_BASES = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

# --- Параметры записи события (ручной и авто) ---
EVENT_DIR = "events"
PRE_EVENT_SEC = 3.0
POST_EVENT_SEC = 6.0
RING_BUFFER_SEC = 15.0

# --- Параметры автодетекции по производной Z ---
AUTO_TRIGGER_ENABLED = True
TRIG_DIFF_N = 4              # окно производной, отсчётов (10 мс при 400 SPS)
TRIG_K = 18.0                # порог = K × std(d) за окно
TRIG_STD_SEC = 5.0           # окно оценки фона
TRIG_HOLD_OFF_SEC = 10.0     # блокировка после срабатывания (v7.4: было 3)
TRIG_MIN_EVENT_SEC = 0.05    # минимальная длительность события (отсечь одиночные спайки)
TRIG_MAX_RECORD_SEC = 30.0   # максимальная длина авто-записи

# --- Диагностика ---
RMS_BEFORE_SEC = 5.0         # окно RMS до триггера (для rms_before)


class StandaloneOscilloscope(QMainWindow):
    def __init__(self):
        super().__init__()

        if config.GUI_CPU_CORES:
            try:
                os.sched_setaffinity(0, config.GUI_CPU_CORES)
                logger.info(f"[OSC v7.4] GUI CPU affinity set to {config.GUI_CPU_CORES}")
            except Exception as e:
                logger.warning(f"[OSC v7.4] GUI CPU affinity failed: {e}")

        self.load_config()
        self.setWindowTitle(
            f"Seismo — Oscilloscope 4CH + Obspy | {self.config['x_seconds']:.2f} sec | {SAMPLE_RATE} SPS"
        )
        self.resize(1680, 920)
        self.sample_rate = SAMPLE_RATE

        # --- GUI: plot ---
        self.plot = pg.PlotWidget()
        self.plot.setBackground('#050505')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setMouseEnabled(x=True, y=True)
        self.plot.setLabel('left', 'Voltage, V')
        self.plot.setLabel('bottom', 'Time, s')
        self.plot.getPlotItem().getViewBox().setBorder(pg.mkPen(None))
        self.plot.setContentsMargins(0, 5, 5, 5)
        self.plot.setAntialiasing(False)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self.plot)
        self.setCentralWidget(central)

        # --- Buttons ---
        btn_layout = QHBoxLayout()
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setCheckable(True)
        self.btn_pause.clicked.connect(self.toggle_pause)

        self.lbl_timebase = QLabel("---")
        self.lbl_timebase.setStyleSheet("color: #00ff64; font-weight: bold; font-size: 12px;")

        self.btn_zoom_out = QPushButton("- Timebase")
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        self.btn_zoom_in = QPushButton("+ Timebase")
        self.btn_zoom_in.clicked.connect(self.zoom_in)

        btn_reset = QPushButton("Reset view")
        btn_reset.clicked.connect(self.reset_view)
        btn_reload = QPushButton("Reload config.json")
        btn_reload.clicked.connect(self.reload_config)

        self.btn_save_event = QPushButton("SAVE EVENT (3s+6s)")
        self.btn_save_event.setStyleSheet(
            "font-weight: bold; color: #ffffff; background-color: #aa2222; padding: 6px;"
        )
        self.btn_save_event.clicked.connect(self.start_save_event)

        btn_layout.addWidget(self.btn_pause)
        btn_layout.addWidget(self.btn_save_event)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_zoom_out)
        btn_layout.addWidget(self.lbl_timebase)
        btn_layout.addWidget(self.btn_zoom_in)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_reset)
        btn_layout.addWidget(btn_reload)
        layout.addLayout(btn_layout)

        # --- Water ---
        water_layout = QHBoxLayout()
        self.lbl_water = QLabel("WATER (CH4): ---")
        self.lbl_water.setStyleSheet(
            "font-family: monospace; font-size: 16px; font-weight: bold; color: #004466;"
        )
        water_layout.addWidget(self.lbl_water)
        water_layout.addStretch()
        layout.addLayout(water_layout)

        # --- Event (результаты heavy_worker) ---
        event_layout = QHBoxLayout()
        self.lbl_event = QLabel("EVENT: waiting...")
        self.lbl_event.setStyleSheet(
            "font-family: monospace; font-size: 14px; font-weight: bold; color: #00260f;"
        )
        event_layout.addWidget(self.lbl_event)
        event_layout.addStretch()
        layout.addLayout(event_layout)

        # --- Статус записи ---
        rec_layout = QHBoxLayout()
        self.lbl_record = QLabel("REC: idle")
        self.lbl_record.setStyleSheet(
            "font-family: monospace; font-size: 14px; font-weight: bold; color: #666666;"
        )
        rec_layout.addWidget(self.lbl_record)
        rec_layout.addStretch()
        layout.addLayout(rec_layout)

        # --- Диагностика триггера ---
        trig_layout = QHBoxLayout()
        self.lbl_trigger = QLabel("TRIG: d=--- thr=--- (K=%.1f)" % TRIG_K)
        self.lbl_trigger.setStyleSheet(
            "font-family: monospace; font-size: 13px; color: #444444;"
        )
        trig_layout.addWidget(self.lbl_trigger)
        trig_layout.addStretch()
        layout.addLayout(trig_layout)

        # --- Diag ---
        diag_layout = QHBoxLayout()
        self.lbl_diag = QLabel("DIAG: waiting...")
        self.lbl_diag.setStyleSheet(
            "font-family: monospace; font-size: 14px; color: #000000"
        )
        diag_layout.addWidget(self.lbl_diag)
        diag_layout.addStretch()
        layout.addLayout(diag_layout)

        self.paused = False
        self.timebase_idx = self._find_timebase_idx(self.config['x_seconds'])
        self._update_timebase_label()

        self.curves = []
        self.offsets = []
        self.sensitivities = []
        self.enabled_channels = []
        self.buffers = []
        self.buffer_write_idx = 0
        self.time_axis = np.array([])
        self._scratches = []

        self._init_curves_and_buffers()

        # --- Кольцевой буфер записи: N, E, Z (детекция — только Z) ---
        self.rec_size = int(RING_BUFFER_SEC * SAMPLE_RATE)
        self.rec_n = np.zeros(self.rec_size, dtype=np.float32)
        self.rec_e = np.zeros(self.rec_size, dtype=np.float32)
        self.rec_z = np.zeros(self.rec_size, dtype=np.float32)
        self.rec_t = np.zeros(self.rec_size, dtype=np.float64)
        self.rec_idx = 0
        self.rec_filled = False
        self.recording = None
        self.auto_recording = None
        os.makedirs(EVENT_DIR, exist_ok=True)

        # --- Детектор по производной Z ---
        self.trig_diff_n = TRIG_DIFF_N
        self.trig_std_n = int(TRIG_STD_SEC * SAMPLE_RATE)
        self.trig_d_ring = np.zeros(self.trig_std_n, dtype=np.float32)
        self.trig_d_ring_idx = 0
        self.trig_d_ring_filled = False
        self.trig_holdoff_until = 0.0
        self.trig_last_thr = 0.0
        self.trig_last_d = 0.0
        self.trig_events_count = 0
        self.trig_z_short = np.zeros(self.trig_diff_n, dtype=np.float32)
        self.trig_z_short_idx = 0
        # Кольцо Z для оценки rms_before (5 с)
        self.rms_before_n = int(RMS_BEFORE_SEC * SAMPLE_RATE)
        self.rms_before_ring = np.zeros(self.rms_before_n, dtype=np.float32)
        self.rms_before_idx = 0
        self.rms_before_filled = False

        # --- Multiprocessing: heavy_worker ---
        self.event_queue = Queue(maxsize=config.QUEUE_MAXSIZE)
        self.result_queue = Queue()
        self.heavy_process = Process(
            target=worker_loop,
            args=(self.event_queue, self.result_queue),
            daemon=True
        )
        self.heavy_process.start()
        logger.info("[OSC v7.4] Heavy worker process started")

        # --- Processor ---
        self.processor = SeismicProcessor(event_queue=self.event_queue)

        # --- Acquisition process ---
        self.data_queue = Queue(maxsize=config.DAQ_QUEUE_MAXSIZE)
        self.stop_event = Event()
        self.acquisition = DataAcquisitionProcess(
            data_queue=self.data_queue,
            stop_event=self.stop_event,
            sample_rate=SAMPLE_RATE,
            batch_size=20
        )
        self.acquisition.start()
        logger.info("[OSC v7.4] Acquisition process started")

        self.daq_timer = QTimer()
        self.daq_timer.timeout.connect(self._poll_daq_queue)
        self.daq_timer.start(50)

        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self._update_plot)
        self.plot_timer.start(40)

        self.water_timer = QTimer()
        self.water_timer.timeout.connect(self._update_water_display)
        self.water_timer.start(3000)

        self.result_timer = QTimer()
        self.result_timer.timeout.connect(self._check_worker_results)
        self.result_timer.start(200)

        self.rec_timer = QTimer()
        self.rec_timer.timeout.connect(self._update_record_status)
        self.rec_timer.start(200)

        # --- Diag state ---
        self.diag_total_samples = 0
        self.diag_plot_calls = 0
        self.diag_last_log = time.time()
        self.diag_batch_count = 0
        self.last_water_volts = 0.0
        self.last_water_raw = 0

        logger.info(
            f"[OSC v7.4] Started | SPS={SAMPLE_RATE} | GUI FPS=25 | BATCH=20 | "
            f"AutoTrig={'ON' if AUTO_TRIGGER_ENABLED else 'OFF'} "
            f"(K={TRIG_K}, N={TRIG_DIFF_N}, std_win={TRIG_STD_SEC}s, "
            f"holdoff={TRIG_HOLD_OFF_SEC}s) | Channels=N,E,Z"
        )

    def load_config(self):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            logger.info(f"[OSC v7.4] Config loaded: {len(self.config['channel_configs'])} channels")
        except Exception as e:
            logger.warning(f"[OSC v7.4] JSON ERROR: {e}, using defaults")
            self.config = {
                "x_seconds": 1.0,
                "channel_configs": [
                    {"name": "X (CH1)", "color": "#FF8C00", "offset": 0.0, "sensitivity_v": 1.0, "enabled": True},
                    {"name": "Y (CH2)", "color": "#228B22", "offset": 0.0, "sensitivity_v": 1.0, "enabled": True},
                    {"name": "Z (CH3)", "color": "#0000FF", "offset": 0.0, "sensitivity_v": 1.0, "enabled": True},
                    {"name": "ObsPy Vector", "color": "#FFFFFF", "offset": -0.75, "sensitivity_v": 1.0, "enabled": True}
                ],
                "line_width": 1.5
            }

    def _find_timebase_idx(self, x_seconds):
        return min(range(len(TIME_BASES)), key=lambda i: abs(TIME_BASES[i] - x_seconds))

    def _update_timebase_label(self):
        sec = TIME_BASES[self.timebase_idx]
        self.lbl_timebase.setText(f"{sec:.2f} s/div" if sec >= 1.0 else f"{int(sec * 1000)} ms/div")

    def _init_curves_and_buffers(self):
        self.plot.clear()
        self.curves = []
        self.offsets = []
        self.sensitivities = []
        self.enabled_channels = []
        self.legend = self.plot.addLegend(offset=(10, 10))

        for i, ch_config in enumerate(self.config['channel_configs']):
            if ch_config.get('enabled', True):
                curve = self.plot.plot(
                    pen=pg.mkPen(ch_config['color'], width=self.config.get('line_width', 1.5)),
                    clipToView=True
                )
                self.curves.append(curve)
                self.legend.addItem(curve, ch_config['name'])
                self.offsets.append(ch_config.get('offset', 0.0))
                self.sensitivities.append(ch_config.get('sensitivity_v', 1.0))
                self.enabled_channels.append(i)

        self.buffer_size = int(TIME_BASES[self.timebase_idx] * self.sample_rate)
        self.buffers = [
            np.zeros(self.buffer_size, dtype=np.float32)
            for _ in range(len(self.enabled_channels))
        ]
        self._scratches = [
            np.zeros(self.buffer_size, dtype=np.float32)
            for _ in range(len(self.enabled_channels))
        ]
        self.time_axis = np.arange(self.buffer_size) / self.sample_rate
        self.buffer_write_idx = 0
        self.reset_view()

    def _apply_timebase(self):
        sec = TIME_BASES[self.timebase_idx]
        self.config['x_seconds'] = sec
        self._update_timebase_label()
        self.setWindowTitle(f"Seismo — Oscilloscope 4CH + Obspy | {sec:.2f} sec | {SAMPLE_RATE} SPS")
        self._init_curves_and_buffers()

    def reset_view(self):
        sec = TIME_BASES[self.timebase_idx]
        self.plot.setXRange(0, sec, padding=0.0)
        self.plot.setYRange(-3.5, 3.5, padding=0.0)

    def reload_config(self):
        self.load_config()
        self.timebase_idx = self._find_timebase_idx(self.config['x_seconds'])
        self._update_timebase_label()
        self._init_curves_and_buffers()

    def toggle_pause(self, checked):
        self.paused = checked
        self.btn_pause.setText("Resume" if self.paused else "Pause")

    def zoom_in(self):
        if self.timebase_idx > 0:
            self.timebase_idx -= 1
            self._apply_timebase()

    def zoom_out(self):
        if self.timebase_idx < len(TIME_BASES) - 1:
            self.timebase_idx += 1
            self._apply_timebase()

    def _update_water_display(self):
        v = self.last_water_volts
        r = self.last_water_raw
        if v > config.WATER_ALARM_THRESHOLD_V:
            status = "FLOOD!"
            color = "#ff0000"
        elif v > 0.5:
            status = "WET"
            color = "#ffaa00"
        else:
            status = "DRY"
            color = "#00331a"
        self.lbl_water.setStyleSheet(
            f"font-family: monospace; font-size: 16px; font-weight: bold; color: {color};"
        )
        self.lbl_water.setText(f"WATER (CH4): {v:+.3f}V  RAW:{r:+6d}  {status}")

    def _poll_daq_queue(self):
        while not self.data_queue.empty():
            try:
                batch = self.data_queue.get_nowait()
                batch_volts, batch_raws, batch_ts, total_samples, errors = batch
                self.on_new_data_batch(batch_volts, batch_raws, batch_ts, total_samples, errors)
            except Exception:
                break

    def _get_rec_chronological(self, arr):
        if not self.rec_filled:
            return arr[:self.rec_idx].copy()
        return np.concatenate((arr[self.rec_idx:], arr[:self.rec_idx]))

    # ============================================================
    # ОБРАБОТКА НОВЫХ ДАННЫХ
    # ============================================================
    def on_new_data_batch(self, batch_volts, batch_raws, batch_ts, total_samples, errors):
        if self.paused:
            return

        self.processor.process_batch(batch_raws, batch_ts)

        self.diag_batch_count += 1
        batch_len = len(batch_volts)
        self.diag_total_samples += batch_len

        for i in range(batch_len):
            volts = batch_volts[i]
            raws = batch_raws[i]
            if len(volts) < 4:
                continue

            w = float(volts[0])   # CH1 = вода
            x = float(volts[1])   # CH2 = N (север)
            y = float(volts[2])   # CH3 = E (восток)
            z = float(volts[3])   # CH4 = Z (вертикаль)
            ts_i = float(batch_ts[i])

            self.last_water_volts = w
            self.last_water_raw = int(raws[0])

            vector_mag = float(np.sqrt(x*x + y*y + z*z))
            channel_values = {0: x, 1: y, 2: z, 3: vector_mag}

            idx = self.buffer_write_idx
            for buf_idx, ch_idx in enumerate(self.enabled_channels):
                val = channel_values.get(ch_idx, 0.0)
                self.buffers[buf_idx][idx] = val + self.offsets[buf_idx]
            self.buffer_write_idx = (idx + 1) % self.buffer_size

            # --- Кольцевой буфер записи: N, E, Z ---
            self.rec_n[self.rec_idx] = x
            self.rec_e[self.rec_idx] = y
            self.rec_z[self.rec_idx] = z
            self.rec_t[self.rec_idx] = ts_i
            self.rec_idx = (self.rec_idx + 1) % self.rec_size
            if self.rec_idx == 0:
                self.rec_filled = True

            # --- Ручная запись: копим пост-буфер ---
            if self.recording is not None and self.recording['phase'] == 'post':
                self.recording['post_n'].append(x)
                self.recording['post_e'].append(y)
                self.recording['post_z'].append(z)
                self.recording['post_t'].append(ts_i)

            # --- Кольцо Z для rms_before ---
            self.rms_before_ring[self.rms_before_idx] = z
            self.rms_before_idx = (self.rms_before_idx + 1) % self.rms_before_n
            if self.rms_before_idx == 0:
                self.rms_before_filled = True

            # --- Автодетекция по производной Z ---
            if AUTO_TRIGGER_ENABLED:
                self._auto_trigger_step(z, ts_i)

            # --- Автозапись: копим пост-буфер ---
            if self.auto_recording is not None and self.auto_recording['phase'] == 'post':
                self.auto_recording['post_n'].append(x)
                self.auto_recording['post_e'].append(y)
                self.auto_recording['post_z'].append(z)
                self.auto_recording['post_t'].append(ts_i)

    # ============================================================
    # АВТОДЕТЕКЦИЯ ПО ПРОИЗВОДНОЙ Z
    # ============================================================
    def _auto_trigger_step(self, z, ts_i):
        self.trig_z_short[self.trig_z_short_idx] = z
        self.trig_z_short_idx = (self.trig_z_short_idx + 1) % self.trig_diff_n
        d = abs(z - self.trig_z_short[self.trig_z_short_idx])

        self.trig_d_ring[self.trig_d_ring_idx] = d
        self.trig_d_ring_idx = (self.trig_d_ring_idx + 1) % self.trig_std_n
        if self.trig_d_ring_idx == 0:
            self.trig_d_ring_filled = True

        if self.trig_d_ring_filled:
            d_hist = self.trig_d_ring
        else:
            d_hist = self.trig_d_ring[:self.trig_d_ring_idx]
        if len(d_hist) < 100:
            return

        d_std = float(np.std(d_hist))
        thr = TRIG_K * max(d_std, 1e-6)
        self.trig_last_thr = thr
        self.trig_last_d = d

        if d < thr:
            return
        now = ts_i
        if now < self.trig_holdoff_until:
            return
        self._start_auto_record(ts_i, d, thr, z)
        self.trig_holdoff_until = now + TRIG_HOLD_OFF_SEC
        self.trig_events_count += 1

    def _start_auto_record(self, ts_trigger, d_peak, thr, z_peak):
        if self.auto_recording is not None:
            return
        pre_n = int(PRE_EVENT_SEC * SAMPLE_RATE)
        pre_n = min(pre_n, self.rec_size)
        buf_n = self._get_rec_chronological(self.rec_n)
        buf_e = self._get_rec_chronological(self.rec_e)
        buf_z = self._get_rec_chronological(self.rec_z)
        buf_t = self._get_rec_chronological(self.rec_t)

        # rms_before: RMS Z за окно RMS_BEFORE_SEC перед триггером
        if self.rms_before_filled:
            rms_before = float(np.sqrt(np.mean(self.rms_before_ring ** 2)))
        else:
            rms_before = float(np.sqrt(np.mean(self.rms_before_ring[:self.rms_before_idx] ** 2)))

        self.auto_recording = {
            'start_wall': time.time(),
            'trigger_t': ts_trigger,
            'd_peak': d_peak,
            'd_threshold': thr,
            'z_peak': z_peak,
            'rms_before': rms_before,
            'pre_n': buf_n[-pre_n:].copy(),
            'pre_e': buf_e[-pre_n:].copy(),
            'pre_z': buf_z[-pre_n:].copy(),
            'pre_t': buf_t[-pre_n:].copy(),
            'post_n': [], 'post_e': [], 'post_z': [], 'post_t': [],
            'phase': 'post',
            'end_wall': time.time() + POST_EVENT_SEC,
        }
        logger.info(
            f"[OSC AUTO] TRIGGER @ {ts_trigger:.3f} | d={d_peak:.5f} "
            f"thr={thr:.5f} z={z_peak:+.5f} rms_before={rms_before:.5f} | pre={pre_n} samples"
        )
        self.lbl_trigger.setStyleSheet(
            "font-family: monospace; font-size: 13px; color: #cc4400; font-weight: bold;"
        )
        self.lbl_trigger.setText(
            f"TRIG: d={d_peak:.4f} thr={thr:.4f} (K={TRIG_K:.1f}) z={z_peak:+.4f} "
            f"rms_bfr={rms_before:.4f}"
        )
        self.lbl_record.setText(f"REC (auto): 0.0 / {POST_EVENT_SEC:.1f} s")
        self.lbl_record.setStyleSheet(
            "font-family: monospace; font-size: 14px; font-weight: bold; color: #cc4400;"
        )

    def _update_record_status(self):
        now = time.time()
        # Ручная запись
        if self.recording is not None:
            rec = self.recording
            if rec['phase'] == 'post':
                elapsed = now - rec['start_wall']
                self.lbl_record.setText(
                    f"REC (manual): {elapsed:.1f} / {POST_EVENT_SEC:.1f} s"
                )
                if now >= rec['end_wall']:
                    self._finalize_save_event()
        # Автозапись
        if self.auto_recording is not None:
            rec = self.auto_recording
            elapsed = now - rec['start_wall']
            if elapsed < POST_EVENT_SEC:
                self.lbl_record.setText(
                    f"REC (auto): {elapsed:.1f} / {POST_EVENT_SEC:.1f} s"
                )
            if now >= rec['end_wall']:
                self._finalize_auto_record()
            elif elapsed >= TRIG_MAX_RECORD_SEC:
                self._finalize_auto_record(reason='max_duration')

    # ============================================================
    # РУЧНАЯ ЗАПИСЬ
    # ============================================================
    def start_save_event(self):
        if self.recording is not None:
            logger.warning("[OSC REC] Ручная запись уже идёт")
            return
        now = time.time()
        pre_n = int(PRE_EVENT_SEC * SAMPLE_RATE)
        pre_n = min(pre_n, self.rec_size)
        buf_n = self._get_rec_chronological(self.rec_n)
        buf_e = self._get_rec_chronological(self.rec_e)
        buf_z = self._get_rec_chronological(self.rec_z)
        buf_t = self._get_rec_chronological(self.rec_t)
        self.recording = {
            'start_wall': now,
            'pre_n': buf_n[-pre_n:].copy(),
            'pre_e': buf_e[-pre_n:].copy(),
            'pre_z': buf_z[-pre_n:].copy(),
            'pre_t': buf_t[-pre_n:].copy(),
            'post_n': [], 'post_e': [], 'post_z': [], 'post_t': [],
            'phase': 'post',
            'end_wall': now + POST_EVENT_SEC,
        }
        logger.info(f"[OSC REC] MANUAL START @ {now:.3f} | pre={pre_n} samples")
        self.lbl_record.setText(f"REC (manual): 0.0 / {POST_EVENT_SEC:.1f} s")
        self.lbl_record.setStyleSheet(
            "font-family: monospace; font-size: 14px; font-weight: bold; color: #ff3232;"
        )
        self.btn_save_event.setEnabled(False)

    def _finalize_save_event(self):
        rec = self.recording
        post_n = np.array(rec['post_n'], dtype=np.float32)
        post_e = np.array(rec['post_e'], dtype=np.float32)
        post_z = np.array(rec['post_z'], dtype=np.float32)
        post_t = np.array(rec['post_t'], dtype=np.float64)
        post_len = int(POST_EVENT_SEC * SAMPLE_RATE)
        if len(post_z) > post_len:
            post_n = post_n[:post_len]; post_e = post_e[:post_len]
            post_z = post_z[:post_len]; post_t = post_t[:post_len]
        n_full = np.concatenate((rec['pre_n'], post_n))
        e_full = np.concatenate((rec['pre_e'], post_e))
        z_full = np.concatenate((rec['pre_z'], post_z))
        t_full = np.concatenate((rec['pre_t'], post_t))
        ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(rec['start_wall']))
        fname = os.path.join(EVENT_DIR, f"event_{ts_str}_manual.npz")
        np.savez_compressed(
            fname,
            n=n_full, e=e_full, z=z_full, t=t_full,
            sample_rate=SAMPLE_RATE,
            pre_sec=PRE_EVENT_SEC, post_sec=POST_EVENT_SEC,
            version="v7.4-manual",
        )
        duration = len(z_full) / SAMPLE_RATE
        logger.info(f"[OSC REC] SAVED (manual) {fname} | {len(z_full)} samples ({duration:.2f}s) | N,E,Z")
        self.lbl_record.setText(f"REC: manual saved {os.path.basename(fname)} ({duration:.2f}s)")
        self.lbl_record.setStyleSheet(
            "font-family: monospace; font-size: 14px; font-weight: bold; color: #00aa44;"
        )
        self.recording = None
        self.btn_save_event.setEnabled(True)

    # ============================================================
    # АВТОЗАПИСЬ — финализация
    # ============================================================
    def _finalize_auto_record(self, reason='post_complete'):
        rec = self.auto_recording
        post_n = np.array(rec['post_n'], dtype=np.float32)
        post_e = np.array(rec['post_e'], dtype=np.float32)
        post_z = np.array(rec['post_z'], dtype=np.float32)
        post_t = np.array(rec['post_t'], dtype=np.float64)
        post_len = int(POST_EVENT_SEC * SAMPLE_RATE)
        if len(post_z) > post_len:
            post_n = post_n[:post_len]; post_e = post_e[:post_len]
            post_z = post_z[:post_len]; post_t = post_t[:post_len]
        n_full = np.concatenate((rec['pre_n'], post_n))
        e_full = np.concatenate((rec['pre_e'], post_e))
        z_full = np.concatenate((rec['pre_z'], post_z))
        t_full = np.concatenate((rec['pre_t'], post_t))
        ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(rec['start_wall']))
        fname = os.path.join(EVENT_DIR, f"event_{ts_str}_auto.npz")
        np.savez_compressed(
            fname,
            n=n_full, e=e_full, z=z_full, t=t_full,
            sample_rate=SAMPLE_RATE,
            pre_sec=PRE_EVENT_SEC, post_sec=POST_EVENT_SEC,
            trigger_t=rec['trigger_t'],
            d_peak=rec['d_peak'],
            d_threshold=rec['d_threshold'],
            z_peak=rec['z_peak'],
            rms_before=rec['rms_before'],
            reason=reason,
            version="v7.4-auto",
        )
        duration = len(z_full) / SAMPLE_RATE
        logger.info(
            f"[OSC AUTO] SAVED {fname} | {len(z_full)} samples ({duration:.2f}s) | N,E,Z | "
            f"d_peak={rec['d_peak']:.5f} thr={rec['d_threshold']:.5f} "
            f"z_peak={rec['z_peak']:+.5f} rms_before={rec['rms_before']:.5f} reason={reason}"
        )
        self.lbl_record.setText(
            f"REC: auto saved {os.path.basename(fname)} ({duration:.2f}s)"
        )
        self.lbl_record.setStyleSheet(
            "font-family: monospace; font-size: 14px; font-weight: bold; color: #00aa44;"
        )
        self.auto_recording = None

    def _update_plot(self):
        if self.paused or not self.buffers:
            return
        self.diag_plot_calls += 1
        idx = self.buffer_write_idx
        size = self.buffer_size
        SMOOTH_WIN = 3
        kernel = np.ones(SMOOTH_WIN, dtype=np.float32) / SMOOTH_WIN
        for i in range(len(self.curves)):
            if i >= len(self.buffers):
                continue
            buf = self.buffers[i]
            scratch = self._scratches[i]
            if idx == 0:
                data = buf
            else:
                scratch[:size-idx] = buf[idx:]
                scratch[size-idx:] = buf[:idx]
                data = scratch
            if len(data) >= SMOOTH_WIN:
                smooth = np.convolve(data, kernel, mode='same')
            else:
                smooth = data
            self.curves[i].setData(self.time_axis, smooth, skipFiniteCheck=True)

        self.lbl_trigger.setText(
            f"TRIG: d={self.trig_last_d:.5f} thr={self.trig_last_thr:.5f} "
            f"(K={TRIG_K:.1f}) events={self.trig_events_count}"
        )

        now = time.time()
        if now - self.diag_last_log >= 300.0:
            expected = int((now - self.diag_last_log) * SAMPLE_RATE)
            received = self.diag_total_samples
            plots = self.diag_plot_calls
            batches = self.diag_batch_count
            loss = expected - received
            loss_pct = (loss / expected * 100) if expected > 0 else 0
            logger.info(
                f"[OSC DIAG] 300s: expected={expected}, received={received}, "
                f"batches={batches}, plots={plots}, loss={loss} ({loss_pct:.1f}%) "
                f"auto_triggers={self.trig_events_count}"
            )
            self.lbl_diag.setText(
                f"DIAG: loss={loss_pct:.1f}% | batches={batches} | "
                f"plots={plots} | auto={self.trig_events_count}"
            )
            self.diag_total_samples = 0
            self.diag_plot_calls = 0
            self.diag_batch_count = 0
            self.diag_last_log = now

    def _check_worker_results(self):
        while not self.result_queue.empty():
            try:
                result = self.result_queue.get_nowait()
                if result and result.get('status') == 'event':
                    dist = result.get('distance', 0)
                    mag = result.get('ml_magnitude', 0)
                    az = result.get('azimuth', 0)
                    etype = result.get('event_type', 'unknown')
                    conf = result.get('event_confidence', 0)
                    text = (
                        f"{etype.upper()} | D={dist:.1f}km | Ml={mag:.2f} "
                        f"| Az={az:.1f}° | conf={conf:.2f}"
                    )
                    self.lbl_event.setText(text)
                    self.lbl_event.setStyleSheet(
                        "font-family: monospace; font-size: 14px; font-weight: bold; color: #ff3232;"
                    )
                    logger.info(f"[OSC] Event detected: {text}")
            except Exception:
                break

    def closeEvent(self, event):
        logger.info("[OSC v7.4] Stopping acquisition, worker, timers...")
        self.stop_event.set()
        self.acquisition.join(timeout=3.0)
        if self.acquisition.is_alive():
            self.acquisition.terminate()
            logger.warning("[OSC v7.4] Acquisition process terminated forcefully")
        self.event_queue.put('STOP')
        self.heavy_process.join(timeout=3.0)
        if self.heavy_process.is_alive():
            self.heavy_process.terminate()
            logger.warning("[OSC v7.4] Heavy worker terminated forcefully")
        self.plot_timer.stop()
        self.water_timer.stop()
        self.result_timer.stop()
        self.daq_timer.stop()
        self.rec_timer.stop()
        event.accept()
        logger.info("[OSC v7.4] UI closed cleanly.")


if __name__ == "__main__":
    config.setup_logging()
    app = QApplication(sys.argv)
    win = StandaloneOscilloscope()
    win.show()
    sys.exit(app.exec_())