#!/usr/bin/env python3
"""
oscilloscope_standalone.py v7.0
Полная интеграция: AD7606B → acquisition → processor → heavy_worker.
Скелет GUI из v6.9.3c (карусель, кнопки, water, diag).
ВСЕ print() → logger 'seismic' (ddd.log / error.log). Терминал чист.
"""
import sys
import json
import time
import os
import numpy as np
from multiprocessing import Process, Queue
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QHBoxLayout, QLabel
)
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg
import logging
import config
from config import SAMPLE_RATE
from acquisition import DataAcquisitionThread
from processor import SeismicProcessor
from heavy_worker import worker_loop

logger = logging.getLogger('seismic')

pg.setConfigOptions(useOpenGL=False, enableExperimental=False)
pg.setConfigOptions(imageAxisOrder='row-major')
pg.setConfigOptions(antialias=False)

CONFIG_FILE = "oscilloscope_config.json"
TIME_BASES = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]


class StandaloneOscilloscope(QMainWindow):
    def __init__(self):
        super().__init__()

        if config.GUI_CPU_CORES:
            try:
                os.sched_setaffinity(0, config.GUI_CPU_CORES)
                logger.info(f"[OSC v7.0] GUI CPU affinity set to {config.GUI_CPU_CORES}")
            except Exception as e:
                logger.warning(f"[OSC v7.0] GUI CPU affinity failed: {e}")

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

        btn_layout.addWidget(self.btn_pause)
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
            "font-family: monospace; font-size: 16px; font-weight: bold; color: #00ff64;"
        )
        water_layout.addWidget(self.lbl_water)
        water_layout.addStretch()
        layout.addLayout(water_layout)

        # --- Event (новое: результаты heavy_worker) ---
        event_layout = QHBoxLayout()
        self.lbl_event = QLabel("EVENT: waiting...")
        self.lbl_event.setStyleSheet(
            "font-family: monospace; font-size: 14px; font-weight: bold; color: #00aaff;"
        )
        event_layout.addWidget(self.lbl_event)
        event_layout.addStretch()
        layout.addLayout(event_layout)

        # --- Diag ---
        diag_layout = QHBoxLayout()
        self.lbl_diag = QLabel("DIAG: waiting...")
        self.lbl_diag.setStyleSheet(
            "font-family: monospace; font-size: 12px; color: #ffaa00;"
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

        # --- Multiprocessing: heavy_worker ---
        self.event_queue = Queue(maxsize=config.QUEUE_MAXSIZE)
        self.result_queue = Queue()
        self.heavy_process = Process(
            target=worker_loop,
            args=(self.event_queue, self.result_queue),
            daemon=True
        )
        self.heavy_process.start()
        logger.info("[OSC v7.0] Heavy worker process started")

        # --- Processor ---
        self.processor = SeismicProcessor(event_queue=self.event_queue)

        # --- Acquisition thread ---
        self.acquisition = DataAcquisitionThread(sample_rate=SAMPLE_RATE, batch_size=20)
        self.acquisition.data_ready.connect(self.on_new_data_batch)
        self.acquisition.start()
        logger.info("[OSC v7.0] Acquisition thread started")

        # --- Timers ---
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self._update_plot)
        self.plot_timer.start(40)

        self.water_timer = QTimer()
        self.water_timer.timeout.connect(self._update_water_display)
        self.water_timer.start(3000)

        self.result_timer = QTimer()
        self.result_timer.timeout.connect(self._check_worker_results)
        self.result_timer.start(200)

        # --- Diag state ---
        self.diag_total_samples = 0
        self.diag_plot_calls = 0
        self.diag_last_log = time.time()
        self.diag_batch_count = 0
        self.last_water_volts = 0.0
        self.last_water_raw = 0

        logger.info(f"[OSC v7.0] Started | SPS={SAMPLE_RATE} | GUI FPS=25 | BATCH=20")

    def load_config(self):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            logger.info(f"[OSC v7.0] Config loaded: {len(self.config['channel_configs'])} channels")
        except Exception as e:
            logger.warning(f"[OSC v7.0] JSON ERROR: {e}, using defaults")
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
                # v9.1.5: убран subsample — теперь сигнал плавный, без "лестниц"
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
            color = "#00ff64"
        self.lbl_water.setStyleSheet(
            f"font-family: monospace; font-size: 16px; font-weight: bold; color: {color};"
        )
        self.lbl_water.setText(f"WATER (CH4): {v:+.3f}V  RAW:{r:+6d}  {status}")

    def on_new_data_batch(self, batch_volts, batch_raws, batch_ts, total_samples, errors):
        if self.paused:
            return

        # 1. Передаём RAW в процессор (STA/LTA, snapshot'ы, очередь для heavy_worker)
        self.processor.process_batch(batch_raws, batch_ts)

        # 2. Обновляем GUI-буферы
        self.diag_batch_count += 1
        batch_len = len(batch_volts)
        self.diag_total_samples += batch_len

        for i in range(batch_len):
            volts = batch_volts[i]
            raws = batch_raws[i]
            if len(volts) < 4:
                continue

            x = float(volts[0])
            y = float(volts[1])
            z = float(volts[2])
            w = float(volts[3])

            self.last_water_volts = w
            self.last_water_raw = int(raws[3])

            vector_mag = float(np.sqrt(x*x + y*y + z*z))
            channel_values = {0: x, 1: y, 2: z, 3: vector_mag}

            idx = self.buffer_write_idx
            for buf_idx, ch_idx in enumerate(self.enabled_channels):
                val = channel_values.get(ch_idx, 0.0)
                self.buffers[buf_idx][idx] = val + self.offsets[buf_idx]

            self.buffer_write_idx = (idx + 1) % self.buffer_size

    def _update_plot(self):
        if self.paused or not self.buffers:
            return

        self.diag_plot_calls += 1
        idx = self.buffer_write_idx
        size = self.buffer_size
        # Коэффициент сглаживания для визуализации (только отрисовка!)
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

            # --- СГЛАЖИВАНИЕ только для отрисовки ---
            if len(data) >= SMOOTH_WIN:
                smooth = np.convolve(data, kernel, mode='same')
            else:
                smooth = data

            self.curves[i].setData(self.time_axis, smooth, skipFiniteCheck=True)

        now = time.time()
        if now - self.diag_last_log >= 5.0:
            expected = int((now - self.diag_last_log) * SAMPLE_RATE)
            received = self.diag_total_samples
            plots = self.diag_plot_calls
            batches = self.diag_batch_count
            loss = expected - received
            loss_pct = (loss / expected * 100) if expected > 0 else 0
            logger.info(
                f"[OSC DIAG] 5s: expected={expected}, received={received}, "
                f"batches={batches}, plots={plots}, loss={loss} ({loss_pct:.1f}%)"
            )
            self.lbl_diag.setText(
                f"DIAG: loss={loss_pct:.1f}% | batches={batches} | plots={plots}"
            )
            self.diag_total_samples = 0
            self.diag_plot_calls = 0
            self.diag_batch_count = 0
            self.diag_last_log = now

    def _check_worker_results(self):
        """Читает результаты heavy_worker и обновляет индикатор события."""
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
        logger.info("[OSC v7.0] Stopping acquisition, worker, timers...")
        self.acquisition.stop()
        self.event_queue.put('STOP')
        self.heavy_process.join(timeout=3.0)
        if self.heavy_process.is_alive():
            self.heavy_process.terminate()
            logger.warning("[OSC v7.0] Heavy worker terminated forcefully")
        self.plot_timer.stop()
        self.water_timer.stop()
        self.result_timer.stop()
        event.accept()
        logger.info("[OSC v7.0] UI closed cleanly.")


if __name__ == "__main__":
    config.setup_logging()
    app = QApplication(sys.argv)
    win = StandaloneOscilloscope()
    win.show()
    sys.exit(app.exec_())
