#!/usr/bin/env python3

import os, sys, time, numpy as np, logging
from logging.handlers import RotatingFileHandler
from multiprocessing import Process, Queue, Event
from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer, Qt
from test_events import inject_test_events
import config
from config import *
from acquisition_process import DataAcquisitionProcess
from processor import SeismicProcessor
from heavy_worker import worker_loop
from widgets import OscilloscopeWidget, MapWidget, WaterAlarmWidget

logger = logging.getLogger('seismic')
events_logger = logging.getLogger('seismic.events')
events_logger.setLevel(logging.INFO)
events_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'events.log')
events_handler = RotatingFileHandler(events_file, maxBytes=512*1024, backupCount=5, encoding='utf-8')
events_handler.setLevel(logging.INFO)
events_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
events_logger.addHandler(events_handler)

def print_banner():
    for line in [
        "=" * 60, f"  🌍 СЕЙСМОСТАНЦИЯ ZPGEOST {config.VERSION}", "=" * 60,
        f"  📡 Частота дискретизации: {SAMPLE_RATE} Гц",
        f"  📊 Буфер экрана: {SAMPLES_PER_SCREEN} отсчётов ({TIME_SCALE} сек)",
        f"  🎯 Порог события: {EVENT_THRESHOLD_MV} мВ",
        f"  📏 Чувствительность осциллографа: ±{GRAPH_SENSITIVITY_MV} мВ",
        f"  🗺️  Карта: {MAP_IMAGE_FILE}",
        f"  💧 Датчик воды: порог {WATER_ALARM_THRESHOLD_V} В",
        f"  ⚡ CPU affinity: DAQ={DAQ_CPU_CORES}, GUI={GUI_CPU_CORES}, PROC={HEAVY_PROCESS_CPU_CORES}",
        f"  🔧 ADC: AD7606B (1 LSB = {ADC_SCALE_V*1e6:.1f} мкВ)",
        f"  📁 Архив: {ARCHIVE_FOLDER}/",
        f"  📝 Логи: ddd.log, error.log, events.log",
        "=" * 60, "  ⌨️  Горячие клавиши карты: O-R-+-Стрелки", "=" * 60,
    ]: logger.info(line)

class SeismicMonitor(QMainWindow):
    def __init__(self):
        super().__init__()
        if config.GUI_CPU_CORES:
            try:
                os.sched_setaffinity(0, config.GUI_CPU_CORES)
                logger.info(f"[MAIN] GUI thread affinity set to {config.GUI_CPU_CORES}")
            except Exception as e:
                logger.warning(f"[MAIN] GUI affinity failed (need root): {e}")
        self.event_count = 0; self.last_event_display = ""
        self.next_scope_id = 4; self.scope_ids = [0, 1, 2, 3]
        self.buffers = [[] for _ in range(CAROUSEL_PANELS)]
        print_banner(); self.init_ui()
        self.data_process.start()
        logger.info("[MAIN] DAQ process started")

    def init_ui(self):
        self.setWindowTitle(f"Сейсмостанция {config.VERSION} | Событий: {self.event_count} | ±{GRAPH_SENSITIVITY_MV}mV")
        self.resize(1380, 820)
        self.event_queue = Queue(maxsize=QUEUE_MAXSIZE); self.result_queue = Queue()
        self.heavy_process = Process(target=worker_loop, args=(self.event_queue, self.result_queue), daemon=True)
        self.heavy_process.start(); logger.info("[MAIN] Heavy worker process started")
        self.processor = SeismicProcessor(event_queue=self.event_queue,
            p_onset_callback=lambda t: self.scopes[0].add_p_marker(t))
        self.processor.scope_id_callback = lambda: self.scope_ids[0]

        self.data_queue = Queue(maxsize=config.DAQ_QUEUE_MAXSIZE)
        self.stop_event = Event()
        self.data_process = DataAcquisitionProcess(
            data_queue=self.data_queue,
            stop_event=self.stop_event,
            sample_rate=config.SAMPLE_RATE,
            batch_size=config.BATCH_SIZE
        )

        central = QWidget(); self.setCentralWidget(central); layout = QHBoxLayout(central)
        self.map = MapWidget(); layout.addWidget(self.map, MAP_GRAPH_RATIO)
        right = QWidget(); right_layout = QVBoxLayout(right)
        right_layout.setSpacing(2); right_layout.setContentsMargins(0, 0, 0, 0)
        right.setMaximumWidth(450); right.setMinimumWidth(280)
        self.water_alarm = WaterAlarmWidget(); right_layout.addWidget(self.water_alarm)
        self.scopes = []
        for i, name in enumerate(["[1/4] ТЕКУЩИЕ", "[2/4] ПРЕДЫДУЩИЕ", "[3/4] ПРЕДЫДУЩИЕ", "[4/4] АРХИВ"]):
            scope = OscilloscopeWidget(name); scope.setMinimumHeight(100); scope.setMinimumWidth(260)
            scope.setStyleSheet("background-color: #050505; border: 2px solid #0f0;" if i == 0 else "background-color: #0a0a0a;")
            right_layout.addWidget(scope, 1); self.scopes.append(scope)
        layout.addWidget(right, 1); self.counter = 0
        self.map_update_timer = QTimer(); self.map_update_timer.timeout.connect(self._update_map); self.map_update_timer.start(config.MAP_UPDATE_MS)
        self.result_timer = QTimer(); self.result_timer.timeout.connect(self._check_results); self.result_timer.start(config.RESULT_TIMER_MS)

        self.daq_timer = QTimer()
        self.daq_timer.timeout.connect(self._poll_daq_queue)
        self.daq_timer.start(20)

    def _poll_daq_queue(self):
        while not self.data_queue.empty():
            try:
                batch = self.data_queue.get_nowait()
                batch_volts, batch_raws, batch_ts, total_samples, errors = batch
                self.on_new_data_batch(batch_volts, batch_raws, batch_ts, total_samples, errors)
            except Exception:
                break

    def on_new_data_batch(self, batch_volts, batch_raws, batch_ts, total_samples, errors):
        batch_len = len(batch_volts); self.processor.process_batch(batch_raws, batch_ts)
        for i in range(batch_len):
            volts = batch_volts[i]
            if len(volts) < 4: continue
            x, y, z, water = volts; t = batch_ts[i]
            p_amp = (x*x + y*y) ** 0.5
            self.buffers[0].append((t, x, y, z, p_amp)); self.scopes[0].update_data(t, x, y, z)
            self.counter += 1
            if self.counter >= SAMPLES_PER_SCREEN: self._rotate(); self.counter = 0
            if i == batch_len - 1: self.water_alarm.set_alarm(water > WATER_ALARM_THRESHOLD_V, water)

    def _rotate(self):
        for i in range(CAROUSEL_PANELS - 1, 0, -1):
            self.buffers[i] = list(self.buffers[i-1])
            self.scope_ids[i] = self.scope_ids[i-1]
            self.scopes[i].set_data(self.buffers[i])
        self.buffers[0] = []; self.scope_ids[0] = self.next_scope_id; self.next_scope_id += 1
        self.scopes[0].set_data([])

    def _check_results(self):
        while not self.result_queue.empty():
            try:
                result = self.result_queue.get_nowait()
                if not result or result.get('status') != 'event': continue
                distance = result.get('distance')
                if distance is not None and distance <= DEAD_ZONE_KM: continue
                added = self.map.add_event(result)
                if not added: continue
                self.event_count += 1
                peak_mv = result.get('peak_amplitude_mv', 0); ml = result.get('ml_magnitude', 0)
                az = result.get('azimuth', 0); conf = result.get('event_confidence', 0)
                depth = result.get('depth', 0); etype = result.get('event_type', '?'); p_s = result.get('p_s_delta', 0)
                s_time_abs = result.get('s_time_abs')
                if s_time_abs: self.scopes[0].add_s_marker(s_time_abs)
                dist_str = f"{distance:5.1f}km" if distance is not None else "N/A   "
                depth_str = f"{depth:4.1f}km" if depth is not None else "N/A "
                p_s_str = f"{p_s:.2f}s" if p_s and p_s > 0 else "N/A"
                event_line = (f"EVENT #{self.event_count:03d} | {etype.upper():>6s} | "
                              f"D={dist_str} | Ml={ml:4.2f} | Az={az:5.1f}° | "
                              f"conf={conf:.2f} | depth={depth_str} | peak={peak_mv:5.2f}mV | "
                              f"ΔP-S={p_s_str}")
                events_logger.info(event_line); logger.info(f"[EVENT] {event_line}")
                self.last_event_display = f"#{self.event_count} {etype} {distance:.1f}km"
                self.setWindowTitle(f"Сейсмостанция {config.VERSION} | {self.last_event_display} | ±{GRAPH_SENSITIVITY_MV}mV")
            except Exception as e: logger.error(f"[MAIN] Error processing result: {e}"); break

    def _update_map(self): self.map.update()

    def closeEvent(self, event):
        logger.info(f"[MAIN] Остановка сейсмостанции {config.VERSION}...")
        logger.info(f"[MAIN] Всего событий за сессию: {self.event_count}")
        self.stop_event.set()
        self.data_process.join(timeout=3.0)
        if self.data_process.is_alive():
            self.data_process.terminate()
            logger.warning("[MAIN] DAQ process terminated forcefully")
        self.event_queue.put('STOP')
        self.heavy_process.join(timeout=config.JOIN_TIMEOUT_SEC)
        if self.heavy_process.is_alive():
            self.heavy_process.terminate()
            logger.warning("[MAIN] Heavy worker terminated forcefully")
        event.accept()

    def keyPressEvent(self, event):
        # Горячие клавиши главного окна.
        key = event.key()
        if key == Qt.Key_J:
            inject_test_events(self.map, self.scopes)
            self.setWindowTitle(
                f"Сейсмостанция {config.VERSION} | ТЕСТ: события на карте | ±{GRAPH_SENSITIVITY_MV}mV"
            )
        else:
            super().keyPressEvent(event)

if __name__ == "__main__":
    try: import obspy, scipy, numpy, PyQt5, pyqtgraph
    except ImportError as e: print(f"\n❌ Отсутствует: {e.name}"); sys.exit(1)
    config.setup_logging()
    try:
        config.validate_config()
        logger.info("[MAIN] Config validation PASSED")
    except AssertionError as e:
        print(f"\n❌ ОШИБКА КОНФИГУРАЦИИ: {e}")
        sys.exit(1)
    app = QApplication(sys.argv); window = SeismicMonitor(); window.show()
    sys.exit(app.exec_())
