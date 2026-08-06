#!/usr/bin/env python3
"""
main.py v9.2.2
SeismicMonitor — полная интеграция acquisition → processor → heavy_worker.
Отдельный events.log для событий, попавших на карту.
ВСЕ сообщения → logger 'seismic' (ddd.log / error.log). Терминал чист.

ИСПРАВЛЕНИЯ v9.2.2:
  - Карусель: данные накапливаются в self.buffers[0] как (t,x,y,z) и корректно
    сдвигаются в панели 2-4 через set_data(), который теперь поддерживает
    dual mode с загрузкой кортежей (widgets.py v9.2.2).
  - scope_id: корректно сдвигается между панелями для привязки S-меток.
"""
import os
import sys
import time
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
from multiprocessing import Process, Queue

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget
)
from PyQt5.QtCore import QTimer

import config
from config import *
from acquisition import DataAcquisitionThread
from processor import SeismicProcessor
from heavy_worker import worker_loop
from widgets import OscilloscopeWidget, MapWidget, WaterAlarmWidget

VERSION = "v9.2.2"

# === ЛОГГЕРЫ ===
logger = logging.getLogger('seismic')

# events.log — ТОЛЬКО события, попавшие на карту
events_logger = logging.getLogger('seismic.events')
events_logger.setLevel(logging.INFO)
events_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'events.log')
events_handler = RotatingFileHandler(
    events_file, maxBytes=512*1024, backupCount=5, encoding='utf-8'
)
events_handler.setLevel(logging.INFO)
events_formatter = logging.Formatter(
    '%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
)
events_handler.setFormatter(events_formatter)
events_logger.addHandler(events_handler)


def print_banner():
    banner = [
        "=" * 60,
        f"  🌍 СЕЙСМОСТАНЦИЯ ZPGEOST {VERSION}",
        "=" * 60,
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
        "=" * 60,
        "  ⌨️  Горячие клавиши карты: O-R-+-Стрелки",
        "=" * 60,
    ]
    for line in banner:
        logger.info(line)


class SeismicMonitor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.event_count = 0
        self.last_event_display = ""
        # scope_ids: уникальные ID для каждой панели карусели
        self.next_scope_id = 4
        self.scope_ids = [0, 1, 2, 3]
        # Буферы данных: списки кортежей (t, x, y, z)
        self.buffers = [[] for _ in range(CAROUSEL_PANELS)]
        print_banner()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(
            f"Сейсмостанция {VERSION} | Событий: {self.event_count} | ±{GRAPH_SENSITIVITY_MV}mV"
        )
        self.resize(1380, 820)

        self.event_queue = Queue(maxsize=QUEUE_MAXSIZE)
        self.result_queue = Queue()
        self.heavy_process = Process(
            target=worker_loop,
            args=(self.event_queue, self.result_queue),
            daemon=True
        )
        self.heavy_process.start()
        logger.info("[MAIN] Heavy worker process started")

        self.processor = SeismicProcessor(
            event_queue=self.event_queue,
            p_onset_callback=lambda t: self.scopes[0].add_p_marker(t)
        )
        self.processor.scope_id_callback = lambda: self.scope_ids[0]

        self.data_thread = DataAcquisitionThread(sample_rate=SAMPLE_RATE, batch_size=20)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        self.map = MapWidget()
        layout.addWidget(self.map, MAP_GRAPH_RATIO)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(2)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right.setMaximumWidth(450)
        right.setMinimumWidth(280)

        self.water_alarm = WaterAlarmWidget()
        right_layout.addWidget(self.water_alarm)

        self.scopes = []
        panel_names = ["[1/4] ТЕКУЩИЕ", "[2/4] ПРЕДЫДУЩИЕ", "[3/4] ПРЕДЫДУЩИЕ", "[4/4] АРХИВ"]

        for i, name in enumerate(panel_names):
            scope = OscilloscopeWidget(name)
            scope.setMinimumHeight(100)
            scope.setMinimumWidth(260)
            if i == 0:
                scope.setStyleSheet("background-color: #050505; border: 2px solid #0f0;")
            else:
                scope.setStyleSheet("background-color: #0a0a0a;")
            right_layout.addWidget(scope, 1)
            self.scopes.append(scope)

        layout.addWidget(right, 1)

        self.counter = 0

        self.map_update_timer = QTimer()
        self.map_update_timer.timeout.connect(self._update_map)
        self.map_update_timer.start(500)

        self.result_timer = QTimer()
        self.result_timer.timeout.connect(self._check_results)
        self.result_timer.start(200)

        self.data_thread.data_ready.connect(self.on_new_data_batch)
        self.data_thread.start()

    def on_new_data_batch(self, batch_volts, batch_raws, batch_ts, total_samples, errors):
        batch_len = len(batch_volts)

        self.processor.process_batch(batch_raws, batch_ts)

        for i in range(batch_len):
            volts = batch_volts[i]
            if len(volts) < 4:
                continue

            x, y, z, water = volts
            t = batch_ts[i]

            # Накапливаем в активный буфер для карусели
            self.buffers[0].append((t, x, y, z))
            self.scopes[0].update_data(t, x, y, z)

            self.counter += 1
            if self.counter >= SAMPLES_PER_SCREEN:
                self._rotate()
                self.counter = 0

            if i == batch_len - 1:
                self.water_alarm.set_alarm(water > WATER_ALARM_THRESHOLD_V, water)

    def _rotate(self):
        """
        Сдвиг карусели вниз.
        buffers[0] → buffers[1] → buffers[2] → buffers[3]
        scope_ids сдвигаются вместе с буферами.
        """
        for i in range(CAROUSEL_PANELS - 1, 0, -1):
            self.buffers[i] = self.buffers[i-1].copy()
            self.scope_ids[i] = self.scope_ids[i-1]
            self.scopes[i].set_data(self.buffers[i])
        # Новый активный буфер
        self.buffers[0] = []
        self.scope_ids[0] = self.next_scope_id
        self.next_scope_id += 1
        self.scopes[0].set_data([])

    def _check_results(self):
        while not self.result_queue.empty():
            try:
                result = self.result_queue.get_nowait()
                if not result or result.get('status') != 'event':
                    continue

                distance = result.get('distance', 0)
                if distance <= DEAD_ZONE_KM:
                    continue

                added = self.map.add_event(result)
                if not added:
                    continue

                self.event_count += 1
                peak_mv = result.get('peak_amplitude_mv', 0)
                ml = result.get('ml_magnitude', 0)
                az = result.get('azimuth', 0)
                conf = result.get('event_confidence', 0)
                depth = result.get('depth', 0)
                etype = result.get('event_type', '?')
                p_s = result.get('p_s_delta', 0)

                # --- S-маркер на карусель ---
                s_time_abs = result.get('s_time_abs')
                scope_id = result.get('scope_id', 0)
                if s_time_abs:
                    try:
                        panel_idx = self.scope_ids.index(scope_id)
                        self.scopes[panel_idx].add_s_marker(s_time_abs)
                    except ValueError:
                        # Панель с этим scope_id уже вытеснена — ставим в текущую
                        self.scopes[0].add_s_marker(s_time_abs)

                # --- events.log: компактная строка ---
                event_line = (
                    f"EVENT #{self.event_count:03d} | {etype.upper():>6s} | "
                    f"D={distance:5.1f}km | Ml={ml:4.2f} | Az={az:5.1f}° | "
                    f"conf={conf:.2f} | depth={depth:4.1f}km | peak={peak_mv:5.2f}mV | "
                    f"ΔP-S={p_s:.2f}s"
                )
                events_logger.info(event_line)

                # --- ddd.log: подробнее ---
                logger.info(f"[EVENT] {event_line}")

                self.last_event_display = f"#{self.event_count} {etype} {distance:.1f}km"
                self.setWindowTitle(
                    f"Сейсмостанция {VERSION} | {self.last_event_display} | ±{GRAPH_SENSITIVITY_MV}mV"
                )

            except Exception as e:
                logger.error(f"[MAIN] Error processing result: {e}")
                break

    def _update_map(self):
        self.map.update()

    def closeEvent(self, event):
        logger.info(f"[MAIN] Остановка сейсмостанции {VERSION}...")
        logger.info(f"[MAIN] Всего событий за сессию: {self.event_count}")
        self.data_thread.stop()
        self.event_queue.put('STOP')
        self.heavy_process.join(timeout=3.0)
        if self.heavy_process.is_alive():
            self.heavy_process.terminate()
            logger.warning("[MAIN] Heavy worker terminated forcefully")
        event.accept()


if __name__ == "__main__":
    try:
        import obspy, scipy, numpy, PyQt5, pyqtgraph
    except ImportError as e:
        print(f"\n❌ Отсутствует: {e.name}")
        sys.exit(1)

    config.setup_logging()
    app = QApplication(sys.argv)
    window = SeismicMonitor()
    window.show()
    sys.exit(app.exec_())
