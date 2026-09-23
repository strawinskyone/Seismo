"""
map_widget.py v10.0.0
Карта сейсмостанции: отрисовка событий, пульсация, стрелки для далёких,
перетаскивание станции, масштабирование.
"""
import time
import json
import os
import logging
import numpy as np
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import (QPainter, QPen, QColor, QPolygon, QFont,
                         QPixmap, QCursor, QPainterPath)
from PyQt5.QtCore import Qt, QTimer, QPoint, pyqtSignal, QRect, QRectF

import config
from config import *

logger = logging.getLogger('seismic')

POS_FILE = "station_position.json"


class SeismicEvent:
    def __init__(self, data):
        self.magnitude = data.get('magnitude', 0.0) or 0.0
        self.ml_magnitude = data.get('ml_magnitude', None)
        if self.ml_magnitude is None:
            self.ml_magnitude = float('nan')
        self.peak_amp = data.get('peak_amplitude', self.magnitude)
        self.azimuth = data['azimuth']
        self.distance = data.get('distance', 50)
        self.depth = data.get('depth', 0.5)
        self.timestamp = time.time()
        self.vector = data.get('vector', (0, 0, 0))
        self.birth_time = time.time()
        self.event_type = data.get('event_type', 'unknown')
        self.event_confidence = data.get('event_confidence', 0.5)
        self.spectral_features = data.get('spectral_features', {})
        self.is_p = data.get('is_p', False)
        self.is_s = data.get('is_s', False)
        self.p_s_delta = data.get('p_s_delta', None)

    def get_opacity(self):
        age = time.time() - self.birth_time
        if age > FADE_OUT_SECONDS:
            return 0.0
        return 1.0 - (age / FADE_OUT_SECONDS)

    def is_visible(self):
        return self.get_opacity() > 0.01

    def get_pulse_radius(self, pulse_index=0):
        age = time.time() - self.birth_time
        adjusted_age = age - (pulse_index * 0.3)
        if adjusted_age < 0 or adjusted_age > PULSE_FADE_SECONDS:
            return None
        if PULSE_USE_MAGNITUDE and not np.isnan(self.ml_magnitude):
            base_radius = self.ml_magnitude * PULSE_MAG_SCALE
        else:
            base_radius = PULSE_BASE_RADIUS_KM
        expansion = adjusted_age * 0.2
        return min(base_radius + expansion, PULSE_MAX_RADIUS_KM)

    def clone(self):
        new = SeismicEvent.__new__(SeismicEvent)
        new.__dict__.update(self.__dict__)
        return new


class MapWidget(QWidget):
    stationMoved = pyqtSignal(float, float)
    mapScaled = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.events = []
        self.setMinimumWidth(600)
        self.setStyleSheet("background-color: #050510;")
        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self.update)
        self.anim_timer.start(config.MAP_UPDATE_MS)

        self._pixel_scale = 1.0
        self.map_scale = MAP_SCALE_KM_PER_PIXEL

        self.map_pixmap = None
        self.map_offset_x = 0
        self.map_offset_y = 0

        self._load_map()

        pos = self._load_position()
        self.station_offset_x = pos.get('station_offset_x', globals().get('STATION_OFFSET_X', 0))
        self.station_offset_y = pos.get('station_offset_y', globals().get('STATION_OFFSET_Y', 0))
        self.map_offset_x = pos.get('map_offset_x', 0)
        self.map_offset_y = pos.get('map_offset_y', 0)

        self.dragging = False
        self.dragging_map = False
        self.drag_start = None
        self.station_start = None
        self.map_start = None
        self._map_visible = True
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

    def _load_position(self):
        self._pixel_scale = 1.0
        try:
            if os.path.exists(POS_FILE):
                with open(POS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'pixel_scale' in data:
                        self._pixel_scale = data['pixel_scale']
                    elif 'map_scale' in data:
                        self._pixel_scale = data['map_scale'] / MAP_SCALE_KM_PER_PIXEL
                    self.map_scale = self._pixel_scale * MAP_SCALE_KM_PER_PIXEL
                    return {
                        'station_offset_x': data.get('station_offset_x', 0),
                        'station_offset_y': data.get('station_offset_y', 0),
                        'map_offset_x': data.get('map_offset_x', 0),
                        'map_offset_y': data.get('map_offset_y', 0),
                    }
        except Exception as e:
            logger.warning(f"[MAP] Ошибка чтения {POS_FILE}: {e}")
        return {}

    def _save_position(self):
        try:
            with open(POS_FILE, 'w', encoding='utf-8') as f:
                json.dump({
                    'station_offset_x': self.station_offset_x,
                    'station_offset_y': self.station_offset_y,
                    'map_offset_x': self.map_offset_x,
                    'map_offset_y': self.map_offset_y,
                    'pixel_scale': self._pixel_scale
                }, f, indent=2)
        except Exception as e:
            logger.error(f"[MAP] Ошибка записи {POS_FILE}: {e}")

    def _load_map(self):
        try:
            self.map_pixmap = QPixmap(MAP_IMAGE_FILE)
            if self.map_pixmap.isNull():
                logger.warning(f"[MAP] Карта {MAP_IMAGE_FILE} не загружена")
                self.map_pixmap = None
                return
            pw = self.map_pixmap.size().width()
            ph = self.map_pixmap.size().height()
            logger.info(f"[MAP] Карта загружена: {pw}x{ph}px")
            w, h = self.width(), self.height()
            if w > 100 and h > 100 and pw > 0 and ph > 0:
                pixel_scale = min((w * 0.85) / pw, (h * 0.85) / ph)
                pixel_scale = max(MAP_MIN_SCALE, min(pixel_scale, MAP_MAX_SCALE))
                self._pixel_scale = pixel_scale
                self.map_scale = pixel_scale * MAP_SCALE_KM_PER_PIXEL
                logger.info(f"[MAP] Авто-масштаб: {self.map_scale:.4f} км/px "
                            f"(pixel_scale={pixel_scale:.2f})")
        except Exception as e:
            logger.warning(f"[MAP] Ошибка загрузки карты: {e}")
            self.map_pixmap = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            cx = self.width() // 2 + self.station_offset_x
            cy = self.height() // 2 + self.station_offset_y
            dx = event.x() - cx; dy = event.y() - cy
            if dx*dx + dy*dy < 400:
                self.dragging = True; self.drag_start = event.pos()
                self.station_start = (self.station_offset_x, self.station_offset_y)
                self.setCursor(QCursor(Qt.ClosedHandCursor))
            else:
                self.dragging_map = True; self.drag_start = event.pos()
                self.map_start = (self.map_offset_x, self.map_offset_y)
        elif event.button() == Qt.RightButton:
            self.reset_station_position()

    def mouseMoveEvent(self, event):
        if self.dragging:
            dx = event.x() - self.drag_start.x(); dy = event.y() - self.drag_start.y()
            self.station_offset_x = self.station_start[0] + dx
            self.station_offset_y = self.station_start[1] + dy
            self.update()
        elif self.dragging_map:
            dx = event.x() - self.drag_start.x(); dy = event.y() - self.drag_start.y()
            self.map_offset_x = self.map_start[0] + dx
            self.map_offset_y = self.map_start[1] + dy
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.dragging:
                self.dragging = False; self.setCursor(QCursor(Qt.ArrowCursor))
                self._save_position(); self.stationMoved.emit(self.station_offset_x, self.station_offset_y)
            if self.dragging_map:
                self.dragging_map = False; self._save_position()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        modifiers = event.modifiers()
        if modifiers & Qt.ShiftModifier:
            scale_factor = 1.15 if delta > 0 else 0.87
        else:
            scale_factor = 1.02 if delta > 0 else 0.98

        self._pixel_scale *= scale_factor
        self._pixel_scale = max(MAP_MIN_SCALE, min(MAP_MAX_SCALE, self._pixel_scale))
        self.map_scale = self._pixel_scale * MAP_SCALE_KM_PER_PIXEL
        self.mapScaled.emit(self.map_scale)
        self.update()

    def keyPressEvent(self, event):
        step = 5
        if event.key() == Qt.Key_Left:
            self.station_offset_x -= step
        elif event.key() == Qt.Key_Right:
            self.station_offset_x += step
        elif event.key() == Qt.Key_Up:
            self.station_offset_y -= step
        elif event.key() == Qt.Key_Down:
            self.station_offset_y += step
        elif event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            self._pixel_scale = min(MAP_MAX_SCALE, self._pixel_scale * 1.02)
            self.map_scale = self._pixel_scale * MAP_SCALE_KM_PER_PIXEL
        elif event.key() == Qt.Key_Minus:
            self._pixel_scale = max(MAP_MIN_SCALE, self._pixel_scale / 1.02)
            self.map_scale = self._pixel_scale * MAP_SCALE_KM_PER_PIXEL
        elif event.text().lower() in ('o', 'щ'):
            if self.map_pixmap and not self.map_pixmap.isNull():
                self._map_visible = not getattr(self, '_map_visible', True)
                self.update()
            return
        elif event.text().lower() in ('r', 'к'):
            self.reset_station_position()
            return
        else:
            super().keyPressEvent(event)
            return
        self.update()

    def reset_station_position(self):
        self.station_offset_x = 0
        self.station_offset_y = 0
        self.map_offset_x = 0
        self.map_offset_y = 0
        if self.map_pixmap and not self.map_pixmap.isNull():
            w, h = self.width(), self.height()
            pw, ph = self.map_pixmap.size().width(), self.map_pixmap.size().height()
            if w > 100 and h > 100 and pw > 0 and ph > 0:
                pixel_scale = min((w * 0.85) / pw, (h * 0.85) / ph)
                pixel_scale = max(MAP_MIN_SCALE, min(pixel_scale, MAP_MAX_SCALE))
                self._pixel_scale = pixel_scale
                self.map_scale = pixel_scale * MAP_SCALE_KM_PER_PIXEL
            else:
                self._pixel_scale = MAP_MIN_SCALE
                self.map_scale = self._pixel_scale * MAP_SCALE_KM_PER_PIXEL
        else:
            self._pixel_scale = MAP_MIN_SCALE
            self.map_scale = self._pixel_scale * MAP_SCALE_KM_PER_PIXEL
        try:
            if os.path.exists(POS_FILE):
                os.remove(POS_FILE)
        except:
            pass
        self.update()

    def add_event(self, data):
        peak_mv = data.get('peak_amplitude_mv', 0)
        if peak_mv < EVENT_THRESHOLD_MV:
            logger.debug(f"[MAP] Событие отфильтровано: peak={peak_mv:.2f}mV < threshold={EVENT_THRESHOLD_MV}mV")
            return False
        distance = data.get('distance')
        if distance is None or distance <= DEAD_ZONE_KM:
            logger.debug(f"[MAP] Событие отфильтровано: distance={distance} <= dead_zone={DEAD_ZONE_KM}km")
            return False
        event = SeismicEvent(data)
        self.events.append(event)
        self.events = [e for e in self.events if e.get_opacity() > 0.01]
        if len(self.events) > MAX_EVENTS_ON_MAP:
            self.events.sort(key=lambda e: e.birth_time, reverse=True)
            self.events = self.events[:MAX_EVENTS_ON_MAP]
        return True

    def _group_events_by_location(self, events):
        if not events:
            return []
        grouped = {}
        for evt in events:
            found_group = False
            for key in list(grouped.keys()):
                existing_evt = grouped[key]
                distance_diff = abs(evt.distance - existing_evt.distance)
                if distance_diff > 3.0:
                    continue
                time_diff = abs(evt.birth_time - existing_evt.birth_time)
                if time_diff > 0.5:
                    continue
                azimuth_diff = abs(evt.azimuth - existing_evt.azimuth)
                if azimuth_diff > 180:
                    azimuth_diff = 360 - azimuth_diff
                if azimuth_diff <= 15.0:
                    if evt.magnitude > existing_evt.magnitude:
                        new_evt = evt.clone()
                    else:
                        new_evt = existing_evt.clone()
                    new_evt.azimuth = (evt.azimuth + existing_evt.azimuth) / 2
                    new_evt.magnitude = max(evt.magnitude, existing_evt.magnitude)
                    grouped[key] = new_evt
                    found_group = True
                    break
            if not found_group:
                key = (round(evt.azimuth, 1), round(evt.distance, 1), round(evt.birth_time, 1))
                grouped[key] = evt
        return list(grouped.values())

    def _draw_event_labels(self, painter, events, cx, cy, px_per_km):
        placed_rects = []
        font = QFont("Segoe UI", MAP_LABEL_FONT_SIZE, QFont.Bold)
        painter.setFont(font)
        fm = painter.fontMetrics()
        line_h = fm.height() + 2

        sorted_events = sorted(events, key=lambda e: (e.distance, -e.magnitude))

        for evt in sorted_events:
            opacity = evt.get_opacity()
            if opacity < 0.01:
                continue

            r_px = evt.distance * px_per_km
            rad = np.radians(evt.azimuth)
            ex = int(cx + r_px * np.sin(rad))
            ey = int(cy - r_px * np.cos(rad))

            if evt.distance <= DEAD_ZONE_KM:
                continue

            if evt.event_type == 'explosion':
                base_color = QColor(255, 200, 50)
                type_letter = 'X'
            elif evt.event_type in ('earthquake', 'quake'):
                base_color = QColor(220, 20, 60)
                type_letter = 'M'
            else:
                base_color = QColor(100, 180, 255)
                type_letter = '?'

            dom_freq = ""
            if evt.spectral_features:
                f = evt.spectral_features.get('dominant_freq', 0)
                if f > 0:
                    dom_freq = f"@{f:.0f}Hz"
            if np.isnan(evt.ml_magnitude):
                label1 = f"{type_letter}?"
            else:
                label1 = f"{type_letter}{evt.ml_magnitude:.2f}"
            label2 = f"{evt.distance:.1f}км"
            if evt.p_s_delta:
                label2 += f" Δ{evt.p_s_delta:.1f}s"

            tw1 = fm.horizontalAdvance(label1)
            tw2 = fm.horizontalAdvance(label2)
            max_tw = max(tw1, tw2) + 10
            total_h = line_h * 2 + 8

            margin = 8
            tail = 8

            cand = {
                'top':         QRect(ex - max_tw // 2, ey - total_h - margin - tail, max_tw, total_h),
                'bottom':      QRect(ex - max_tw // 2, ey + margin + tail, max_tw, total_h),
                'left':        QRect(ex - max_tw - margin - tail, ey - total_h // 2, max_tw, total_h),
                'right':       QRect(ex + margin + tail, ey - total_h // 2, max_tw, total_h),
                'top-left':    QRect(ex - max_tw - margin - tail, ey - total_h - margin - tail, max_tw, total_h),
                'top-right':   QRect(ex + margin + tail, ey - total_h - margin - tail, max_tw, total_h),
                'bottom-left': QRect(ex - max_tw - margin - tail, ey + margin + tail, max_tw, total_h),
                'bottom-right':QRect(ex + margin + tail, ey + margin + tail, max_tw, total_h),
            }

            order = []
            if ex >= cx:
                order.extend(['left', 'top-left', 'bottom-left'])
            else:
                order.extend(['right', 'top-right', 'bottom-right'])
            if ey >= cy:
                order.extend(['top', 'top-left', 'top-right'])
            else:
                order.extend(['bottom', 'bottom-left', 'bottom-right'])

            seen = set()
            final_order = []
            for o in order:
                if o not in seen:
                    seen.add(o)
                    final_order.append(o)
            for o in ['top', 'bottom', 'left', 'right',
                      'top-left', 'top-right', 'bottom-left', 'bottom-right']:
                if o not in seen:
                    final_order.append(o)

            best_rect = None
            for key in final_order:
                rect = cand[key]
                if rect.left() < 4 or rect.right() > self.width() - 4:
                    continue
                if rect.top() < 4 or rect.bottom() > self.height() - 4:
                    continue
                padded = rect.adjusted(-4, -4, 4, 4)
                ok = True
                for pr in placed_rects:
                    if padded.intersects(pr):
                        ok = False
                        break
                if ok:
                    best_rect = rect
                    break

            if best_rect is None:
                for key in final_order:
                    rect = cand[key]
                    if (rect.left() >= 4 and rect.right() <= self.width() - 4 and
                            rect.top() >= 4 and rect.bottom() <= self.height() - 4):
                        best_rect = rect
                        break
                if best_rect is None:
                    continue

            placed_rects.append(best_rect.adjusted(-4, -4, 4, 4))

            tip = QPoint(ex, ey)
            path = self._bubble_path(best_rect, tip, radius=6, tail_width=10)

            bg = QColor(base_color)
            bg.setAlpha(int(220 * opacity))
            border = QColor(base_color)
            border.setAlpha(int(255 * opacity))

            painter.setPen(Qt.NoPen)
            painter.setBrush(bg)
            painter.drawPath(path)

            painter.setPen(QPen(border, 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)

            painter.setPen(QColor(255, 255, 255, int(255 * opacity)))
            tx = best_rect.x() + 5
            ty = best_rect.y() + line_h + 2
            painter.drawText(tx, ty, label1)
            painter.drawText(tx, ty + line_h, label2)

    def paintEvent(self, event):
        if not hasattr(self, 'map_scale'):
            return
        w, h = self.width(), self.height()
        cx = w // 2 + self.station_offset_x
        cy = h // 2 + self.station_offset_y
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.fillRect(0, 0, w, h, QColor(5, 5, 15))
        if self.map_pixmap and not self.map_pixmap.isNull() and getattr(self, '_map_visible', True):
            try:
                base_scale = self.map_scale / globals().get('MAP_SCALE_KM_PER_PIXEL', 0.5)
            except:
                base_scale = 1.0
            scaled_w = int(self.map_pixmap.width() * base_scale)
            scaled_h = int(self.map_pixmap.height() * base_scale)
            scaled_pixmap = self.map_pixmap.scaled(scaled_w, scaled_h,
                                       Qt.KeepAspectRatio, Qt.FastTransformation)
            map_x = cx - scaled_w // 2 + self.map_offset_x
            map_y = cy - scaled_h // 2 + self.map_offset_y
            painter.drawPixmap(map_x, map_y, scaled_pixmap)
        px_per_km = (min(w, h) / 2) / MAP_MAX_DISTANCE_KM
        pen_grid = QPen(QColor(80, 140, 200), 1, Qt.DotLine)
        painter.setPen(pen_grid)
        for dist in range(MAP_STEP_KM, MAP_MAX_DISTANCE_KM + 1, MAP_STEP_KM):
            r = dist * px_per_km
            painter.drawEllipse(int(cx - r), int(cy - r), int(2*r), int(2*r))
            painter.setPen(QColor(180, 200, 240))
            painter.setFont(QFont("Segoe UI", 10))
            painter.drawText(int(cx + 3), int(cy - r + 3), f"{dist}")
            painter.setPen(pen_grid)
        painter.setPen(QPen(QColor(80, 80, 80), 1))
        painter.drawLine(cx, 0, cx, h); painter.drawLine(0, cy, w, cy)
        painter.setPen(QColor(150, 150, 200)); painter.setFont(QFont("Segoe UI", 10, QFont.Light))
        painter.drawText(cx - 4, 14, "N"); painter.drawText(cx - 4, h - 4, "S")
        painter.drawText(w - 12, cy + 4, "E"); painter.drawText(4, cy + 4, "W")
        painter.setPen(QPen(QColor(255, 50, 50), 2)); painter.setBrush(QColor(255, 0, 0))
        painter.drawEllipse(cx - 6, cy - 6, 12, 12)
        painter.setPen(QColor(255, 255, 255)); painter.setFont(QFont("Segoe UI", 10, QFont.Light))
        painter.drawText(cx + 12, cy - 12, "СТАНЦИЯ")
        visible_events = [e for e in self.events
                         if e.distance <= MAP_MAX_DISTANCE_KM and e.get_opacity() > 0.01]
        grouped_events = self._group_events_by_location(visible_events)
        for evt in grouped_events:
            opacity = evt.get_opacity()
            if opacity < 0.01:
                continue
            r_px = evt.distance * px_per_km
            rad = np.radians(evt.azimuth)
            ex = int(cx + r_px * np.sin(rad))
            ey = int(cy - r_px * np.cos(rad))
            if evt.distance <= DEAD_ZONE_KM:
                continue
            if evt.event_type == 'explosion':
                pulse_color = QColor(255, 200, 50, int(opacity * 0.45 * 255))
            elif evt.event_type in ('earthquake', 'quake'):
                pulse_color = QColor(255, 50, 50, int(opacity * 0.45 * 255))
            else:
                pulse_color = QColor(100, 160, 255, int(opacity * 0.4 * 255))
            for i in range(3):
                pulse_r = evt.get_pulse_radius(i)
                if pulse_r is not None and pulse_r > 0:
                    pulse_px = pulse_r * px_per_km; pulse_px = min(pulse_px, 60.0)
                    if pulse_px < 2:
                        continue
                    painter.setPen(QPen(pulse_color, 1)); painter.setBrush(Qt.NoBrush)
                    painter.drawEllipse(int(ex - pulse_px), int(ey - pulse_px),
                                       int(2 * pulse_px), int(2 * pulse_px))
            if evt.event_type == 'explosion':
                center_color = QColor(255, 220, 0, int(255 * opacity))
            elif evt.event_type in ('earthquake', 'quake'):
                center_color = QColor(220, 20, 60, int(255 * opacity))
            else:
                center_color = QColor(100, 180, 255, int(255 * opacity))
            painter.setPen(QPen(center_color, 2)); painter.setBrush(center_color)
            painter.drawEllipse(ex - 5, ey - 5, 10, 10)
        self._draw_event_labels(painter, grouped_events, cx, cy, px_per_km)
        out_of_bounds_events = [e for e in self.events
                               if MAP_MAX_DISTANCE_KM < e.distance <= MAP_ARROW_MAX_DISTANCE_KM
                               and e.get_opacity() > 0.01]
        grouped_arrows = self._group_events_by_location(out_of_bounds_events)
        for evt in grouped_arrows:
            arrow_age = time.time() - evt.birth_time
            if arrow_age > FADE_OUT_SECONDS:
                continue
            progress = arrow_age / 50
            rad = np.radians(evt.azimuth)
            r_max = min(w, h) / 2 - 25
            if progress < 0.1:
                position_r = (progress / 0.1) * r_max; arrow_opacity = 1.0
            else:
                position_r = r_max
                fade_progress = (progress - 0.1) / 0.9
                arrow_opacity = 1.0 - fade_progress
            arrow_alpha = int(255 * arrow_opacity)
            ex = int(cx + position_r * np.sin(rad))
            ey = int(cy - position_r * np.cos(rad))
            arrow_length = 20
            start_r = position_r - arrow_length / 2
            end_r = position_r + arrow_length / 2
            start_x = cx + start_r * np.sin(rad); start_y = cy - start_r * np.cos(rad)
            end_x = cx + end_r * np.sin(rad); end_y = cy - end_r * np.cos(rad)
            if evt.event_type == 'explosion':
                arrow_color = QColor(255, 220, 0, arrow_alpha); type_letter = 'X'
            elif evt.event_type in ('earthquake', 'quake'):
                arrow_color = QColor(220, 20, 60, arrow_alpha); type_letter = 'M'
            else:
                arrow_color = QColor(100, 180, 255, arrow_alpha); type_letter = '?'
            painter.setPen(QPen(arrow_color, MAP_ARROW_WIDTH, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(int(start_x), int(start_y), int(end_x), int(end_y))
            arrow_size = 8
            dx = np.sin(rad); dy = -np.cos(rad)
            back_x = end_x - dx * arrow_size; back_y = end_y - dy * arrow_size
            perp_x = -dy; perp_y = dx
            left_x = back_x + perp_x * arrow_size * 0.45
            left_y = back_y + perp_y * arrow_size * 0.45
            right_x = back_x - perp_x * arrow_size * 0.45
            right_y = back_y - perp_y * arrow_size * 0.45
            tip_x = end_x; tip_y = end_y
            arrow_head = QPolygon([QPoint(int(tip_x), int(tip_y)),
                                   QPoint(int(left_x), int(left_y)),
                                   QPoint(int(right_x), int(right_y))])
            painter.setBrush(arrow_color); painter.setPen(QPen(arrow_color, 1))
            painter.drawPolygon(arrow_head)
            if position_r > r_max * 0.9:
                text_offset = 15
                text_x = cx + (position_r + text_offset) * np.sin(rad)
                text_y = cy - (position_r + text_offset) * np.cos(rad)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(0, 0, 0, int(200 * arrow_opacity)))
                painter.setFont(QFont("Segoe UI", MAP_LABEL_FONT_SIZE, QFont.Light))
                fm = painter.fontMetrics()
                if np.isnan(evt.ml_magnitude):
                    lbl1 = f"{type_letter}?"
                else:
                    lbl1 = f"{type_letter}{evt.ml_magnitude:.2f}"
                lbl2 = f"{evt.distance:.1f}км"
                tw = max(fm.horizontalAdvance(lbl1), fm.horizontalAdvance(lbl2)) + 6
                painter.drawRect(int(text_x - 2), int(text_y - 14), int(tw), 28)
                painter.setPen(QColor(255, 255, 255, arrow_alpha))
                painter.setFont(QFont("Segoe UI", MAP_LABEL_FONT_SIZE, QFont.Light))
                painter.drawText(int(text_x), int(text_y), lbl1)
                painter.drawText(int(text_x), int(text_y) + 18, lbl2)
        painter.setPen(QColor(200, 200, 200)); painter.setFont(QFont("Segoe UI", 9))
        hint = "ЛКМ-тащить станцию, ПКМ-сброс, Колесо-масштаб, Стрелки-точно, R-сброс, O-карта"
        painter.drawText(10, h - 10, hint)
        painter.end()

    def _bubble_path(self, rect, tip, radius=6, tail_width=10):
        path = QPainterPath()
        path.addRoundedRect(QRectF(rect), radius, radius)

        cx, cy = rect.center().x(), rect.center().y()
        dx = tip.x() - cx
        dy = tip.y() - cy
        tw = tail_width

        if abs(dx) > abs(dy):
            if dx > 0:
                my = rect.center().y()
                p1 = QPoint(rect.right(), my - tw // 2)
                p2 = QPoint(rect.right(), my + tw // 2)
            else:
                my = rect.center().y()
                p1 = QPoint(rect.left(), my + tw // 2)
                p2 = QPoint(rect.left(), my - tw // 2)
        else:
            if dy > 0:
                mx = rect.center().x()
                p1 = QPoint(mx + tw // 2, rect.bottom())
                p2 = QPoint(mx - tw // 2, rect.bottom())
            else:
                mx = rect.center().x()
                p1 = QPoint(mx - tw // 2, rect.top())
                p2 = QPoint(mx + tw // 2, rect.top())

        path.moveTo(tip.x(), tip.y())
        path.lineTo(p1.x(), p1.y())
        path.lineTo(p2.x(), p2.y())
        path.closeSubpath()
        return path