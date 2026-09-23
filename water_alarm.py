"""
water_alarm.py v9.6.34
Виджет датчика воды: ПОТОП/СУХО с миганием.
"""
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QFont
from PyQt5.QtCore import QTimer

import config


class WaterAlarmWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.alarm = False; self.voltage = 0.0; self.blink_state = False
        self.setMinimumHeight(40); self.setMaximumHeight(60)
        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self._blink)
        self.blink_timer.start(config.BLINK_TIMER_MS)

    def _blink(self):
        if self.alarm:
            self.blink_state = not self.blink_state; self.update()
        else:
            if self.blink_state:
                self.blink_state = False; self.update()

    def set_alarm(self, alarm: bool, voltage: float):
        self.alarm = bool(alarm)
        try:
            self.voltage = float(voltage)
        except Exception:
            self.voltage = 0.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self); painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        if self.alarm:
            bg = QColor(220, 20, 20) if self.blink_state else QColor(120, 10, 10)
            text = f"ПОТОП: {self.voltage:.2f} V"
        else:
            bg = QColor(30, 120, 30); text = f"СУХО: {self.voltage:.2f} V"
        painter.fillRect(0, 0, w, h, bg)
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
        painter.drawText(10, h // 2 + 6, text)
        painter.end()