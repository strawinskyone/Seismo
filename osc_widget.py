"""
Осциллограф/карусель.
Два режима отображения (config.CAROUSEL_DRAW_MODE):
    FULL  — полные обработанные сигналы: H и Z
    CURVE — огибающие: env_h и env_z
Маркеры P/S — на дне графика.
"""
import numpy as np
from collections import deque
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg

import config

SPS = config.SAMPLES_PER_SCREEN
BG = config.GRAPH_BACKGROUND_COLOR
LC = config.GRAPH_LINE_COLOR
PC = config.GRAPH_POINT_COLOR
LW = max(1.0, getattr(config, 'LINE_WIDTH', 1.0))
PS = max(6, getattr(config, 'GRAPH_POINT_SIZE', 6))
DS = getattr(config, 'CAROUSEL_DOWNSAMPLE', 1)
SENS = getattr(config, 'GRAPH_SENSITIVITY_MV', 100.0) / 1000.0
SR = getattr(config, 'SAMPLE_RATE', 400)


class OscilloscopeWidget(pg.PlotWidget):

    def __init__(self, title="Channel"):
        super().__init__()
        self.title = title
        self.mode = 'single'
        self._dirty = False

        self.setMinimumHeight(100)
        self.setBackground(BG)
        self.plotItem.showGrid(x=True, y=True, alpha=0.15)
        self.plotItem.setMouseEnabled(False, False)
        self.plotItem.setMenuEnabled(False)
        self.plotItem.hideButtons()
        self.plotItem.hideAxis('left')
        self.plotItem.hideAxis('bottom')

        # --- Одноканальный режим (legacy) ---
        self.raw_data = deque(maxlen=SPS)
        self.plot_data = pg.PlotDataItem(
            pen=pg.mkPen(LC, width=LW), downsample=DS, clipToView=True
        )
        self.plotItem.addItem(self.plot_data)
        self.scatter = pg.ScatterPlotItem(
            pen=pg.mkPen(PC), brush=pg.mkBrush(PC), size=PS
        )
        self.plotItem.addItem(self.scatter)

        # --- Двухканальный режим (карусель) ---
        self.dual_t = deque(maxlen=SPS)
        self.dual_h = deque(maxlen=SPS)       # H-сигнал (FULL)
        self.dual_z = deque(maxlen=SPS)       # Z-сигнал (FULL)
        self.dual_env_h = deque(maxlen=SPS)   # огибающая H (CURVE)
        self.dual_env_z = deque(maxlen=SPS)   # огибающая Z (CURVE)

        pen_p = pg.mkPen(color='#00ff64', width=1.2)
        pen_z = pg.mkPen(color='#00c8ff', width=1.2)
        self.curve_p = pg.PlotDataItem(
            pen=pen_p, name='H (N,E)', downsample=DS, clipToView=True
        )
        self.curve_z = pg.PlotDataItem(
            pen=pen_z, name='Z', downsample=DS, clipToView=True
        )
        self.plotItem.addItem(self.curve_p)
        self.plotItem.addItem(self.curve_z)
        self.curve_p.setVisible(False)
        self.curve_z.setVisible(False)

        # --- Маркеры P/S ---
        self.p_markers = pg.ScatterPlotItem(
            symbol='t', size=10,
            brush=pg.mkBrush('#00ff64'),
            pen=pg.mkPen(color='#004d1a', width=1)
        )
        self.plotItem.addItem(self.p_markers)
        self.s_markers = pg.ScatterPlotItem(
            symbol='d', size=10,
            brush=pg.mkBrush('#ff3232'),
            pen=pg.mkPen(color='#4d0000', width=1)
        )
        self.plotItem.addItem(self.s_markers)
        self.p_marker_times = []
        self.s_marker_times = []

        self._refresh_timer = QTimer()
        self._refresh_timer.timeout.connect(self._refresh)
        self._refresh_timer.start(config.REFRESH_TIMER_MS)
        self.setFocusPolicy(Qt.NoFocus)

    # --------------------------------------------------------
    # Одноканальный режим (legacy)
    # --------------------------------------------------------
    def add_point(self, value):
        if self.mode != 'single':
            return
        try:
            self.raw_data.append(float(value))
        except (ValueError, TypeError):
            return
        self._dirty = True

    # --------------------------------------------------------
    # Массовое заполнение буфера (при ротации карусели)
    # --------------------------------------------------------
    def set_data(self, data_list):
        if not data_list:
            if self.mode == 'single':
                self.raw_data.clear()
                self.plot_data.setData([], [])
                self.scatter.setData([], [])
            else:
                self.dual_t.clear()
                self.dual_h.clear()
                self.dual_z.clear()
                self.dual_env_h.clear()
                self.dual_env_z.clear()
                self.p_marker_times.clear()
                self.s_marker_times.clear()
                self.curve_p.setData([], [])
                self.curve_z.setData([], [])
                self.p_markers.setData([], [])
                self.s_markers.setData([], [])
            self._dirty = True
            self._refresh()
            return

        first = data_list[0]
        is_carousel = (isinstance(first, (list, tuple, np.ndarray))
                       and len(first) >= 5)

        if is_carousel:
            if self.mode != 'dual':
                self.mode = 'dual'
                self.plot_data.setVisible(False)
                self.scatter.setVisible(False)
                self.curve_p.setVisible(True)
                self.curve_z.setVisible(True)
                self.raw_data.clear()
            self.dual_t.clear()
            self.dual_h.clear()
            self.dual_z.clear()
            self.dual_env_h.clear()
            self.dual_env_z.clear()
            self.p_marker_times.clear()
            self.s_marker_times.clear()
            for item in data_list:
                if len(item) >= 5:
                    try:
                        t = float(item[0])
                        h = float(item[1])
                        z = float(item[2])
                        eh = float(item[3])
                        ez = float(item[4])
                        self.dual_t.append(t)
                        self.dual_h.append(h)
                        self.dual_z.append(z)
                        self.dual_env_h.append(eh)
                        self.dual_env_z.append(ez)
                    except (ValueError, TypeError):
                        continue
            self.plotItem.setXRange(0, SPS, padding=0)
            self._dirty = True
            self._refresh()
        else:
            if self.mode != 'single':
                self.mode = 'single'
                self.plot_data.setVisible(True)
                self.scatter.setVisible(True)
                self.curve_p.setVisible(False)
                self.curve_z.setVisible(False)
                self.dual_t.clear()
                self.dual_h.clear()
                self.dual_z.clear()
                self.dual_env_h.clear()
                self.dual_env_z.clear()
            self.raw_data.clear()
            for val in data_list:
                if not np.isnan(val) and not np.isinf(val):
                    try:
                        self.raw_data.append(float(val))
                    except (ValueError, TypeError):
                        continue
            self._dirty = True
            self._refresh()

    # --------------------------------------------------------
    # Точка входа для карусели (FULL и CURVE)
    # --------------------------------------------------------
    def update_carousel(self, t, h, z, env_h, env_z):
        """v11.0.1: единая точка входа для карусели (FULL и CURVE)."""
        if self.mode != 'dual':
            self.mode = 'dual'
            self.plot_data.setVisible(False)
            self.scatter.setVisible(False)
            self.curve_p.setVisible(True)
            self.curve_z.setVisible(True)
            self.raw_data.clear()
        self.dual_t.append(float(t))
        self.dual_h.append(float(h))
        self.dual_z.append(float(z))
        self.dual_env_h.append(float(env_h))
        self.dual_env_z.append(float(env_z))
        self._dirty = True

    # --------------------------------------------------------
    # Маркеры P/S
    # --------------------------------------------------------
    def add_p_marker(self, t):
        self.p_marker_times.append(float(t))
        self._cleanup_markers(float(t))

    def add_s_marker(self, t):
        self.s_marker_times.append(float(t))
        self._cleanup_markers(float(t))

    def _cleanup_markers(self, current_t):
        cutoff = current_t - (SPS / SR) - 5.0
        self.p_marker_times = [tm for tm in self.p_marker_times if tm > cutoff]
        self.s_marker_times = [tm for tm in self.s_marker_times if tm > cutoff]

    # --------------------------------------------------------
    # Перерисовка
    # --------------------------------------------------------
    def _refresh(self):
        if not self._dirty:
            return
        self._dirty = False
        if self.mode == 'single':
            self._refresh_single()
        else:
            self._refresh_dual()

    def _refresh_single(self):
        data = list(self.raw_data)
        if not data:
            self.plot_data.setData([], [])
            self.scatter.setData([], [])
            return
        y_arr = np.array(data, dtype=np.float32)
        if len(y_arr) > 1:
            y_arr -= np.mean(y_arr)
        x_arr = np.arange(len(y_arr), dtype=np.float32)
        self.plot_data.setData(x_arr, y_arr)
        self.scatter.setData([len(y_arr) - 1], [y_arr[-1]])
        n = len(y_arr)
        if n < SPS:
            self.plotItem.setXRange(0, SPS, padding=0)
        else:
            self.plotItem.setXRange(n - SPS, n, padding=0)
        self.plotItem.setYRange(-SENS, SENS, padding=0)

    def _refresh_dual(self):
        n = len(self.dual_t)
        if n == 0:
            self.curve_p.setData([], [])
            self.curve_z.setData([], [])
            self.p_markers.setData([], [])
            self.s_markers.setData([], [])
            return

        x_arr = np.arange(n, dtype=np.float32)
        mode = getattr(config, 'CAROUSEL_DRAW_MODE', 'FULL')

        if mode == 'CURVE':
            y_h = np.array(self.dual_env_h, dtype=np.float32)
            y_z = np.array(self.dual_env_z, dtype=np.float32)
        else:  # FULL
            y_h = np.array(self.dual_h, dtype=np.float32)
            y_z = np.array(self.dual_z, dtype=np.float32)

        self.curve_p.setData(x_arr, y_h)
        self.curve_z.setData(x_arr, y_z)

        vb = self.plotItem.getViewBox()
        vb.enableAutoRange(axis='y', enable=False)
        vb.setYRange(-SENS, SENS, padding=0)

        if n < SPS:
            self.plotItem.setXRange(0, SPS, padding=0)
        else:
            self.plotItem.setXRange(n - SPS, n, padding=0)

        # --- Маркеры P/S на дне графика ---
        t_arr = np.array(self.dual_t, dtype=np.float64)
        t_min = float(t_arr[0])
        t_max = float(t_arr[-1])
        marker_pad = SENS * 0.05
        y_p = -SENS + marker_pad
        y_s = -SENS + marker_pad * 2

        p_vis = [tm for tm in self.p_marker_times if t_min <= tm <= t_max]
        s_vis = [tm for tm in self.s_marker_times if t_min <= tm <= t_max]

        if p_vis:
            idx_p = np.searchsorted(t_arr, p_vis, side='left')
            idx_p = np.clip(idx_p, 0, n - 1)
            self.p_markers.setData(idx_p.astype(np.float64),
                                   np.full(len(idx_p), y_p, dtype=np.float64))
        else:
            self.p_markers.setData([], [])

        if s_vis:
            idx_s = np.searchsorted(t_arr, s_vis, side='left')
            idx_s = np.clip(idx_s, 0, n - 1)
            self.s_markers.setData(idx_s.astype(np.float64),
                                   np.full(len(idx_s), y_s, dtype=np.float64))
        else:
            self.s_markers.setData([], [])