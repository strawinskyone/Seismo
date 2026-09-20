"""
processor.py v9.6.2
Real-time детекция P-волны по каналу Z.
- Работа в ВОЛЬТАХ (raw LSB * ADC_SCALE_V).
- Один детектор: STA/LTA (рекурсивный).
- Абсолютные пороги из config (в вольтах).
- БЕЗ интегрирования: работаем с СЫРЫМ Z (скорость).
  Интегрирование перенесено в heavy_worker (offline, окно 9-15 с).
"""
import time
import numpy as np
from scipy import signal
from scipy.signal import butter, sosfilt
import logging

import config

logger = logging.getLogger('seismic')
flush_logger = logging.getLogger('seismic.flush')

class EventTracker:
    def __init__(self, onset_idx, onset_time, p_idx, p_time_abs,
                 sample_rate, scope_id=0):
        self.state = 'P_RISING'
        self.onset_idx = onset_idx
        self.onset_time = onset_time
        self.p_idx = p_idx
        self.p_time_abs = p_time_abs
        self.sample_rate = sample_rate
        self.scope_id = scope_id
        self.p_refined = False
        self.p_peak = 0.0
        self.p_peak_idx = onset_idx
        self.p_peak_time = onset_time
        self.p_end_idx = None
        self.p_end_time = None
        self.flush_delay_logged = False
       
    def update(self, current_idx, current_time, abs_z, adaptive_end_thr):
        if self.state == 'P_RISING':
            if abs_z > self.p_peak:
                self.p_peak = abs_z
                self.p_peak_idx = current_idx
                self.p_peak_time = current_time
            p_duration = current_time - self.onset_time
            if p_duration >= config.P_MIN_DURATION_SEC:
                drop_from_peak = abs_z < self.p_peak * 0.90
                after_peak = current_time > self.p_peak_time + 0.1
                if (drop_from_peak and after_peak):
                    self.state = 'P_PEAKED'
                elif p_duration >= config.P_MAX_DURATION_SEC * 0.5:
                    self.state = 'P_PEAKED'
        elif self.state == 'P_PEAKED':
            p_duration = current_time - self.onset_time
            env_below = abs_z < adaptive_end_thr
            timeout = p_duration >= config.P_MAX_DURATION_SEC
            if env_below or timeout:
                self.p_end_idx = current_idx
                self.p_end_time = current_time
                self.state = 'P_ENDED'
                reason = 'timeout' if timeout else 'amplitude_decay'
                logger.info(
                    f"[TRACKER] P_END @ {current_time:.3f} "
                    f"(reason={reason}, peak={self.p_peak:.5f}V, "
                    f"thr={adaptive_end_thr:.5f}V, dur={p_duration:.2f}s)"
                )
        elif self.state in ('P_ENDED', 'CLOSED'):
            pass

class SeismicProcessor:
    def __init__(self, event_queue=None, p_onset_callback=None, scope_id_callback=None):
        self.sample_rate = config.SAMPLE_RATE
        self.dt = 1.0 / config.SAMPLE_RATE
        self.warmup_until = time.time() + config.ADC_WARMUP_SEC
        self.buffer_size = config.BUFFER_SIZE

        self.buf_z = np.zeros(self.buffer_size, dtype=np.float64)
        self.buf_n = np.zeros(self.buffer_size, dtype=np.float64)
        self.buf_e = np.zeros(self.buffer_size, dtype=np.float64)
        self.buf_t = np.zeros(self.buffer_size, dtype=np.float64)
        self.buf_idx = 0
        self.buf_filled = False

        # DC removal (EMA, tau = 10 с)
        self.dc_z = 0.0
        self.dc_n = 0.0
        self.dc_e = 0.0
        self.dc_alpha = 1.0 / (10.0 * self.sample_rate)

        # === STA/LTA для ДЕТЕКЦИИ P ===
        # v9.6.15: alpha = 1 - exp(-1/N) — точная формула (как в obspy).
        self.sta_val = 0.0
        self.lta_val = 0.0
        self.sta_alpha = 1.0 - np.exp(-1.0 / (config.P_STA_SEC * self.sample_rate))
        self.lta_alpha = 1.0 - np.exp(-1.0 / (config.P_LTA_SEC * self.sample_rate))
        self.sta_lta_ratio = 0.0
        self.sta_lta_triggered = False
        self.last_sta_lta_onset_t = 0.0
        self.sta_lta_reset_timeout = 30.0   # принудительный сброс через N с

        # v9.6.15: медианный LTA для P-детекции (окно из H_DISPLAY — короткое).
        # НЕ путать с S_PICKER_LTA_SEC (то для heavy_worker).
        self.lta_ring = np.zeros(config.H_DISPLAY_LTA_N, dtype=np.float64)
        self.lta_ring_idx = 0
        self.lta_ring_filled = False
        self.lta_median_counter = 0
        self.lta_median_interval = max(1, config.SAMPLE_RATE // 10)  # 10 раз в секунду
        self.lta_median_val = 0.0

        # v9.6.15: инициализация STA/LTA первым сэмплом + "пол" для LTA.
        self._sta_lta_initialized = False
        self.lta_floor = config.P_DETECT_ABS_MIN_V * 0.5   # 25 мВ при пороге 50 мВ

        # === Кольцо производной Z ===
        self.diff_n = config.P_DETECT_DIFF_N
        self.diff_ring = np.zeros(self.diff_n, dtype=np.float64)
        self.diff_idx = 0

        # === Трекеры и очереди ===
        self.active_trackers = []
        self.max_trackers = config.MAX_TRACKERS
        self.recent_events = []
        self.samples_since_heavy = 0
        self.heavy_interval = config.HEAVY_PROCESS_INTERVAL_N
        self.event_queue = event_queue
        self.p_onset_callback = p_onset_callback
        self.scope_id_callback = scope_id_callback
        self.latest_t = 0.0

        # === Счётчики диагностики ===
        self.onsets_total = 0
        self.onsets_rejected_amp = 0
        self.onsets_rejected_deriv = 0
        self.onsets_rejected_guard = 0
        self.onsets_rejected_max = 0
        self.samples_since_diag = 0
        self.diag_abs_max = 0.0
        self.diag_d_max = 0.0
        self.diag_ratio_max = 0.0
        self.diag_interval = max(1, int(config.PROC_DIAG_INTERVAL_SEC * config.SAMPLE_RATE))
        self.detect_warmup_until = time.time() + config.P_LTA_SEC * 5

        # === Огибающая H для ВИЗУАЛИЗАЦИИ (карусель) ===
        # v9.6.16: линейный индикатор вместо STA/LTA — не «дышит», показывает амплитуду.
        self.env_h_val = 0.0
        self.env_h_alpha = 1.0 - np.exp(-1.0 / (0.3 * self.sample_rate))  # окно 0.3 с
        self.h_ratio_callback = None

        # v9.6.15: полосовой фильтр N/E для h_abs (realtime sosfilt).
        # Убирает наводку 7.8 Гц и низкочастотный дрейф — как в plot_h.py.
        self.sos_h = signal.butter(
            4,
            [config.S_FILTER_FREQMIN, config.S_FILTER_FREQMAX],
            btype='band',
            fs=self.sample_rate,
            output='sos'
        )
        # Состояния фильтров (zi) — для непрерывной работы sosfilt.
        self.zi_n = np.zeros((self.sos_h.shape[0], 2))
        self.zi_e = np.zeros((self.sos_h.shape[0], 2))

        logger.info(
            f"[PROCESSOR {config.VERSION}] Init: SR={config.SAMPLE_RATE}, "
            f"buf={self.buffer_size} samp ({self.buffer_size/config.SAMPLE_RATE:.1f}s), "
            f"max_trackers={config.MAX_TRACKERS}, "
            f"signal=VOLTS (raw LSB * {config.ADC_SCALE_V*1e6:.2f} uV/LSB), "
            f"abs_min_V={config.P_DETECT_ABS_MIN_V}, "
            f"abs_min_d={config.P_DETECT_ABS_MIN_D}, "
            f"P_STA/LTA={config.P_STA_SEC}/{config.P_LTA_SEC}s, "
            f"H_STA/LTA={config.H_DISPLAY_STA_SEC}/{config.H_DISPLAY_LTA_SEC}s, "
            f"lta_floor={self.lta_floor:.5f}V"
        )

    def _append(self, x, y, z, t):
        idx = self.buf_idx
        self.buf_n[idx] = float(x)
        self.buf_e[idx] = float(y)
        self.buf_z[idx] = float(z)
        self.buf_t[idx] = float(t)
        self.buf_idx = (idx + 1) % self.buffer_size
        if self.buf_idx == 0:
            self.buf_filled = True

    def _init_sta_lta_if_needed(self, abs_z):
        """
        v9.6.12: Первый валидный |Z| инициализирует STA и LTA.
        Без этого EMA стартует с 0 и ~20 секунд "догоняет" реальный шум,
        что даёт ложные триггеры на старте.
        """
        if not self._sta_lta_initialized:
            self.lta_val = max(abs_z, self.lta_floor)
            self.sta_val = abs_z
            self._sta_lta_initialized = True
            logger.info(
                f"[PROC] STA/LTA initialized: "
                f"sta={self.sta_val:.5f}V, lta={self.lta_val:.5f}V, "
                f"floor={self.lta_floor:.5f}V"
            )

    def _get_chronological(self):
        if not self.buf_filled:
            return (self.buf_n[:self.buf_idx].copy(),
                    self.buf_e[:self.buf_idx].copy(),
                    self.buf_z[:self.buf_idx].copy(),
                    self.buf_t[:self.buf_idx].copy())
        n = np.concatenate((self.buf_n[self.buf_idx:], self.buf_n[:self.buf_idx]))
        e = np.concatenate((self.buf_e[self.buf_idx:], self.buf_e[:self.buf_idx]))
        z = np.concatenate((self.buf_z[self.buf_idx:], self.buf_z[:self.buf_idx]))
        t = np.concatenate((self.buf_t[self.buf_idx:], self.buf_t[:self.buf_idx]))
        return n, e, z, t

    def _current_chronological_idx(self):
        if not self.buf_filled:
            return max(0, self.buf_idx - 1)
        return self.buffer_size - 1

    def _process_sample(self, n_raw, e_raw, z_raw, ts):
        # 1. DC removal
        self.dc_n += self.dc_alpha * (n_raw - self.dc_n)
        self.dc_e += self.dc_alpha * (e_raw - self.dc_e)
        self.dc_z += self.dc_alpha * (z_raw - self.dc_z)

        n_cent = n_raw - self.dc_n
        e_cent = e_raw - self.dc_e
        z_cent = z_raw - self.dc_z

        # 2. В буфер
        self._append(n_cent, e_cent, z_cent, ts)
        self.latest_t = float(ts)

        # 3. STA/LTA на |Z|
        abs_z = abs(z_cent)

        self._init_sta_lta_if_needed(abs_z)

        # v9.6.15: полосовой фильтр N/E (realtime sosfilt, с сохранением состояния)
        n_filt, self.zi_n = signal.sosfilt(self.sos_h, [n_cent], zi=self.zi_n)
        e_filt, self.zi_e = signal.sosfilt(self.sos_h, [e_cent], zi=self.zi_e)
        n_filt = n_filt[0]
        e_filt = e_filt[0]

        # H = sqrt(N² + E²) — как в plot_h.py
        h_abs = np.sqrt(n_filt**2 + e_filt**2)

        # v9.6.16: огибающая H — линейный индикатор для оператора.
        self.env_h_val += self.env_h_alpha * (h_abs - self.env_h_val)

        # --- LTA для детекции P ---
        self.lta_ring[self.lta_ring_idx] = abs_z
        self.lta_ring_idx = (self.lta_ring_idx + 1) % len(self.lta_ring)
        if self.lta_ring_idx == 0:
            self.lta_ring_filled = True

        self.lta_median_counter += 1
        if self.lta_median_counter >= self.lta_median_interval:
            self.lta_median_counter = 0
            if self.lta_ring_filled:
                med = float(np.median(self.lta_ring))
            else:
                med = float(np.median(self.lta_ring[:self.lta_ring_idx + 1]))
            if med < self.lta_floor:
                med = self.lta_floor
            self.lta_val = med

        self.sta_val += self.sta_alpha * (abs_z - self.sta_val)
        self.sta_lta_ratio = self.sta_val / self.lta_val

        # 4. Производная Z
        self.diff_ring[self.diff_idx] = z_cent
        self.diff_idx = (self.diff_idx + 1) % self.diff_n
        z_prev = self.diff_ring[self.diff_idx]
        d_z = abs(z_cent - z_prev)
        # v9.6.x: накопление диагностических максимумов
        if abs_z > self.diag_abs_max:
            self.diag_abs_max = abs_z
        if d_z > self.diag_d_max:
            self.diag_d_max = d_z
        if self.sta_lta_ratio > self.diag_ratio_max:
            self.diag_ratio_max = self.sta_lta_ratio

        # 5. Обновление трекеров
        current_idx = self._current_chronological_idx()
        self._update_trackers(current_idx, float(ts), abs_z)

        # 6. Детекция P
        self._check_sta_lta_onset(current_idx, float(ts), abs_z, d_z)

        # 7. Тяжёлая обработка — по интервалу
        self.samples_since_heavy += 1
        if self.samples_since_heavy >= self.heavy_interval:
            self.samples_since_heavy = 0
            self._refine_active_onsets()
            self._flush_ended_trackers()

        # v9.6.x: диагностика раз в PROC_DIAG_INTERVAL_SEC секунд
        if config.PROC_DIAG_INTERVAL_SEC > 0:
            self.samples_since_diag += 1
            if self.samples_since_diag >= self.diag_interval:
                # Печатать только если сигнал был интересным
                interesting = (
                    self.diag_abs_max > config.P_DETECT_ABS_MIN_V * 0.5 or
                    self.diag_ratio_max > 1.3
                )
                if interesting:
                    logger.info(
                        f"[PROC-DIAG] "
                        f"|Z|max={self.diag_abs_max:.5f}V, "
                        f"dZ_max={self.diag_d_max:.5f}V, "
                        f"ratio_max={self.diag_ratio_max:.2f}, "
                        f"thr_V={config.P_DETECT_ABS_MIN_V}, "
                        f"thr_d={config.P_DETECT_ABS_MIN_D}, "
                        f"thr_ratio={config.P_TRIGGER_RATIO}, "
                        f"trig={self.sta_lta_triggered}, "
                        f"rejected: amp={self.onsets_rejected_amp}, "
                        f"deriv={self.onsets_rejected_deriv}, "
                        f"guard={self.onsets_rejected_guard}, "
                        f"lta={self.lta_val:.5f}V, "
                        f"sta={self.sta_val:.5f}V"
                    )
                self.samples_since_diag = 0
                self.diag_abs_max = 0.0
                self.diag_d_max = 0.0
                self.diag_ratio_max = 0.0

        # v9.6.x: принудительный сброс триггера по таймауту
        if self.sta_lta_triggered:
            if (time.time() - self.last_sta_lta_onset_t) > self.sta_lta_reset_timeout:
                self.sta_lta_triggered = False
                logger.info("[PROC] STA/LTA trigger force reset (timeout)")

        if self.h_ratio_callback:
            try:
                self.h_ratio_callback(self.env_h_val)
            except Exception:
                pass			

        return z_cent


    def _check_sta_lta_onset(self, current_idx, current_t, abs_z, d_z):
        in_warmup = time.time() < self.warmup_until
        if in_warmup:
            return False

        if time.time() < self.detect_warmup_until:
            return False

        if abs_z < config.P_DETECT_ABS_MIN_V:
            self.onsets_rejected_amp += 1
            return False

        if d_z < config.P_DETECT_ABS_MIN_D:
            self.onsets_rejected_deriv += 1
            return False

        if self.sta_lta_ratio < config.P_TRIGGER_RATIO:
            if self.sta_lta_triggered and self.sta_lta_ratio < config.P_DETRIGGER_RATIO:
                self.sta_lta_triggered = False
            return False
        if self.sta_lta_triggered:
            return False

        debounce_sec = config.EVENT_DEBOUNCE_MS / 1000.0
        for closed_t, _ in self.recent_events:
            if abs(current_t - closed_t) < debounce_sec:
                self.onsets_rejected_guard += 1
                return False

        for trk in self.active_trackers:
            if trk.state in ('P_RISING', 'P_PEAKED'):
                if abs(trk.onset_time - current_t) < config.ONSET_GUARD_SEC:
                    self.onsets_rejected_guard += 1
                    return False

        if len(self.active_trackers) >= self.max_trackers:
            oldest = None
            for trk in self.active_trackers:
                if trk.state == 'P_ENDED':
                    oldest = trk
                    break
            if oldest is None:
                for trk in self.active_trackers:
                    if trk.state == 'P_PEAKED':
                        oldest = trk
                        break
            if oldest is not None:
                self.active_trackers.remove(oldest)
            else:
                self.onsets_rejected_max += 1
                return False

        scope_id = 0
        if self.scope_id_callback:
            try:
                scope_id = self.scope_id_callback()
            except Exception:
                pass

        tracker = EventTracker(
            onset_idx=current_idx, onset_time=current_t,
            p_idx=current_idx, p_time_abs=current_t,
            sample_rate=self.sample_rate, scope_id=scope_id
        )
        self.active_trackers.append(tracker)
        self.sta_lta_triggered = True
        self.last_sta_lta_onset_t = time.time()   # v9.6.x
        self.onsets_total += 1

        logger.info(
            f"[ONSET] New tracker @ {current_t:.3f} "
            f"(|Z|={abs_z:.5f}V, dZ={d_z:.5f}V, ratio={self.sta_lta_ratio:.2f}, "
            f"sta={self.sta_val:.5f}, lta={self.lta_val:.5f}, scope={scope_id})"
        )
        if self.p_onset_callback:
            self.p_onset_callback(current_t)
        return True

    def _adaptive_p_end_threshold(self, tracker):
        rel_thr = tracker.p_peak * config.P_END_REL_BASE
        return max(config.P_END_ABS_V, rel_thr)

    def _update_trackers(self, current_idx, current_t, abs_z):
        for tracker in self.active_trackers:
            if tracker.state in ('P_RISING', 'P_PEAKED'):
                thr = self._adaptive_p_end_threshold(tracker)
                prev_state = tracker.state
                tracker.update(current_idx, current_t, abs_z, thr)
                # v9.6.x: фиксируем P_END в recent_events СРАЗУ,
                # а не при flush snapshot. Иначе debounce не работает,
                # и один сигнал даёт цепочку трекеров.
                if prev_state != 'P_ENDED' and tracker.state == 'P_ENDED':
                    self.recent_events.append((tracker.p_end_time, tracker.p_time_abs))

    def _refine_active_onsets(self):
        for tracker in self.active_trackers:
            if tracker.state == 'P_RISING' and not tracker.p_refined:
                p_time_abs, p_idx = self._find_p_onset(tracker.onset_idx)
                if p_idx >= 0:
                    tracker.p_idx = p_idx
                    tracker.p_time_abs = p_time_abs
                    tracker.p_refined = True
                    logger.info(
                        f"[REFINE] P refined @ {p_time_abs:.3f} "
                        f"(was {tracker.onset_time:.3f}, "
                        f"delta={tracker.onset_time - p_time_abs:.3f}s)"
                    )

    def _find_p_onset(self, current_idx):
        _, _, z_chron, t_chron = self._get_chronological()
        if len(z_chron) < 100:
            return (float(t_chron[0]) if len(t_chron) > 0 else 0.0), 0

        analysis_sec = config.P_LTA_SEC + 2.0
        analysis_samples = int(analysis_sec * self.sample_rate)
        a = max(0, current_idx - analysis_samples + 1)
        z_win = z_chron[a:current_idx + 1].copy()
        if len(z_win) < 200:
            return float(t_chron[a]), a

        try:
            sos = signal.butter(4, [config.FILTER_FREQMIN, config.FILTER_FREQMAX],
                                btype='band', fs=self.sample_rate, output='sos')
            z_filt = signal.sosfiltfilt(sos, z_win)
        except Exception:
            z_filt = z_win

        data = np.abs(z_filt)
        sta_n = config.P_STA_N
        lta_n = config.P_LTA_N
        if len(data) < lta_n + sta_n:
            return float(t_chron[a]), a

        sta = np.convolve(data, np.ones(sta_n) / sta_n, mode='valid')
        lta = np.convolve(data, np.ones(lta_n) / lta_n, mode='valid')
        offset = lta_n - sta_n
        if offset < 0:
            offset = 0
        if offset >= len(sta):
            return float(t_chron[a]), a

        sta_aligned = sta[offset:]
        min_len = min(len(sta_aligned), len(lta))
        sta_aligned = sta_aligned[:min_len]
        lta = lta[:min_len]
        ratio = sta_aligned / (lta + 1e-12)
        if len(ratio) == 0:
            return float(t_chron[a]), a

        rel_idx = len(ratio) - 1
        while rel_idx >= 0 and ratio[rel_idx] >= config.P_DETRIGGER_RATIO:
            rel_idx -= 1
        onset_rel = rel_idx + 1
        onset_rel = max(0, min(onset_rel, len(ratio) - 1))
        onset_idx_win = onset_rel + lta_n - 1
        onset_idx_win = max(0, min(onset_idx_win, len(data) - 1))
        onset_idx_global = a + onset_idx_win
        onset_idx_global = max(0, min(onset_idx_global, len(z_chron) - 1))
        onset_idx_global = min(onset_idx_global, current_idx)
        return float(t_chron[onset_idx_global]), onset_idx_global

    def get_snapshot(self, p_time_abs, p_end_idx, tracker):
        n, e, z, t = self._get_chronological()
        if len(t) == 0:
            return None
        p_idx = int((p_time_abs - t[0]) * self.sample_rate)
        p_idx = max(0, min(p_idx, len(t) - 1))
        pre_samples = config.SNAPSHOT_PRE_P_N
        post_samples = config.SNAPSHOT_POST_P_N
        a = max(0, p_idx - pre_samples)
        b = min(len(t), p_idx + post_samples)
        if tracker.p_end_time is not None:
            p_end_snapshot = int((tracker.p_end_time - t[0]) * self.sample_rate) - a
            p_end_snapshot = max(0, min(p_end_snapshot, b - a - 1))
        else:
            p_end_snapshot = None
        return {
            'n': n[a:b].copy(), 'e': e[a:b].copy(), 'z': z[a:b].copy(), 't': t[a:b].copy(),
            'p_idx': p_idx - a, 'p_end_idx': p_end_snapshot,
            'sample_rate': self.sample_rate, 'scope_id': tracker.scope_id,
        }

    def _flush_ended_trackers(self):
        ready = []
        remaining = []
        for tracker in self.active_trackers:
            if tracker.state == 'P_ENDED':
                required_t = tracker.p_time_abs + config.SNAPSHOT_POST_P_SEC
                if self.latest_t < required_t:
                    if not tracker.flush_delay_logged:
                        flush_logger.debug(
                            f"[FLUSH-DELAY] Tracker P@{tracker.p_time_abs:.3f} "
                            f"отложен до {required_t:.3f} (latest={self.latest_t:.3f})"
                        )
                        tracker.flush_delay_logged = True
                    remaining.append(tracker)
                    continue
                snapshot = self.get_snapshot(tracker.p_time_abs, tracker.p_end_idx, tracker)
                if snapshot is not None:
                    ready.append(snapshot)
                tracker.state = 'CLOSED'
                self.recent_events.append((tracker.p_end_time, tracker.p_time_abs))
            else:
                remaining.append(tracker)
        self.active_trackers = remaining

        cutoff = time.time() - (config.EVENT_DEBOUNCE_MS / 1000.0) * 2
        self.recent_events = [(ct, pt) for ct, pt in self.recent_events if ct > cutoff]

        for snapshot in ready:
            if self.event_queue is not None:
                try:
                    self.event_queue.put(snapshot, block=False)
                    p_t = snapshot['t'][snapshot['p_idx']]
                    pe_t = snapshot['t'][snapshot['p_end_idx']]
                    logger.info(
                        f"[DISPATCHER] Snapshot sent "
                        f"(P @ {p_t:.3f}, P_end @ {pe_t:.3f}, "
                        f"len={len(snapshot['z'])}, scope={snapshot['scope_id']}, "
                        f"trackers_active={len(self.active_trackers)}, "
                        f"buf_filled={self.buf_filled})"
                    )
                except Exception:
                    logger.warning("[DISPATCHER] QUEUE FULL, event dropped")

    def process_batch(self, batch_raws, batch_timestamps):
        for raw, ts in zip(batch_raws, batch_timestamps):
            if len(raw) < 3:
                continue
            n_raw = float(raw[0]) * config.ADC_SCALE_V * config.GAIN_CORRECTION_N
            e_raw = float(raw[1]) * config.ADC_SCALE_V * config.GAIN_CORRECTION_E
            z_raw = float(raw[2]) * config.ADC_SCALE_V * config.GAIN_CORRECTION_Z
            self._process_sample(n_raw, e_raw, z_raw, float(ts))

    def process_single(self, raw, timestamp):
        if len(raw) < 3:
            return None
        n_raw = float(raw[0]) * config.ADC_SCALE_V * config.GAIN_CORRECTION_N
        e_raw = float(raw[1]) * config.ADC_SCALE_V * config.GAIN_CORRECTION_E
        z_raw = float(raw[2]) * config.ADC_SCALE_V * config.GAIN_CORRECTION_Z
        self._process_sample(n_raw, e_raw, z_raw, float(timestamp))
        return None

    def get_onset_stats(self):
        return {
            'total_accepted': self.onsets_total,
            'rejected_by_amplitude': self.onsets_rejected_amp,
            'rejected_by_derivative': self.onsets_rejected_deriv,
            'rejected_by_guard': self.onsets_rejected_guard,
            'rejected_by_max_trackers': self.onsets_rejected_max,
        }