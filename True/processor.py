import os
import time
import numpy as np
from obspy.core import UTCDateTime
from scipy import signal
import logging

import config

logger = logging.getLogger('seismic')


class EventTracker:
    """
    Один трекер = одна P-волна.
    Состояния: P_RISING → P_PEAKED → P_ENDED → CLOSED
    """

    def __init__(self, onset_idx, onset_time, p_idx, p_time_abs,
                 sample_rate, scope_id=0):
        self.state = 'P_RISING'
        self.onset_idx = onset_idx
        self.onset_time = onset_time
        self.p_idx = p_idx
        self.p_time_abs = p_time_abs

        self.sample_rate = sample_rate
        self.scope_id = scope_id

        self.p_peak = 0.0
        self.p_peak_idx = onset_idx
        self.p_peak_time = onset_time

        self.p_end_idx = None
        self.p_end_time = None

    def update(self, current_idx, current_time, abs_z, global_env,
               adaptive_end_thr):
        """Обновление на каждом сэмпле."""

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

            env_below = global_env < adaptive_end_thr
            timeout = p_duration >= config.P_MAX_DURATION_SEC

            if env_below or timeout:
                self.p_end_idx = current_idx
                self.p_end_time = current_time
                self.state = 'P_ENDED'
                reason = 'timeout' if timeout else 'envelope_decay'
                logger.info(
                    f"[TRACKER] P_END @ {current_time:.3f} "
                    f"(reason={reason}, peak={self.p_peak:.1f}, "
                    f"thr={adaptive_end_thr:.1f}, dur={p_duration:.2f}s)"
                )

        elif self.state in ('P_ENDED', 'CLOSED'):
            pass


class SeismicProcessor:
    """
    v9.2 — Multi-event adaptive envelope tracking + CPU fix.
    Heavy STA/LTA throttled to 0.2s. Noise stats every 2.0s.
    Добавлен scope_id для привязки событий к панелям карусели.
    """

    def __init__(self, event_queue=None, p_onset_callback=None, scope_id_callback=None):
        for key, value in vars(config).items():
            if not key.startswith('__'):
                globals()[key] = value

        self.sample_rate = SAMPLE_RATE
        self.dt = 1.0 / SAMPLE_RATE
        self.buffer_size = int(
            max(80, SNAPSHOT_PRE_P_SEC + SNAPSHOT_POST_P_SEC + 20) * SAMPLE_RATE
        )

        # Кольцевые буферы
        self.buf_z = np.zeros(self.buffer_size, dtype=np.float64)
        self.buf_n = np.zeros(self.buffer_size, dtype=np.float64)
        self.buf_e = np.zeros(self.buffer_size, dtype=np.float64)
        self.buf_t = np.zeros(self.buffer_size, dtype=np.float64)
        self.buf_idx = 0
        self.buf_filled = False

        # Envelope: peak-hold скользящее окно
        env_win = max(1, int(ENVELOPE_WIN_SEC * self.sample_rate))
        self.env_abs_buf = np.zeros(env_win, dtype=np.float64)
        self.env_buf_idx = 0
        self.global_env = 0.0

        # История огибающей для noise estimation
        noise_len = int(NOISE_ESTIMATE_SEC * self.sample_rate)
        self.noise_env_history = np.zeros(noise_len, dtype=np.float64)
        self.noise_hist_idx = 0
        self.noise_hist_filled = False

        # История производных для adaptive MAD
        deriv_len = max(100, int(NOISE_ESTIMATE_SEC * self.sample_rate))
        self.deriv_history = np.zeros(deriv_len, dtype=np.float64)
        self.deriv_hist_idx = 0

        self.noise_floor = (EVENT_THRESHOLD_MV / 1000.0) / ADC_SCALE_V
        self.noise_rms = self.noise_floor * 0.5
        self.threshold_derivative = self.noise_floor * 10.0
        self.last_noise_update = 0.0

        # История env для delayed access (производная)
        delay = max(1, int(0.1 * self.sample_rate))
        self.env_history = np.zeros(delay * 2, dtype=np.float64)
        self.env_hist_idx = 0
        self.env_hist_count = 0

        # Multi-event state
        self.active_trackers = []
        self.max_trackers = MAX_TRACKERS
        self.recent_events = []

        self.samples_since_heavy = 0
        self.heavy_interval = HEAVY_PROCESS_INTERVAL_SEC
        self.event_queue = event_queue
        self.p_onset_callback = p_onset_callback
        self.scope_id_callback = scope_id_callback

        if DAQ_CPU_CORES:
            try:
                os.sched_setaffinity(0, DAQ_CPU_CORES)
                logger.info(f"[PROCESSOR {VERSION}] CPU affinity {DAQ_CPU_CORES}")
            except Exception as e:
                logger.warning(f"[PROCESSOR {VERSION}] CPU affinity failed: {e}")

        logger.info(
            f"[PROCESSOR {VERSION}] Init: SR={SAMPLE_RATE}, "
            f"env_win={ENVELOPE_WIN_SEC}s, noise_est={NOISE_ESTIMATE_SEC}s, "
            f"max_trackers={MAX_TRACKERS}"
        )

    # ------------------------------------------------------------------ #
    #  Буферы
    # ------------------------------------------------------------------ #
    def _append(self, x, y, z, t):
        idx = self.buf_idx
        self.buf_n[idx] = float(x)
        self.buf_e[idx] = float(y)
        self.buf_z[idx] = float(z)
        self.buf_t[idx] = float(t)
        self.buf_idx = (idx + 1) % self.buffer_size
        if self.buf_idx == 0:
            self.buf_filled = True

    def _get_chronological(self):
        if not self.buf_filled:
            return (self.buf_n[:self.buf_idx].copy(),
                    self.buf_e[:self.buf_idx].copy(),
                    self.buf_z[:self.buf_idx].copy(),
                    self.buf_t[:self.buf_idx].copy())
        n = np.concatenate((self.buf_n[self.buf_idx:],
                            self.buf_n[:self.buf_idx]))
        e = np.concatenate((self.buf_e[self.buf_idx:],
                            self.buf_e[:self.buf_idx]))
        z = np.concatenate((self.buf_z[self.buf_idx:],
                            self.buf_z[:self.buf_idx]))
        t = np.concatenate((self.buf_t[self.buf_idx:],
                            self.buf_t[:self.buf_idx]))
        return n, e, z, t

    def _current_chronological_idx(self):
        if not self.buf_filled:
            return max(0, self.buf_idx - 1)
        return self.buffer_size - 1

    # ------------------------------------------------------------------ #
    #  Envelope & adaptive noise
    # ------------------------------------------------------------------ #
    def _update_envelope(self, abs_z):
        self.env_abs_buf[self.env_buf_idx] = abs_z
        self.env_buf_idx = (self.env_buf_idx + 1) % len(self.env_abs_buf)
        self.global_env = float(np.max(self.env_abs_buf))

        self.noise_env_history[self.noise_hist_idx] = self.global_env
        self.noise_hist_idx = (self.noise_hist_idx + 1) % len(self.noise_env_history)
        if self.noise_hist_idx == 0:
            self.noise_hist_filled = True

        self.env_history[self.env_hist_idx] = self.global_env
        self.env_hist_idx = (self.env_hist_idx + 1) % len(self.env_history)
        self.env_hist_count += 1

    def _get_delayed_envelope(self, delay_samples):
        if self.env_hist_count < delay_samples:
            return 0.0
        idx = (self.env_hist_idx - delay_samples) % len(self.env_history)
        return self.env_history[idx]

    def _update_noise_stats(self, current_t):
        # v9.2: интервал 2.0 сек — перцентиль на 12000 точек
        if current_t - self.last_noise_update < 2.0:
            return
        self.last_noise_update = current_t

        hist = (self.noise_env_history if self.noise_hist_filled
                else self.noise_env_history[:self.noise_hist_idx])
        if len(hist) > 100:
            self.noise_floor = float(np.percentile(hist, NOISE_PERCENTILE))
            quiet = hist[hist < self.noise_floor * 2.0]
            if len(quiet) > 10:
                self.noise_rms = float(
                    np.sqrt(np.mean((quiet - self.noise_floor) ** 2))
                )
            else:
                self.noise_rms = self.noise_floor * 0.3

        if self.deriv_hist_idx > 100:
            d_hist = self.deriv_history[:self.deriv_hist_idx]
            median_d = float(np.median(d_hist))
            mad = float(np.median(np.abs(d_hist - median_d))) * 1.4826
            self.threshold_derivative = max(
                self.noise_rms * 2.0,
                median_d + ONSET_DERIVATIVE_FACTOR * max(mad, self.noise_rms * 0.1)
            )

    def _adaptive_p_end_threshold(self, tracker):
        abs_thr = self.noise_floor * P_END_ABS_FACTOR
        snr = tracker.p_peak / max(self.noise_floor, 1e-12)
        rel_factor = P_END_REL_BASE + P_END_REL_SNR_FACTOR / np.sqrt(snr + 1.0)
        rel_factor = min(0.60, max(0.15, rel_factor))
        rel_thr = tracker.p_peak * rel_factor
        return max(abs_thr, rel_thr)

    # ------------------------------------------------------------------ #
    #  Onset detection
    # ------------------------------------------------------------------ #
    def _check_onset(self, current_idx, current_t, abs_z):
        if self.global_env < self.noise_floor * SNR_THRESHOLD:
            return False

        delay = max(1, int(0.1 * self.sample_rate))
        env_delayed = self._get_delayed_envelope(delay)
        d_env = (self.global_env - env_delayed) / (delay * self.dt)

        self.deriv_history[self.deriv_hist_idx] = d_env
        self.deriv_hist_idx = (self.deriv_hist_idx + 1) % len(self.deriv_history)

        if d_env < self.threshold_derivative:
            return False

        for trk in self.active_trackers:
            if trk.state in ('P_RISING', 'P_PEAKED'):
                if abs(trk.onset_time - current_t) < ONSET_GUARD_SEC:
                    return False

        debounce_sec = EVENT_DEBOUNCE_MS / 1000.0
        for closed_t, _ in self.recent_events:
            if abs(current_t - closed_t) < debounce_sec:
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
                return False

        p_time_abs, p_idx = self._find_p_onset(current_idx)
        if p_idx < 0:
            p_idx = current_idx
            p_time_abs = current_t

        scope_id = 0
        if self.scope_id_callback:
            try:
                scope_id = self.scope_id_callback()
            except Exception:
                pass

        tracker = EventTracker(
            onset_idx=current_idx,
            onset_time=current_t,
            p_idx=p_idx,
            p_time_abs=p_time_abs,
            sample_rate=self.sample_rate,
            scope_id=scope_id
        )
        self.active_trackers.append(tracker)
        logger.info(
            f"[ONSET] New tracker @ {p_time_abs:.3f} "
            f"(env={self.global_env:.1f}, d_env={d_env:.1f}, "
            f"noise_floor={self.noise_floor:.1f}, scope={scope_id})"
        )
        if self.p_onset_callback:
            self.p_onset_callback(p_time_abs)
        return True

    # ------------------------------------------------------------------ #
    #  STA/LTA onset refinement
    # ------------------------------------------------------------------ #
    def _find_p_onset(self, current_idx):
        _, _, z_chron, t_chron = self._get_chronological()
        if len(z_chron) < 100:
            return (float(t_chron[0]) if len(t_chron) > 0 else 0.0), 0

        analysis_sec = P_LTA_SEC + 2.0
        analysis_samples = int(analysis_sec * self.sample_rate)
        a = max(0, current_idx - analysis_samples + 1)
        z_win = z_chron[a:current_idx + 1].copy()
        if len(z_win) < 200:
            return float(t_chron[a]), a

        try:
            sos = signal.butter(4, [FILTER_FREQMIN, FILTER_FREQMAX],
                                btype='band', fs=self.sample_rate, output='sos')
            z_filt = signal.sosfiltfilt(sos, z_win)
        except Exception:
            z_filt = z_win

        data = np.abs(z_filt)

        sta_n = max(1, int(P_STA_SEC * self.sample_rate))
        lta_n = max(1, int(P_LTA_SEC * self.sample_rate))
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
        while rel_idx >= 0 and ratio[rel_idx] >= P_DETRIGGER_RATIO:
            rel_idx -= 1
        onset_rel = rel_idx + 1
        onset_rel = max(0, min(onset_rel, len(ratio) - 1))

        onset_idx_win = onset_rel + lta_n - 1
        onset_idx_win = max(0, min(onset_idx_win, len(data) - 1))
        onset_idx_global = a + onset_idx_win
        onset_idx_global = max(0, min(onset_idx_global, len(z_chron) - 1))
        onset_idx_global = min(onset_idx_global, current_idx)

        return float(t_chron[onset_idx_global]), onset_idx_global

    # ------------------------------------------------------------------ #
    #  Snapshot
    # ------------------------------------------------------------------ #
    def get_snapshot(self, p_time_abs, p_end_idx, tracker):
        n, e, z, t = self._get_chronological()
        if len(t) == 0:
            return None

        p_idx = int((p_time_abs - t[0]) * self.sample_rate)
        p_idx = max(0, min(p_idx, len(t) - 1))

        pre_samples = int(SNAPSHOT_PRE_P_SEC * self.sample_rate)
        post_samples = int(SNAPSHOT_POST_P_SEC * self.sample_rate)

        a = max(0, p_idx - pre_samples)
        b = min(len(t), p_idx + post_samples)

        p_end_snapshot = p_end_idx - a
        p_end_snapshot = max(0, min(p_end_snapshot, b - a - 1))

        return {
            'n': n[a:b].copy(),
            'e': e[a:b].copy(),
            'z': z[a:b].copy(),
            't': t[a:b].copy(),
            'p_idx': p_idx - a,
            'p_end_idx': p_end_snapshot,
            'sample_rate': self.sample_rate,
            'scope_id': tracker.scope_id,
        }

    # ------------------------------------------------------------------ #
    #  Tracker management
    # ------------------------------------------------------------------ #
    def _update_trackers(self, current_idx, current_t, abs_z):
        for tracker in self.active_trackers:
            if tracker.state in ('P_RISING', 'P_PEAKED'):
                adaptive_thr = self._adaptive_p_end_threshold(tracker)
                tracker.update(current_idx, current_t, abs_z,
                               self.global_env, adaptive_thr)

    def _flush_ended_trackers(self):
        ready = []
        remaining = []
        for tracker in self.active_trackers:
            if tracker.state == 'P_ENDED':
                snapshot = self.get_snapshot(tracker.p_time_abs,
                                             tracker.p_end_idx, tracker)
                if snapshot is not None:
                    ready.append(snapshot)
                tracker.state = 'CLOSED'
                self.recent_events.append(
                    (tracker.p_end_time, tracker.p_time_abs)
                )
            elif tracker.state == 'CLOSED':
                self.recent_events.append(
                    (tracker.p_end_time, tracker.p_time_abs)
                )
            else:
                remaining.append(tracker)

        self.active_trackers = remaining

        cutoff = time.time() - (EVENT_DEBOUNCE_MS / 1000.0) * 2
        self.recent_events = [(ct, pt) for ct, pt in self.recent_events
                              if ct > cutoff]

        for snapshot in ready:
            if self.event_queue is not None:
                try:
                    self.event_queue.put(snapshot, block=False)
                    p_t = snapshot['t'][snapshot['p_idx']]
                    pe_t = snapshot['t'][snapshot['p_end_idx']]
                    logger.info(
                        f"[DISPATCHER] Snapshot sent "
                        f"(P @ {p_t:.3f}, P_end @ {pe_t:.3f}, "
                        f"len={len(snapshot['z'])}, scope={snapshot['scope_id']})"
                    )
                except Exception:
                    logger.warning("[DISPATCHER] QUEUE FULL, event dropped")

    # ------------------------------------------------------------------ #
    #  Public API  —  v9.2
    # ------------------------------------------------------------------ #
    def process_batch(self, batch_raws, batch_timestamps):
        for raw, ts in zip(batch_raws, batch_timestamps):
            self.process_single(raw, ts)

    def process_single(self, raw, timestamp):
        if len(raw) < 3:
            return None
        self._append(raw[0], raw[1], raw[2], timestamp)

        current_t = float(timestamp)
        abs_z = abs(float(raw[2]))

        # === LIGHT: на каждый сэмпл (O(1), быстро) ===
        self._update_envelope(abs_z)
        self._update_noise_stats(current_t)
        current_idx = self._current_chronological_idx()
        self._update_trackers(current_idx, current_t, abs_z)

        # === HEAVY: throttle 0.2s (STA/LTA — тяжёлый) ===
        self.samples_since_heavy += 1
        if self.samples_since_heavy < int(self.heavy_interval * self.sample_rate):
            return None
        self.samples_since_heavy = 0

        self._check_onset(current_idx, current_t, abs_z)
        self._flush_ended_trackers()

        return None
