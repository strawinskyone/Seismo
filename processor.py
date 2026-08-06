import os
import time
import numpy as np
from obspy.core import UTCDateTime
from scipy import signal
import logging

import config

logger = logging.getLogger('seismic')


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

    def update(self, current_idx, current_time, abs_z, global_env,
               adaptive_end_thr):
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
    def __init__(self, event_queue=None, p_onset_callback=None, scope_id_callback=None):
        self.sample_rate = config.SAMPLE_RATE
        self.dt = 1.0 / config.SAMPLE_RATE
        self.buffer_size = int(
            max(80, config.SNAPSHOT_PRE_P_SEC + config.SNAPSHOT_POST_P_SEC + 20) * config.SAMPLE_RATE
        )
        self.buf_z = np.zeros(self.buffer_size, dtype=np.float64)
        self.buf_n = np.zeros(self.buffer_size, dtype=np.float64)
        self.buf_e = np.zeros(self.buffer_size, dtype=np.float64)
        self.buf_t = np.zeros(self.buffer_size, dtype=np.float64)
        self.buf_idx = 0
        self.buf_filled = False

        env_win = max(1, int(config.ENVELOPE_WIN_SEC * self.sample_rate))
        self.env_abs_buf = np.zeros(env_win, dtype=np.float64)
        self.env_buf_idx = 0
        self.global_env = 0.0

        noise_len = int(config.NOISE_ESTIMATE_SEC * self.sample_rate)
        self.noise_env_history = np.zeros(noise_len, dtype=np.float64)
        self.noise_hist_idx = 0
        self.noise_hist_filled = False

        deriv_len = max(100, int(config.NOISE_ESTIMATE_SEC * self.sample_rate))
        self.deriv_history = np.zeros(deriv_len, dtype=np.float64)
        self.deriv_hist_idx = 0

        self.noise_floor = (config.EVENT_THRESHOLD_MV / 1000.0) / config.ADC_SCALE_V
        self.noise_rms = self.noise_floor * 0.5
        self.threshold_derivative = self.noise_floor * 10.0
        self.last_noise_update = 0.0

        delay = max(1, int(0.1 * self.sample_rate))
        self.env_history = np.zeros(delay * 2, dtype=np.float64)
        self.env_hist_idx = 0
        self.env_hist_count = 0

        self.active_trackers = []
        self.max_trackers = config.MAX_TRACKERS
        self.recent_events = []
        self.samples_since_heavy = 0
        self.heavy_interval = config.HEAVY_PROCESS_INTERVAL_SEC
        self.event_queue = event_queue
        self.p_onset_callback = p_onset_callback
        self.scope_id_callback = scope_id_callback
        self.latest_t = 0.0

        logger.info(
            f"[PROCESSOR {config.VERSION}] Init: SR={config.SAMPLE_RATE}, "
            f"buf={self.buffer_size} samp ({self.buffer_size/self.sample_rate:.1f}s), "
            f"max_trackers={config.MAX_TRACKERS}"
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
        if current_t - self.last_noise_update < config.NOISE_UPDATE_INTERVAL_SEC:
            return
        self.last_noise_update = current_t
        hist = (self.noise_env_history if self.noise_hist_filled
                else self.noise_env_history[:self.noise_hist_idx])
        if len(hist) > 100:
            self.noise_floor = float(np.percentile(hist, config.NOISE_PERCENTILE))
            quiet = hist[hist < self.noise_floor * 2.0]
            if len(quiet) > 10:
                self.noise_rms = float(np.sqrt(np.mean((quiet - self.noise_floor) ** 2)))
            else:
                self.noise_rms = self.noise_floor * 0.3
        if self.deriv_hist_idx > 100:
            d_hist = self.deriv_history[:self.deriv_hist_idx]
            median_d = float(np.median(d_hist))
            mad = float(np.median(np.abs(d_hist - median_d))) * 1.4826
            self.threshold_derivative = max(
                self.noise_rms * 2.0,
                median_d + config.ONSET_DERIVATIVE_FACTOR * max(mad, self.noise_rms * 0.1)
            )

    def _adaptive_p_end_threshold(self, tracker):
        abs_thr = self.noise_floor * config.P_END_ABS_FACTOR
        snr = tracker.p_peak / max(self.noise_floor, 1e-12)
        rel_factor = config.P_END_REL_BASE + config.P_END_REL_SNR_FACTOR / np.sqrt(snr + 1.0)
        rel_factor = min(0.60, max(0.15, rel_factor))
        rel_thr = tracker.p_peak * rel_factor
        return max(abs_thr, rel_thr)

    def _check_onset_fast(self, current_idx, current_t, abs_z):
        if self.global_env < self.noise_floor * config.SNR_THRESHOLD:
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
                if abs(trk.onset_time - current_t) < config.ONSET_GUARD_SEC:
                    return False
        debounce_sec = config.EVENT_DEBOUNCE_MS / 1000.0
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
        logger.info(
            f"[ONSET] New tracker @ {current_t:.3f} "
            f"(env={self.global_env:.1f}, d_env={d_env:.1f}, "
            f"noise_floor={self.noise_floor:.1f}, scope={scope_id})"
        )
        if self.p_onset_callback:
            self.p_onset_callback(current_t)
        return True

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
                        f"(was {tracker.onset_time:.3f}, delta={tracker.onset_time - p_time_abs:.3f}s)"
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
        sta_n = max(1, int(config.P_STA_SEC * self.sample_rate))
        lta_n = max(1, int(config.P_LTA_SEC * self.sample_rate))
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
        pre_samples = int(config.SNAPSHOT_PRE_P_SEC * self.sample_rate)
        post_samples = int(config.SNAPSHOT_POST_P_SEC * self.sample_rate)
        a = max(0, p_idx - pre_samples)
        b = min(len(t), p_idx + post_samples)
        # FIX v9.4.1: p_end_idx — кольцевой индекс, теряет смысл при обёртке.
        # Используем p_end_time (абсолютное время) для корректного индекса в snapshot.
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

    def _update_trackers(self, current_idx, current_t, abs_z):
        for tracker in self.active_trackers:
            if tracker.state in ('P_RISING', 'P_PEAKED'):
                adaptive_thr = self._adaptive_p_end_threshold(tracker)
                tracker.update(current_idx, current_t, abs_z, self.global_env, adaptive_thr)

    def _flush_ended_trackers(self):
        ready = []
        remaining = []
        for tracker in self.active_trackers:
            if tracker.state == 'P_ENDED':
                # FIX v9.4.1: не сбрасываем, пока в буфере не накопилось
                # достаточно данных после P для полноценного snapshot.
                required_t = tracker.p_time_abs + config.SNAPSHOT_POST_P_SEC
                if self.latest_t < required_t:
                    if not tracker.flush_delay_logged:
                        logger.debug(
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
            elif tracker.state == 'CLOSED':
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
                        f"len={len(snapshot['z'])}, scope={snapshot['scope_id']})"
                    )
                except Exception:
                    logger.warning("[DISPATCHER] QUEUE FULL, event dropped")

    def process_batch(self, batch_raws, batch_timestamps):
        for raw, ts in zip(batch_raws, batch_timestamps):
            self.process_single(raw, ts)

    def process_single(self, raw, timestamp):
        if len(raw) < 3:
            return None
        self._append(raw[0], raw[1], raw[2], timestamp)
        current_t = float(timestamp)
        abs_z = abs(float(raw[2]))
        current_idx = self._current_chronological_idx()
        self.latest_t = current_t
        self._update_envelope(abs_z)
        self._update_noise_stats(current_t)
        self._update_trackers(current_idx, current_t, abs_z)
        self._check_onset_fast(current_idx, current_t, abs_z)
        self.samples_since_heavy += 1
        if self.samples_since_heavy >= int(self.heavy_interval * self.sample_rate):
            self.samples_since_heavy = 0
            self._refine_active_onsets()
            self._flush_ended_trackers()
        return None
