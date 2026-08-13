import os
import time
import numpy as np
from obspy import Trace, Stream
from obspy.core import UTCDateTime
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
from scipy.signal import welch
import logging

import config

logger = logging.getLogger('seismic')


def worker_loop(event_queue, result_queue):
    if config.HEAVY_PROCESS_CPU_CORES:
        try:
            os.sched_setaffinity(0, config.HEAVY_PROCESS_CPU_CORES)
            logger.info(f"[WORKER {config.VERSION}] CPU affinity {config.HEAVY_PROCESS_CPU_CORES}")
        except Exception as e:
            logger.warning(f"[WORKER {config.VERSION}] CPU affinity failed: {e}")
    logger.info(f"[WORKER {config.VERSION}] Запущен и ожидает события...")
    while True:
        try:
            snapshot = event_queue.get(timeout=1.0)
        except Exception:
            continue
        if snapshot == 'STOP':
            logger.info(f"[WORKER {config.VERSION}] Получен STOP. Завершение.")
            break
        result = process_snapshot(snapshot)
        if result is not None:
            result_queue.put(result)


# --- v9.5.1: 3D Polarization Filter (Flynn / Remi) с ускорением ---
def _apply_polarization_filter_3d(z, n, e, sample_rate):
    """
    Применяет 3D поляризационный фильтр к Z-каналу.
    Возвращает z_filtered = z * L^2, где L — rectilinearity.
    Оптимизация: вычисление с шагом + линейная интерполяция.
    """
    win_sec = 0.5
    win_samples = max(10, int(win_sec * sample_rate))
    # v9.5.1: шаг = 0.1 секунды (~40 отсчётов при 400 SPS) для ускорения
    step = max(1, int(0.1 * sample_rate))
    L = np.zeros(len(z), dtype=np.float64)

    indices = np.arange(win_samples, len(z), step)
    L_sparse = np.zeros(len(indices), dtype=np.float64)

    for j, i in enumerate(indices):
        zz = z[i - win_samples:i]
        nn = n[i - win_samples:i]
        ee = e[i - win_samples:i]
        data = np.vstack([zz, nn, ee])
        cov = np.cov(data)
        vals = np.linalg.eigvalsh(cov)
        vals = np.sort(vals)[::-1]  # lambda1 >= lambda2 >= lambda3
        l1, l2, l3 = vals[0], vals[1], vals[2]
        rect = 1.0 - (l2 + l3) / (2.0 * l1 + 1e-12)
        L_sparse[j] = rect ** 2

    if len(indices) > 0:
        # Линейная интерполяция на полную сетку
        L[indices] = L_sparse
        if indices[0] > 0:
            L[:indices[0]] = L_sparse[0]
        if len(indices) > 1:
            for k in range(len(indices) - 1):
                a = indices[k]
                b = indices[k + 1]
                L[a:b] = np.linspace(L_sparse[k], L_sparse[k + 1], b - a, endpoint=False)
        L[indices[-1]:] = L_sparse[-1]
    else:
        L[:] = 1.0

    return z * L, L


def process_snapshot(snapshot):
    t0 = time.time()
    sample_rate = snapshot['sample_rate']
    n = snapshot['n']
    e = snapshot['e']
    z = snapshot['z']
    t = snapshot['t']
    p_idx = snapshot['p_idx']
    p_end_idx = snapshot.get('p_end_idx')
    scope_id = snapshot.get('scope_id', 0)
    if len(t) == 0:
        return None
    if p_end_idx is None:
        p_end_idx = p_idx + config.AZIMUTH_WIN_N
    start_time = UTCDateTime(t[0])
    tr_z = Trace(data=z.copy())
    tr_n = Trace(data=n.copy())
    tr_e = Trace(data=e.copy())
    for tr, ch in [(tr_z, 'EHZ'), (tr_n, 'EHN'), (tr_e, 'EHE')]:
        tr.stats.channel = ch
        tr.stats.starttime = start_time
        tr.stats.sampling_rate = sample_rate
    stream_clean = Stream([tr_z, tr_n, tr_e])
    st_p = stream_clean.copy()
    st_p.detrend('linear')
    st_p.filter('bandpass', freqmin=config.FILTER_FREQMIN,
                freqmax=config.FILTER_FREQMAX, corners=4, zerophase=True)
    st_s = stream_clean.copy()
    st_s.detrend('linear')
    st_s.filter('bandpass', freqmin=1.0, freqmax=6.0, corners=4, zerophase=True)

    # --- v9.5.1: 3D Polarization Filter на Z для "подсветки" P-волны ---
    z_orig = z.copy()
    n_orig = n.copy()
    e_orig = e.copy()
    z_pol, L = _apply_polarization_filter_3d(z_orig, n_orig, e_orig, sample_rate)

    # ===== S-PICKER (на исходных горизонталях, т.к. S эллиптическая) =====
    is_s = False
    p_s_delta = None
    s_idx_global = None
    s_pick_info = "S не искалась"
    try:
        n_filt = st_s.select(channel='EHN')[0].data
        e_filt = st_s.select(channel='EHE')[0].data
        h_data = np.sqrt(n_filt**2 + e_filt**2)
        s_start = p_end_idx + config.S_SEARCH_POST_P_END_N
        min_s_start = p_idx + config.MIN_P_S_TIME_N
        s_start = max(s_start, min_s_start)
        s_end = p_idx + config.MAX_P_S_SEARCH_N
        s_end = min(s_end, len(h_data))
        h_zone = h_data[s_start:s_end]
        s_pick_info = f"h_zone len={len(h_zone)}, s_start={s_start}, s_end={s_end}"
        if len(h_zone) > 100:
            cft_s = recursive_sta_lta(
                h_zone,
                config.S_PICKER_STA_N,
                config.S_PICKER_LTA_N
            )
            cft_max = float(np.max(cft_s)) if len(cft_s) > 0 else 0.0
            triggers_s = trigger_onset(cft_s, config.S_PICKER_TRIGGER, config.S_PICKER_DETRIGGER)
            s_pick_info += f", cft_max={cft_max:.2f}, triggers={len(triggers_s)}"
            if len(triggers_s) > 0:
                s_rel = int(triggers_s[0][0])
                s_idx_global = s_start + s_rel
                p_s_delta = (s_idx_global - p_idx) / sample_rate
                s_pick_info += f", s_rel={s_rel}, p_s_delta={p_s_delta:.3f}s"
                if config.MIN_P_S_TIME_SEC < p_s_delta < config.MAX_P_S_TIME_SEC + 1.0:
                    is_s = True
                    s_pick_info += " -> ACCEPTED"
                else:
                    s_pick_info += f" -> REJECTED (out of {config.MIN_P_S_TIME_SEC}-{config.MAX_P_S_TIME_SEC+1}s window)"
            else:
                s_pick_info += " -> no trigger"
        else:
            s_pick_info += " -> too short"
            logger.warning(
                f"[WORKER] h_zone слишком короткий ({len(h_zone)}), S не ищется. "
                f"s_start={s_start}, s_end={s_end}, len(h_data)={len(h_data)}"
            )
    except Exception as e:
        s_pick_info = f"ERROR: {e}"
        logger.error(f"[WORKER] Ошибка S-пикера: {e}")

    # --- v9.5.1: Спектральный анализ и RSAM на polarization-filtered Z ---
    spectral_features = _analyze_spectrum(z_pol, sample_rate)
    azimuth, rectilinearity = _calculate_azimuth_s(st_s, s_idx_global, sample_rate)
    rsam, peak_amp = _calculate_rsam(z_pol, p_idx, sample_rate)
    distance, depth = _calculate_distance(is_s, p_s_delta)
    peak_amp_v = peak_amp * config.ADC_SCALE_V
    velocity_m_s = peak_amp_v / config.GEOPHONE_SENSITIVITY
    velocity_mm_s = velocity_m_s * 1000.0
    velocity_um_s = velocity_m_s * 1e6
    if distance is not None and distance > 0:
        magnitude = (np.log10(velocity_um_s + 1e-10)
                     + 1.6 * np.log10(distance) - 0.15)
    else:
        magnitude = None
    event_type, confidence = _classify_event(
        is_s, depth, distance, spectral_features, rectilinearity, p_s_delta
    )
    dist_str = f"{distance:.1f} км" if distance is not None else "N/A"
    depth_str = f"{depth:.1f} км" if depth is not None else "N/A"
    mag_str = f"{magnitude:.2f}" if magnitude is not None else "N/A"
    report = [
        "",
        "="*70,
        f"--- ОТЧЁТ HEAVY WORKER {config.VERSION} ---",
        "-"*70,
        f" Наличие S-волны:          {'НАЙДЕНА' if is_s else 'ОТСУТСТВУЕТ / НЕ УВЕРЕН'}",
        f" S-picker диагностика:     {s_pick_info}",
        f" Дельта фаз P–S:           {f'{p_s_delta:.3f} с' if is_s else 'N/A'}",
        f" Дистанция:                {dist_str} | Глубина: {depth_str}",
        f" Азимут (S-PCA):           {azimuth:.1f}° | Линейность: {rectilinearity:.2f}",
        f" Пиковая амплитуда:        {peak_amp:.1f} LSB ({peak_amp_v*1000:.2f} мВ)",
        f" Скорость грунта:          {velocity_mm_s:.4f} мм/с",
        f" Магнитуда Ml:             {mag_str}",
        f" Тип события:              {event_type.upper()} (conf: {confidence:.2f})",
        f" Время расчёта:            {(time.time()-t0)*1000:.1f} мс",
        "="*70,
        ""
    ]
    logger.info("\n".join(report))

    if distance is not None and distance <= config.DEAD_ZONE_KM:
        logger.info(f"[WORKER] Событие отброшено: distance={distance:.1f}km <= dead_zone={config.DEAD_ZONE_KM}km")
        return None

    return {
        'status': 'event',
        'magnitude': float(rsam),
        'magnitude_mv': float(rsam * config.ADC_SCALE_V * 1000.0),
        'peak_amplitude': float(peak_amp),
        'peak_amplitude_mv': float(peak_amp_v * 1000.0),
        'peak_velocity_mm_s': float(velocity_mm_s),
        'ml_magnitude': float(magnitude) if magnitude is not None else 0.0,
        'azimuth': float(azimuth),
        'azimuth_reliable': rectilinearity > 0.3,
        'rectilinearity': float(rectilinearity),
        'distance': float(distance) if distance is not None else None,
        'depth': float(depth) if depth is not None else None,
        'event_type': event_type,
        'event_confidence': float(confidence),
        'p_s_delta': float(p_s_delta) if p_s_delta is not None else 0.0,
        'p_time_abs': float(t[p_idx]),
        's_time_abs': float(t[p_idx] + p_s_delta) if (is_s and p_s_delta) else None,
        'spectral_features': spectral_features,
        'scope_id': scope_id,
        'is_s': is_s,
    }


def _analyze_spectrum(data, sample_rate):
    try:
        sig = np.array(data, dtype=np.float64) - np.mean(data)
        nperseg = min(config.FFT_NPERSEG, len(sig))
        if nperseg < 256:
            return {}
        noverlap = min(config.FFT_NOVERLAP, nperseg // 2)
        freqs, psd = welch(sig, fs=sample_rate,
                           nperseg=nperseg,
                           noverlap=noverlap,
                           window=config.FFT_WINDOW)
        dom_idx = int(np.argmax(psd))
        dom_freq = float(freqs[dom_idx])
        low_mask = freqs < 2.0
        high_mask = freqs > 8.0
        low_power = float(np.mean(psd[low_mask])) if np.any(low_mask) else 1e-10
        high_power = float(np.mean(psd[high_mask])) if np.any(high_mask) else 1e-10
        ratio = high_power / low_power
        return {'dominant_freq': dom_freq, 'high_to_low_ratio': ratio}
    except Exception:
        return {}


def _calculate_azimuth_s(st_s, s_idx, sample_rate):
    try:
        if s_idx is None or s_idx < 0:
            return 0.0, 0.0
        n = st_s.select(channel='EHN')[0].data
        e = st_s.select(channel='EHE')[0].data
        win = config.AZIMUTH_WIN_N
        a = max(0, s_idx - win)
        b = min(len(n), s_idx + win)
        n_win = n[a:b].astype(np.float64)
        e_win = e[a:b].astype(np.float64)
        if len(n_win) < 10:
            return 0.0, 0.0
        n_win -= np.mean(n_win)
        e_win -= np.mean(e_win)
        cov = np.cov(n_win, e_win)
        vals, vecs = np.linalg.eigh(cov)
        idx_max = np.argmax(vals)
        v_n, v_e = vecs[:, idx_max]
        az = np.degrees(np.arctan2(v_e, v_n)) % 360.0
        az = (az + config.AZIMUTH_OFFSET) % 360.0
        rect = 1.0 - (np.min(vals) / (np.max(vals) + 1e-10))
        return float(az), float(rect)
    except Exception as e:
        logger.error(f"[WORKER] Ошибка азимута: {e}")
        return 0.0, 0.0


def _calculate_rsam(z_data, p_idx, sample_rate):
    try:
        start = max(0, p_idx)
        end = min(len(z_data), p_idx + config.RSAM_WIN_N)
        win = z_data[start:end]
        if len(win) == 0:
            win = z_data[-config.AZIMUTH_WIN_N:]
        win_detrend = win - np.mean(win)
        rsam = float(np.sqrt(np.mean(win_detrend**2)))
        peak = float(np.max(np.abs(win_detrend)))
        return rsam, peak
    except Exception:
        return 0.0, 0.0


def _calculate_distance(is_s, p_s_delta):
    if is_s and p_s_delta and p_s_delta > 0:
        distance = (p_s_delta * config.VP * config.VS) / (config.VP - config.VS)
        depth = max(0.0, distance * config.DEPTH_FACTOR - config.DEPTH_OFFSET)
        return float(distance), float(depth)
    return None, None


def _classify_event(is_s, depth, distance, spectral_features, rectilinearity, p_s_delta):
    confidence = 0.75
    if rectilinearity > 0.3:
        confidence += 0.05
    if distance is not None and distance > config.DEAD_ZONE_KM:
        confidence += 0.05
    if is_s and p_s_delta and 2.0 < p_s_delta < config.MAX_P_S_TIME_SEC + 1.0:
        confidence += 0.05
    dom_freq = spectral_features.get('dominant_freq', 0.0)
    hilo = spectral_features.get('high_to_low_ratio', 0.0)
    if (dom_freq >= config.EXPLOSION_DOMINANT_FREQ_MIN and
            hilo > config.EXPLOSION_SPECTRAL_THRESHOLD):
        if not is_s or (distance is not None and distance < 3.0):
            return 'explosion', min(0.99, confidence + 0.10)
    if is_s and p_s_delta and 2.0 < p_s_delta < config.MAX_P_S_TIME_SEC + 1.0 \
            and depth is not None and depth > 1.0:
        return 'quake', min(0.99, confidence + 0.10)
    return 'quake', min(0.99, confidence)
