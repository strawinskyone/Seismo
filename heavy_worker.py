"""
heavy_worker.py v9.6.6
Offline обработка snapshot:
- S-поиск по горизонталям (N/E).
- Взвешивание N/E по RMS — опционально (config.S_USE_RMS_WEIGHTING).
- Интегрирование N/E перед S-поиском — опционально (config.INTEGRATE_FOR_S).
- P-волна для азимута/магнитуды — по Z.
- Азимут — по сырым N/E, без весов.
- Магнитуда Ml — через СМЕЩЕНИЕ (velocity / 2π·f_char).
- v9.6.6: 3 проверки S (амплитуда, граница зоны, резкость cft).
- Сохранение NPZ в snapshots/ для последующего анализа.

Порядок обработки N/E для S-поиска:
    1. detrend linear
    2. (опц.) интегрирование + повторный detrend
    3. bandpass S_FILTER_FREQMIN..S_FILTER_FREQMAX
    4. (опц.) взвешивание по RMS (w_n, w_e)
    5. h = sqrt((N*w_n)^2 + (E*w_e)^2)

ВНИМАНИЕ: processor.py v9.6.2+ отдаёт snapshot в ВОЛЬТАХ (raw LSB * ADC_SCALE_V).
"""
import os
import time
import glob
import numpy as np
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
from scipy.signal import welch, butter, sosfiltfilt
import logging

import config

logger = logging.getLogger('seismic')


# ============================================================
# СОХРАНЕНИЕ NPZ
# ============================================================
SNAPSHOT_DIR = "snapshots"
SNAPSHOT_MAX_FILES = 1000


def _save_snapshot(snapshot, p_idx, p_end_idx, s_idx_global, cft_max,
                   azimuth, rectilinearity, distance, depth,
                   magnitude, is_s, event_type, confidence, p_s_delta,
                   extra=None):
    """Сохраняет snapshot в snapshots/ для анализа. Ошибки не критичны."""
    try:
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        t = snapshot['t']
        ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(float(t[0])))
        ms = int((float(t[0]) % 1.0) * 1000)
        fname = os.path.join(SNAPSHOT_DIR,
                             f"{ts_str}_{ms:03d}_{event_type}.npz")
        payload = dict(
            n=snapshot['n'], e=snapshot['e'], z=snapshot['z'], t=snapshot['t'],
            sample_rate=snapshot['sample_rate'],
            p_idx=p_idx, p_end_idx=p_end_idx, s_idx=s_idx_global,
            cft_max=cft_max if cft_max is not None else np.nan,
            azimuth=azimuth, rectilinearity=rectilinearity,
            distance=distance if distance is not None else np.nan,
            depth=depth if depth is not None else np.nan,
            ml_magnitude=magnitude if magnitude is not None else np.nan,
            is_s=is_s, event_type=event_type, confidence=confidence,
            p_s_delta=p_s_delta if p_s_delta is not None else np.nan,
            scope_id=snapshot.get('scope_id', 0),
            version=config.VERSION,
        )
        if extra:
            for k, v in extra.items():
                payload[k] = v if v is not None else np.nan
        np.savez_compressed(fname, **payload)
        logger.info(f"[WORKER] Snapshot saved: {fname}")
        _rotate_snapshots()
    except Exception as e:
        logger.warning(f"[WORKER] Snapshot save failed: {e}")


def _rotate_snapshots():
    """Удаляет самые старые файлы, если превышен лимит."""
    try:
        files = sorted(glob.glob(os.path.join(SNAPSHOT_DIR, "*.npz")))
        if len(files) > SNAPSHOT_MAX_FILES:
            to_delete = files[:len(files) - SNAPSHOT_MAX_FILES]
            for f in to_delete:
                try:
                    os.remove(f)
                except Exception:
                    pass
            logger.info(f"[WORKER] Rotated {len(to_delete)} old snapshots")
    except Exception as e:
        logger.warning(f"[WORKER] Snapshot rotation failed: {e}")


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


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================
def _bandpass(data, fs, fmin, fmax, order=4):
    if len(data) < 100:
        return data.copy()
    try:
        sos = butter(order, [fmin, fmax], btype='band', fs=fs, output='sos')
        return sosfiltfilt(sos, data)
    except Exception:
        return data.copy()


def _detrend_linear(data):
    if len(data) < 10:
        return data.copy()
    x = np.arange(len(data), dtype=np.float64)
    coef = np.polyfit(x, data, 1)
    return data - (coef[0] * x + coef[1])


def _integrate(data, dt):
    if len(data) < 10:
        return data.copy()
    return np.cumsum(data) * dt


# ============================================================
# ОСНОВНАЯ ОБРАБОТКА
# ============================================================
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

    dt = 1.0 / sample_rate

    # --------------------------------------------------------
    # 1. P-волна (для азимута и магнитуды) — по Z, фильтр 4-28 Гц
    # --------------------------------------------------------
    z_p = _detrend_linear(z)
    z_p = _bandpass(z_p, sample_rate, config.FILTER_FREQMIN,
                    config.FILTER_FREQMAX)

    # --------------------------------------------------------
    # 2. S-поиск — по горизонталям (N/E)
    # --------------------------------------------------------
    n_s = _detrend_linear(n.copy())
    e_s = _detrend_linear(e.copy())

    if getattr(config, 'INTEGRATE_FOR_S', False):
        n_s = _integrate(n_s, dt)
        e_s = _integrate(e_s, dt)
        n_s = _detrend_linear(n_s)
        e_s = _detrend_linear(e_s)

    n_s = _bandpass(n_s, sample_rate, config.S_FILTER_FREQMIN,
                    config.S_FILTER_FREQMAX)
    e_s = _bandpass(e_s, sample_rate, config.S_FILTER_FREQMIN,
                    config.S_FILTER_FREQMAX)

    if getattr(config, 'S_USE_RMS_WEIGHTING', False):
        rms_n = float(np.std(n_s))
        rms_e = float(np.std(e_s))
        if rms_n > 1e-12 and rms_e > 1e-12:
            rms_ref = np.sqrt(rms_n * rms_e)
            w_n = rms_ref / rms_n
            w_e = rms_ref / rms_e
        else:
            w_n = 1.0
            w_e = 1.0
    else:
        w_n = 1.0
        w_e = 1.0

    h_data = np.sqrt((n_s * w_n) ** 2 + (e_s * w_e) ** 2)

    # Зона S-поиска
    s_start = p_idx + config.S_SEARCH_START_AFTER_P_N
    s_end = min(len(h_data), p_idx + config.MAX_P_S_SEARCH_N)
    guard_skip = int(0.2 * sample_rate)
    s_start = min(s_start + guard_skip, s_end - 1)

    s_pick_info = (f"h_zone len={s_end - s_start}, s_start={s_start}, s_end={s_end}, "
                   f"w_n={w_n:.2f}, w_e={w_e:.2f}")
    is_s = False
    p_s_delta = None
    s_idx_global = None
    cft_max = None
    s_amp_ratio = None
    zone_pos = None
    cft_sharpness = None

    if s_end - s_start > 100:
        h_zone = h_data[s_start:s_end]
        try:
            cft_s = recursive_sta_lta(
                h_zone,
                config.S_PICKER_STA_N,
                config.S_PICKER_LTA_N
            )
            cft_max = float(np.max(cft_s)) if len(cft_s) > 0 else 0.0
            cft_mean = float(np.mean(cft_s)) if len(cft_s) > 0 else 0.0
            triggers = trigger_onset(cft_s, config.S_PICKER_TRIGGER,
                                     config.S_PICKER_DETRIGGER)
            s_pick_info += (f", cft_max={cft_max:.2f}, cft_mean={cft_mean:.2f}, "
                            f"trig={config.S_PICKER_TRIGGER}, "
                            f"triggers={len(triggers)}")

            if len(triggers) > 0:
                # v9.6.x: берём ПЕРВЫЙ trigger, а не максимум cft.
                # Артефакт recursive_sta_lta даёт максимум на 1×LTA от начала.
                # Настоящая S может быть раньше.
                best_rel = None
                best_cft = None
                for trg in triggers:
                    on_rel = int(trg[0])
                    if 0 <= on_rel < len(cft_s):
                        best_rel = on_rel
                        best_cft = float(cft_s[on_rel])
                        break   # первый — берём и выходим

                if best_rel is not None:
                    s_rel = best_rel
                    s_idx_global = s_start + s_rel
                    p_s_delta = (s_idx_global - p_idx) / sample_rate
                    zone_len = s_end - s_start
                    zone_pos = s_rel / max(1, zone_len)

                    # --- Проверка 1: амплитуда S > амплитуда P на h ---
                    win_p = int(0.25 * sample_rate)
                    win_s = int(0.25 * sample_rate)
                    p_lo = max(0, p_idx - win_p)
                    p_hi = min(len(h_data), p_idx + win_p)
                    s_lo = max(0, s_idx_global - win_s)
                    s_hi = min(len(h_data), s_idx_global + win_s)
                    h_p_amp = float(np.max(np.abs(h_data[p_lo:p_hi]))) if p_hi > p_lo else 0.0
                    h_s_amp = float(np.max(np.abs(h_data[s_lo:s_hi]))) if s_hi > s_lo else 0.0
                    s_amp_ratio = h_s_amp / (h_p_amp + 1e-12)

                    # --- Проверка 2: не на границе зоны ---
                    near_boundary = (zone_pos < 0.05) or (zone_pos > 0.95)

                    # --- Проверка 3: резкость cft ---
                    # v9.6.7: baseline по ВСЕЙ зоне, не по началу.
                    cft_baseline = float(np.median(cft_s))
                    if cft_baseline < 0.5:
                        cft_baseline = 0.5
                    cft_sharpness = best_cft / cft_baseline

                    s_pick_info += (f", best_rel={best_rel}, best_cft={best_cft:.2f}, "
                                    f"p_s_delta={p_s_delta:.3f}s, "
                                    f"zone_pos={zone_pos:.2f}, "
                                    f"s/p_amp={s_amp_ratio:.2f}, "
                                    f"cft_sharp={cft_sharpness:.2f}")

                    # Все три проверки
                    pass_min_ps = config.MIN_P_S_TIME_SEC < p_s_delta < config.MAX_P_S_TIME_SEC + 1.0
                    pass_amp = s_amp_ratio > 1.0
                    pass_boundary = not near_boundary
                    pass_sharpness = cft_sharpness > 1.1

                    if pass_min_ps and pass_amp and pass_boundary and pass_sharpness:
                        is_s = True
                        s_pick_info += " -> ACCEPTED"
                    else:
                        fails = []
                        if not pass_min_ps:
                            fails.append("min_ps")
                        if not pass_amp:
                            fails.append("amp")
                        if not pass_boundary:
                            fails.append("boundary")
                        if not pass_sharpness:
                            fails.append("sharpness")
                        s_pick_info += f" -> REJECTED ({','.join(fails)})"
        except Exception as ex:
            s_pick_info += f", ERROR: {ex}"
            logger.error(f"[WORKER] Ошибка S-пикера: {ex}")
    else:
        s_pick_info += " -> too short"
        logger.warning(
            f"[WORKER] h_zone слишком короткий ({s_end - s_start}), "
            f"S не ищется. s_start={s_start}, s_end={s_end}, len(h_data)={len(h_data)}"
        )

    # --------------------------------------------------------
    # 3. Азимут — по СЫРЫМ N/E, без весов
    # --------------------------------------------------------
    azimuth, rectilinearity = _calculate_azimuth(n_s, e_s, s_idx_global, sample_rate)

    # --------------------------------------------------------
    # 4. Магнитуда Ml — через СМЕЩЕНИЕ (v9.6.6)
    # --------------------------------------------------------
    spectral_features = _analyze_spectrum(z_p, sample_rate)
    dom_freq = spectral_features.get('dominant_freq', 0.0)
    f_char = dom_freq if dom_freq > 1.0 else 10.0

    rsam, peak_amp = _calculate_rsam(z_p, p_idx, sample_rate)
    peak_amp_v = float(peak_amp)
    velocity_m_s = peak_amp_v / config.GEOPHONE_SENSITIVITY_Z
    velocity_mm_s = velocity_m_s * 1000.0
    velocity_um_s = velocity_m_s * 1e6

    # displacement (смещение) — приближение: v / (2π·f_char)
    displacement_um = velocity_um_s / (2.0 * np.pi * f_char)
    displacement_mm = displacement_um / 1000.0

    distance, depth = _calculate_distance(is_s, p_s_delta)
    if distance is not None and distance > 0:
        # Ml = log10(A_мм) + 1.6·log10(R_км) − 0.15
        magnitude = (np.log10(displacement_mm + 1e-12)
                     + 1.6 * np.log10(distance) - 0.15)
    else:
        magnitude = None

    # --------------------------------------------------------
    # 5. Классификация
    # --------------------------------------------------------
    event_type, confidence = _classify_event(
        is_s, depth, distance, spectral_features, rectilinearity, p_s_delta
    )

    dist_str = f"{distance:.1f} км" if distance is not None else "N/A"
    depth_str = f"{depth:.1f} км" if depth is not None else "N/A"
    mag_str = f"{magnitude:.2f}" if magnitude is not None else "N/A"

    report = [
        "",
        "=" * 70,
        f"--- ОТЧЁТ HEAVY WORKER {config.VERSION} ---",
        "-" * 70,
        f" Наличие S-волны:          {'НАЙДЕНА' if is_s else 'ОТСУТСТВУЕТ / НЕ УВЕРЕН'}",
        f" S-picker диагностика:     {s_pick_info}",
        f" Дельта фаз P–S:           {f'{p_s_delta:.3f} с' if is_s else 'N/A'}",
        f" Дистанция:                {dist_str} | Глубина: {depth_str}",
        f" Азимут:                   {azimuth:.1f}° | Линейность: {rectilinearity:.2f}",
        f" Пиковая амплитуда Z:      {peak_amp_v*1000:.2f} мВ",
        f" Скорость грунта:          {velocity_mm_s:.4f} мм/с",
        f" f_char (dom_freq):        {f_char:.2f} Гц",
        f" Смещение (displacement):  {displacement_um:.2f} µm = {displacement_mm:.5f} мм",
        f" Магнитуда Ml:             {mag_str}",
        f" Тип события:              {event_type.upper()} (conf: {confidence:.2f})",
        f" Время расчёта:            {(time.time()-t0)*1000:.1f} мс",
        "=" * 70,
        ""
    ]
    logger.info("\n".join(report))

    # --- Сохранение NPZ ДО отбрасывания по dead_zone ---
    extra = {
        'displacement_um': displacement_um,
        'displacement_mm': displacement_mm,
        'velocity_mm_s': velocity_mm_s,
        'f_char': f_char,
        'dom_freq': dom_freq,
        's_amp_ratio': s_amp_ratio if s_amp_ratio is not None else np.nan,
        'zone_pos': zone_pos if zone_pos is not None else np.nan,
        'cft_sharpness': cft_sharpness if cft_sharpness is not None else np.nan,
    }
    _save_snapshot(
        snapshot, p_idx, p_end_idx, s_idx_global, cft_max,
        azimuth, rectilinearity, distance, depth,
        magnitude, is_s, event_type, confidence, p_s_delta,
        extra=extra
    )

    if distance is not None and distance <= config.DEAD_ZONE_KM:
        logger.info(f"[WORKER] Событие отброшено: distance={distance:.1f}km <= dead_zone={config.DEAD_ZONE_KM}km")
        return None

    return {
        'status': 'event',
        'magnitude': float(rsam),
        'magnitude_mv': float(rsam * 1000.0),
        'peak_amplitude': float(peak_amp_v),
        'peak_amplitude_mv': float(peak_amp_v * 1000.0),
        'peak_velocity_mm_s': float(velocity_mm_s),
        'displacement_um': float(displacement_um),
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


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ВЫЧИСЛЕНИЯ
# ============================================================
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


def _calculate_azimuth(n_filt, e_filt, s_idx, sample_rate):
    try:
        if s_idx is None or s_idx < 0:
            return 0.0, 0.0
        win = config.AZIMUTH_WIN_N
        a = max(0, s_idx - win)
        b = min(len(n_filt), s_idx + win)
        n_win = n_filt[a:b].astype(np.float64)
        e_win = e_filt[a:b].astype(np.float64)
        if len(n_win) < 10:
            return 0.0, 0.0
        n_win = n_win - np.mean(n_win)
        e_win = e_win - np.mean(e_win)
        cov = np.cov(n_win, e_win)
        vals, vecs = np.linalg.eigh(cov)
        idx_max = int(np.argmax(vals))
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
            win = z_data[-config.RSAM_WIN_N:]
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