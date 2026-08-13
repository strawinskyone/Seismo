#!/usr/bin/env python3
"""
test_events.py v1.0
Генератор тестовых сейсмособытий для отладки карты и интерфейса.
"""
import time
import random
import logging
import numpy as np

import config

logger = logging.getLogger('seismic')
events_logger = logging.getLogger('seismic.events')

# Реалистичные диапазоны для тестовых событий
TEST_AZIMUTHS = list(range(0, 360, 15))          # 0..345°
TEST_DISTANCES = [1.5, 3.2, 5.8, 8.0, 12.5, 15.0, 18.0, 25.0, 35.0]
TEST_TYPES = ['quake', 'explosion', 'quake', 'quake']  # чаще землетрясения
TEST_MAGNITUDES = [0.8, 1.2, 1.5, 1.9, 2.3, 2.7, 3.1]


def _make_event(azimuth, distance, event_type, ml, now):
    """Формирует словарь события, совместимый с MapWidget.add_event и _check_results."""
    peak_mv = 10.0 + (ml ** 2) * 15.0 + random.uniform(-5, 10)
    peak_mv = max(2.0, peak_mv)
    peak_amp = peak_mv / (config.ADC_SCALE_V * 1000.0)
    p_s = None
    is_s = False
    if event_type == 'quake' and distance > 2.0:
        p_s = distance * (config.VP - config.VS) / (config.VP * config.VS)
        p_s = max(config.MIN_P_S_TIME_SEC, min(p_s, config.MAX_P_S_TIME_SEC))
        p_s += random.uniform(-0.3, 0.3)
        is_s = True
    depth = max(0.0, distance * config.DEPTH_FACTOR - config.DEPTH_OFFSET)
    dom_freq = 4.0 + random.uniform(0, 20)
    if event_type == 'explosion':
        dom_freq = 12.0 + random.uniform(0, 25)
    return {
        'status': 'event',
        'magnitude': float(peak_amp * 0.7),
        'magnitude_mv': float(peak_mv * 0.7),
        'peak_amplitude': float(peak_amp),
        'peak_amplitude_mv': float(peak_mv),
        'peak_velocity_mm_s': float(peak_mv / config.GEOPHONE_SENSITIVITY),
        'ml_magnitude': float(ml),
        'azimuth': float(azimuth),
        'azimuth_reliable': random.random() > 0.2,
        'rectilinearity': random.uniform(0.2, 0.95),
        'distance': float(distance),
        'depth': float(depth),
        'event_type': event_type,
        'event_confidence': random.uniform(0.65, 0.98),
        'p_s_delta': float(p_s) if p_s else 0.0,
        'p_time_abs': float(now),
        's_time_abs': float(now + p_s) if p_s else None,
        'spectral_features': {'dominant_freq': float(dom_freq), 'high_to_low_ratio': random.uniform(0.5, 5.0)},
        'scope_id': 0,
        'is_s': is_s,
    }


def inject_test_events(map_widget, scopes, count=None):
    """
    Генерирует и внедряет count тестовых событий на карту.
    Если count=None — случайно 5..7.
    """
    if count is None:
        count = random.randint(5, 7)
    now = time.time()
    logger.info(f"[TEST] Генерация {count} тестовых событий...")
    events_injected = 0

    for i in range(count):
        az = random.choice(TEST_AZIMUTHS)
        # Небольшой разброс, чтобы не ложились точно в одну точку
        az += random.uniform(-8, 8)
        az %= 360.0

        dist = random.choice(TEST_DISTANCES)
        # Разброс дистанции
        dist *= random.uniform(0.85, 1.15)
        dist = max(config.DEAD_ZONE_KM + 0.5, dist)

        etype = random.choice(TEST_TYPES)
        ml = random.choice(TEST_MAGNITUDES)
        # Масштабируем магнитуду под дистанцию (ближе = тише)
        if dist < 5.0:
            ml *= random.uniform(0.6, 1.0)
        elif dist > 20.0:
            ml *= random.uniform(1.0, 1.8)

        evt = _make_event(az, dist, etype, ml, now + i * 0.15)

        # На карту
        added = map_widget.add_event(evt)
        if not added:
            continue
        events_injected += 1

        # P/S маркеры на текущий осциллограф [0]
        if scopes:
            scopes[0].add_p_marker(evt['p_time_abs'])
            if evt['s_time_abs']:
                scopes[0].add_s_marker(evt['s_time_abs'])

        # Лог
        dist_str = f"{dist:5.1f}km"
        p_s_str = f"{evt['p_s_delta']:.2f}s" if evt['is_s'] else "N/A"
        line = (f"TEST EVENT {i+1}/{count} | {etype.upper():>6s} | "
                f"D={dist_str} | Ml={ml:4.2f} | Az={az:5.1f}° | "
                f"conf={evt['event_confidence']:.2f} | ΔP-S={p_s_str}")
        events_logger.info(line)
        logger.info(f"[TEST] {line}")

    logger.info(f"[TEST] Внедрено {events_injected}/{count} событий")
    return events_injected