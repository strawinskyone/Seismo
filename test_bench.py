#!/usr/bin/env python3
"""
test_bench.py v2.6.1
Двойной тест для seismic processor + heavy worker.
"""
import numpy as np
import time
import sys
from multiprocessing import Process, Queue

import config
from processor import SeismicProcessor
from heavy_worker import worker_loop


def generate_two_events():
    fs = config.SAMPLE_RATE
    total_sec = 85.0
    total = int(total_sec * fs)
    np.random.seed(42)
    noise_lsb = 1.5
    data_x = np.random.normal(0, noise_lsb, total)
    data_y = np.random.normal(0, noise_lsb, total)
    data_z = np.random.normal(0, noise_lsb, total)
    data_w = np.random.normal(0, noise_lsb, total)

    p1 = int(20.0 * fs)
    p1_len = int(1.5 * fs)
    t_p = np.linspace(0, 1.5, p1_len)
    p_wave = 800.0 * np.exp(-4.0 * t_p) * np.sin(2 * np.pi * 25.0 * t_p)
    data_z[p1:p1 + p1_len] += p_wave
    data_x[p1:p1 + p1_len] += p_wave * 0.1
    data_y[p1:p1 + p1_len] += p_wave * 0.05

    s1 = int(25.0 * fs)
    s1_len = int(4.0 * fs)
    t_s = np.linspace(0, 4.0, s1_len)
    s_wave = 1200.0 * np.sin(np.pi * t_s / 4.0) * np.sin(2 * np.pi * 4.5 * t_s)
    az1 = np.radians(60.0)
    data_x[s1:s1 + s1_len] += s_wave * np.cos(az1)
    data_y[s1:s1 + s1_len] += s_wave * np.sin(az1)

    p2 = int(45.0 * fs)
    p2_len = int(1.5 * fs)
    t_p2 = np.linspace(0, 1.5, p2_len)
    p_wave2 = 850.0 * np.exp(-4.0 * t_p2) * np.sin(2 * np.pi * 25.0 * t_p2)
    data_z[p2:p2 + p2_len] += p_wave2
    data_x[p2:p2 + p2_len] += p_wave2 * 0.1
    data_y[p2:p2 + p2_len] += p_wave2 * 0.05

    s2 = int(50.0 * fs)
    s2_len = int(4.0 * fs)
    t_s2 = np.linspace(0, 4.0, s2_len)
    s_wave2 = 1300.0 * np.sin(np.pi * t_s2 / 4.0) * np.sin(2 * np.pi * 4.5 * t_s2)
    az2 = np.radians(120.0)
    data_x[s2:s2 + s2_len] += s_wave2 * np.cos(az2)
    data_y[s2:s2 + s2_len] += s_wave2 * np.sin(az2)

    start_time = time.time()
    timestamps = np.array([start_time + (i / fs) for i in range(total)])
    return data_x, data_y, data_z, data_w, timestamps


def run_test():
    total_sec = 85.0
    print("=" * 70)
    print(f"🚀  СТАРТ ДВОЙНОГО ТЕСТА  (test_bench v2.6.1 → processor {config.VERSION})")
    print("=" * 70)

    data_x, data_y, data_z, data_w, timestamps = generate_two_events()
    event_queue = Queue(maxsize=config.QUEUE_MAXSIZE)
    result_queue = Queue()
    worker = Process(target=worker_loop, args=(event_queue, result_queue), daemon=True)
    worker.start()
    processor = SeismicProcessor(event_queue=event_queue)

    fs = config.SAMPLE_RATE
    batch_size = 20
    total = len(data_x)
    print(f"[INFO] Сигнал: {total} точек ({total_sec:.1f} с), {fs} Гц")
    print(f"[INFO] Событие 1: P@20с S@25с az=60°")
    print(f"[INFO] Событие 2: P@45с S@50с az=120°")
    print("[INFO] Эмуляция DAQ-потока...")

    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        if end_idx - start_idx < batch_size:
            break
        chunk_raws = []
        chunk_ts = []
        for i in range(start_idx, end_idx):
            chunk_raws.append((float(data_x[i]), float(data_y[i]),
                               float(data_z[i]), float(data_w[i])))
            chunk_ts.append(timestamps[i])
        processor.process_batch(chunk_raws, chunk_ts)
        time.sleep(batch_size / fs)

    print("[INFO] Добавляем 2с тишины для завершения трекеров...")
    silence_t = timestamps[-1] if len(timestamps) > 0 else time.time()
    for i in range(int(2.0 * fs)):
        silence_t += 1.0 / fs
        processor.process_single((0.0, 0.0, 0.0, 0.0), silence_t)
    processor._refine_active_onsets()
    processor._flush_ended_trackers()

    print(f"[INFO] Активных трекеров перед flush: {len(processor.active_trackers)}")
    print(f"[INFO] Активных трекеров после flush: {len(processor.active_trackers)}")
    print(f"[INFO] Закрытых событий (recent_events): {len(processor.recent_events)}")
    print(f"[INFO] Событий в event_queue: {event_queue.qsize()}")
    print("[INFO] Ожидание 5с для обработки heavy_worker...")

    time.sleep(5.0)
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    event_queue.put('STOP')
    worker.join(timeout=5.0)
    if worker.is_alive():
        worker.terminate()

    print(f"\n[INFO] Получено результатов: {len(results)}")
    for i, r in enumerate(results):
        print(f"  Result {i+1}: type={r.get('event_type')}, "
              f"dist={r.get('distance'):.1f} км, az={r.get('azimuth'):.1f}°")

    if len(results) == 2:
        print("\n✅ ТЕСТ ПРОЙДЕН: оба события обнаружены и обработаны.")
        return True
    else:
        print(f"\n❌ ОЖИДАЛОСЬ 2 СОБЫТИЯ, ПОЛУЧЕНО {len(results)}")
        return False


if __name__ == "__main__":
    try:
        ok = run_test()
        sys.exit(0 if ok else 1)
    except KeyboardInterrupt:
        print("\nПрервано.")
        sys.exit(0)
