"""
PoC: Measure accuracy loss when quantizing float32 vectors to int8.

Run:  python scripts/poc_quantization.py

Simulates sqlite-vec's int8 quantization. Compares top-K ranking produced
by cosine distance on float32 vectors vs int8 vectors.
"""

import math
import random
import statistics
import time
from typing import List

import numpy as np

DIM = 384
NUM_VECTORS = 1000
TOP_K = 10


def normalize_l2(vector: List[float]) -> List[float]:
    squared_sum = sum(x ** 2 for x in vector)
    magnitude = math.sqrt(squared_sum)
    if magnitude == 0:
        return vector
    return [x / magnitude for x in vector]


def vec_quantize_int8(vector: List[float]) -> List[int]:
    """Simulate sqlite-vec's int8 quantization: scale to [-127, 127]."""
    arr = np.array(vector, dtype=np.float32)
    max_abs = np.max(np.abs(arr))
    if max_abs == 0:
        return [0] * len(vector)
    scaled = arr / max_abs * 127.0
    return np.round(scaled).astype(np.int8).tolist()


def cosine_distance_f32(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return 1.0 - max(-1.0, min(1.0, dot))


def cosine_distance_i8(a: List[int], b: List[int]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(v * v for v in a))
    norm_b = math.sqrt(sum(v * v for v in b))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    cosine = dot / (norm_a * norm_b)
    return 1.0 - max(-1.0, min(1.0, cosine))


def main():
    print(f"Quantization PoC: {NUM_VECTORS} random vectors, dim={DIM}, top_k={TOP_K}")
    print("=" * 60)

    rng = random.Random(42)

    # Generate random normalized vectors to simulate real embeddings
    vectors_f32 = [normalize_l2([rng.gauss(0, 1) for _ in range(DIM)]) for _ in range(NUM_VECTORS)]
    vectors_i8 = [vec_quantize_int8(v) for v in vectors_f32]

    total_scores = []
    nonzero_devs = []

    start = time.perf_counter()

    for i in range(min(100, NUM_VECTORS)):
        q_f32 = vectors_f32[i]
        q_i8 = vectors_i8[i]

        # Top-K in float32 space (ground truth)
        f32_dists = [(j, cosine_distance_f32(q_f32, vectors_f32[j])) for j in range(NUM_VECTORS) if j != i]
        f32_dists.sort(key=lambda x: x[1])
        f32_topk = {j for j, _ in f32_dists[:TOP_K]}

        # Top-K in int8 space (approximation)
        i8_dists = [(j, cosine_distance_i8(q_i8, vectors_i8[j])) for j in range(NUM_VECTORS) if j != i]
        i8_dists.sort(key=lambda x: x[1])
        i8_topk = {j for j, _ in i8_dists[:TOP_K]}

        # Overlap score
        overlap = len(f32_topk & i8_topk)
        total_scores.append(overlap / TOP_K)

        # Per-vector cosine deviation
        for j in range(min(20, NUM_VECTORS - 1)):
            if j == i:
                continue
            f32_d = cosine_distance_f32(q_f32, vectors_f32[j])
            i8_d = cosine_distance_i8(q_i8, vectors_i8[j])
            dev = abs(f32_d - i8_d)
            if dev > 0.0001:
                nonzero_devs.append(dev)

    elapsed = time.perf_counter() - start

    # Results
    avg_overlap = statistics.mean(total_scores) * 100
    avg_dev = statistics.mean(nonzero_devs) if nonzero_devs else 0
    max_dev = max(nonzero_devs) if nonzero_devs else 0

    print(f"\nResults ({len(total_scores)} queries, top-{TOP_K}):")
    print(f"  Avg top-K overlap:  {avg_overlap:.1f}%")
    print(f"  Avg cosine deviation: {avg_dev:.6f}")
    print(f"  Max cosine deviation: {max_dev:.6f}")
    print(f"  Time: {elapsed:.3f}s")

    print(f"\nVerdict: ", end="")
    if avg_overlap >= 85:
        print("✅ SAFE — proceed with int8 quantization")
    elif avg_overlap >= 70:
        print("⚠️  CAUTION — measure on real data before enabling")
    else:
        print("❌ NOT SAFE — accuracy loss exceeds threshold")

    print(f"\nStorage estimate for {NUM_VECTORS} vectors:")
    print(f"  float32: {NUM_VECTORS * DIM * 4 / 1024:.0f} KB")
    print(f"  int8:    {NUM_VECTORS * DIM * 1 / 1024:.0f} KB")
    print(f"  saving:  ~75%")


if __name__ == "__main__":
    main()
