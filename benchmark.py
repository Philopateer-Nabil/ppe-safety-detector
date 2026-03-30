"""Benchmark inference latency for PyTorch vs ONNX backends.

Usage::

    python benchmark.py                  # default 200 iterations
    python benchmark.py --iterations 500
"""

from __future__ import annotations

import argparse

import numpy as np

from src.detector import Backend, Detector


def run_benchmark(backend: Backend, iterations: int) -> dict[str, float]:
    detector = Detector(backend=backend)
    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # warm-up
    for _ in range(10):
        detector.detect(dummy)

    times: list[float] = []
    for _ in range(iterations):
        _, ms = detector.detect(dummy)
        times.append(ms)

    arr = np.array(times)
    return {
        "backend": backend.name,
        "iterations": iterations,
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "fps_mean": 1000.0 / arr.mean(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PPE detector")
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    header = f"{'Backend':<10} {'Mean':>8} {'Median':>8} {'P95':>8} {'FPS':>8}"
    print(header)
    print("-" * len(header))

    for backend in (Backend.PYTORCH, Backend.ONNX):
        try:
            stats = run_benchmark(backend, args.iterations)
            print(
                f"{stats['backend']:<10} "
                f"{stats['mean_ms']:>7.1f}ms "
                f"{stats['median_ms']:>7.1f}ms "
                f"{stats['p95_ms']:>7.1f}ms "
                f"{stats['fps_mean']:>7.1f}"
            )
        except Exception as exc:
            print(f"{backend.name:<10}  skipped ({exc})")


if __name__ == "__main__":
    main()
