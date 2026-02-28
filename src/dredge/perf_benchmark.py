"""
DREDGE Performance Benchmark
Comprehensive throughput benchmark covering all three optimisation areas:
  1. Async GitHub API client (mocked to avoid network dependency)
  2. Batch inference pipeline (StringTheoryNN)
  3. Event stream handler throughput
"""
import asyncio
import statistics
import time
from typing import Any, Dict, List

from .batch_inference_pipeline import BatchInferenceEngine, BatchVibrationalModes
from .event_stream_handler import EventStreamProcessor, process_events_from_list
from .string_theory import get_optimal_device


# ---------------------------------------------------------------------------
# 1.  GitHub API throughput (mocked)
# ---------------------------------------------------------------------------


async def _mock_github_get(path: str) -> Dict[str, Any]:
    """Simulates a GitHub API round-trip with 20 ms latency."""
    await asyncio.sleep(0.02)
    return {"path": path, "status": "ok"}


async def _benchmark_github_async(num_requests: int = 50, concurrency: int = 10) -> Dict[str, Any]:
    """
    Benchmark the async batch approach vs sequential mock requests.

    Args:
        num_requests: Total number of simulated API calls.
        concurrency: Max concurrent requests for the async run.

    Returns:
        Dictionary comparing sequential vs async throughput.
    """
    paths = [f"/repos/owner/repo/pulls/{i}" for i in range(num_requests)]
    semaphore = asyncio.Semaphore(concurrency)

    # --- Sequential baseline ---
    t0 = time.perf_counter()
    for path in paths:
        await _mock_github_get(path)
    sequential_elapsed = time.perf_counter() - t0

    # --- Async concurrent ---
    async def _bounded_get(path: str) -> Dict[str, Any]:
        async with semaphore:
            return await _mock_github_get(path)

    t0 = time.perf_counter()
    await asyncio.gather(*[_bounded_get(p) for p in paths])
    async_elapsed = time.perf_counter() - t0

    speedup = sequential_elapsed / async_elapsed if async_elapsed > 0 else 0

    return {
        "benchmark": "github_api",
        "num_requests": num_requests,
        "concurrency": concurrency,
        "sequential_seconds": sequential_elapsed,
        "async_seconds": async_elapsed,
        "speedup": speedup,
        "sequential_rps": num_requests / sequential_elapsed if sequential_elapsed > 0 else 0,
        "async_rps": num_requests / async_elapsed if async_elapsed > 0 else 0,
    }


def benchmark_github_api(
    num_requests: int = 50, concurrency: int = 10
) -> Dict[str, Any]:
    """Synchronous wrapper for the GitHub API benchmark."""
    return asyncio.run(_benchmark_github_async(num_requests, concurrency))


# ---------------------------------------------------------------------------
# 2.  Batch inference throughput
# ---------------------------------------------------------------------------


def benchmark_batch_inference(
    num_samples: int = 500,
    batch_sizes: List[int] = None,
    dimensions: int = 10,
    device: str = "auto",
) -> Dict[str, Any]:
    """
    Benchmark StringTheoryNN inference throughput at several batch sizes.

    Args:
        num_samples: Total number of samples to infer.
        batch_sizes: Batch sizes to test (default: [1, 8, 32, 64]).
        dimensions: Input dimensionality.
        device: Compute device (``'auto'`` = best available).

    Returns:
        Dictionary mapping each batch size to throughput metrics.
    """
    if batch_sizes is None:
        batch_sizes = [1, 8, 32, 64]

    resolved_device = get_optimal_device() if device == "auto" else device

    results: Dict[str, Any] = {
        "benchmark": "batch_inference",
        "num_samples": num_samples,
        "device": resolved_device,
        "results_by_batch_size": {},
    }

    for bs in batch_sizes:
        engine = BatchInferenceEngine(
            dimensions=dimensions,
            batch_size=bs,
            device=resolved_device,
        )
        stats = engine.benchmark(num_samples=num_samples, chunk_size=bs)
        results["results_by_batch_size"][bs] = {
            "throughput_per_sec": stats["throughput_per_sec"],
            "latency_ms_per_sample": stats["latency_ms_per_sample"],
            "total_seconds": stats["total_seconds"],
        }

    # Best throughput achieved
    best_bs = max(
        results["results_by_batch_size"],
        key=lambda k: results["results_by_batch_size"][k]["throughput_per_sec"],
    )
    results["best_batch_size"] = best_bs
    results["best_throughput_per_sec"] = results["results_by_batch_size"][best_bs][
        "throughput_per_sec"
    ]

    return results


# ---------------------------------------------------------------------------
# 3.  Event stream throughput
# ---------------------------------------------------------------------------


def benchmark_event_stream(
    num_events: int = 200,
    worker_counts: List[int] = None,
    simulated_latency_ms: float = 5.0,
) -> Dict[str, Any]:
    """
    Benchmark event stream handler at several worker-count settings.

    Args:
        num_events: Total events to process per run.
        worker_counts: Worker counts to test (default: [1, 2, 4, 8]).
        simulated_latency_ms: Per-event processing delay in ms.

    Returns:
        Dictionary comparing throughput across worker counts.
    """
    if worker_counts is None:
        worker_counts = [1, 2, 4, 8]

    async def _slow_handler(event: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate async work per event."""
        await asyncio.sleep(simulated_latency_ms / 1000.0)
        return {"event_id": event.get("id"), "status": "processed"}

    events = [{"id": i, "type": "push", "payload": {}} for i in range(num_events)]

    results: Dict[str, Any] = {
        "benchmark": "event_stream",
        "num_events": num_events,
        "simulated_latency_ms": simulated_latency_ms,
        "results_by_workers": {},
    }

    for n_workers in worker_counts:
        drain_timeout = (num_events * simulated_latency_ms / 1000.0 / n_workers) * 3 + 5
        summary = process_events_from_list(
            events=events,
            handler=_slow_handler,
            num_workers=n_workers,
            drain_timeout=drain_timeout,
        )
        results["results_by_workers"][n_workers] = {
            "throughput_per_sec": summary["throughput_per_sec"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "successful": summary["successful"],
        }

    best_w = max(
        results["results_by_workers"],
        key=lambda k: results["results_by_workers"][k]["throughput_per_sec"],
    )
    results["best_workers"] = best_w
    results["best_throughput_per_sec"] = results["results_by_workers"][best_w][
        "throughput_per_sec"
    ]

    return results


# ---------------------------------------------------------------------------
# 4.  Unified summary
# ---------------------------------------------------------------------------


def run_all_benchmarks(
    github_requests: int = 50,
    inference_samples: int = 500,
    event_count: int = 200,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run all three benchmarks and print a human-readable summary.

    Args:
        github_requests: Number of GitHub API requests to simulate.
        inference_samples: Number of inference samples.
        event_count: Number of events to process.
        verbose: Print results to stdout.

    Returns:
        Combined results dictionary.
    """
    if verbose:
        print("=" * 60)
        print("DREDGE Throughput Benchmark Suite")
        print("=" * 60)

    # --- GitHub API ---
    if verbose:
        print("\n[1/3] GitHub API throughput …")
    github_results = benchmark_github_api(num_requests=github_requests)
    if verbose:
        print(
            f"  Sequential: {github_results['sequential_rps']:.1f} req/s  "
            f"Async: {github_results['async_rps']:.1f} req/s  "
            f"Speedup: {github_results['speedup']:.1f}x"
        )

    # --- Batch inference ---
    if verbose:
        print("\n[2/3] Batch inference throughput …")
    inference_results = benchmark_batch_inference(num_samples=inference_samples)
    if verbose:
        print(
            f"  Best batch size: {inference_results['best_batch_size']}  "
            f"Throughput: {inference_results['best_throughput_per_sec']:.1f} samples/s"
        )

    # --- Event stream ---
    if verbose:
        print("\n[3/3] Event stream throughput …")
    event_results = benchmark_event_stream(num_events=event_count)
    if verbose:
        print(
            f"  Best workers: {event_results['best_workers']}  "
            f"Throughput: {event_results['best_throughput_per_sec']:.1f} events/s"
        )

    if verbose:
        print("\n" + "=" * 60)
        print("Benchmark complete.")
        print("=" * 60)

    return {
        "github_api": github_results,
        "batch_inference": inference_results,
        "event_stream": event_results,
    }


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="DREDGE Throughput Benchmark")
    parser.add_argument("--github-requests", type=int, default=50)
    parser.add_argument("--inference-samples", type=int, default=500)
    parser.add_argument("--event-count", type=int, default=200)
    args = parser.parse_args()

    run_all_benchmarks(
        github_requests=args.github_requests,
        inference_samples=args.inference_samples,
        event_count=args.event_count,
    )
