"""
DREDGE Batch Processing Pipeline
Demonstrates real-world use case for batch unified_inference with load testing.
"""
import time
import json
import asyncio
import statistics
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from pathlib import Path

# Add src to path if running standalone
sys.path.insert(0, str(Path(__file__).parent.parent))

from dredge.mcp_server import QuasimotoMCPServer
from dredge.monitoring import get_metrics_collector


class BatchInferencePipeline:
    """Pipeline for batch processing unified inference requests."""
    
    def __init__(self, use_cache: bool = True, num_workers: int = 4):
        """
        Initialize batch pipeline.
        
        Args:
            use_cache: Enable caching for faster repeat queries
            num_workers: Number of parallel workers
        """
        self.server = QuasimotoMCPServer(use_cache=use_cache, enable_metrics=True)
        self.num_workers = num_workers
        self.metrics = get_metrics_collector()
        print(f"✓ Initialized pipeline with {num_workers} workers, caching={'ON' if use_cache else 'OFF'}")
    
    def process_single(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single unified inference task."""
        start = time.time()
        
        result = self.server.unified_inference({
            'dredge_insight': task['insight'],
            'quasimoto_coords': task['coords'],
            'string_modes': task['modes']
        })
        
        duration = time.time() - start
        
        return {
            'task_id': task['task_id'],
            'success': result.get('success', False),
            'duration': duration,
            'cached': result.get('cached', False)
        }
    
    def process_batch(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process batch of tasks in parallel.
        
        Args:
            tasks: List of task dictionaries
            
        Returns:
            Batch processing results with statistics
        """
        print(f"\n📊 Processing {len(tasks)} tasks...")
        start_time = time.time()
        
        results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(self.process_single, task): task for task in tasks}
            
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                results.append(result)
                
                if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
                    print(f"  Progress: {i + 1}/{len(tasks)} tasks completed")
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        durations = [r['duration'] for r in results]
        successes = sum(1 for r in results if r['success'])
        cache_hits = sum(1 for r in results if r.get('cached', False))
        
        stats = {
            'total_tasks': len(tasks),
            'successful': successes,
            'failed': len(tasks) - successes,
            'cache_hits': cache_hits,
            'cache_hit_rate': cache_hits / len(tasks) if tasks else 0,
            'total_time': total_time,
            'throughput': len(tasks) / total_time if total_time > 0 else 0,
            'latency': {
                'min': min(durations) if durations else 0,
                'max': max(durations) if durations else 0,
                'mean': statistics.mean(durations) if durations else 0,
                'median': statistics.median(durations) if durations else 0,
                'p95': statistics.quantiles(durations, n=20)[18] if len(durations) >= 20 else max(durations) if durations else 0,
                'p99': statistics.quantiles(durations, n=100)[98] if len(durations) >= 100 else max(durations) if durations else 0,
            }
        }
        
        return {
            'results': results,
            'stats': stats
        }


def generate_test_tasks(count: int, with_repeats: bool = True) -> List[Dict[str, Any]]:
    """
    Generate test tasks for load testing.
    
    Args:
        count: Number of tasks to generate
        with_repeats: Include repeat queries to test caching
        
    Returns:
        List of task dictionaries
    """
    insights = [
        "Digital memory must be human-reachable",
        "Quantum coherence in neural networks",
        "String vibrations encode information",
        "Unified field theory emergence",
        "Spacetime curvature affects computation"
    ]
    
    tasks = []
    for i in range(count):
        # Create some repeats to test caching
        if with_repeats and i % 5 == 0 and i > 0:
            # Repeat an earlier task
            task = tasks[i // 5].copy()
            task['task_id'] = f"task_{i:04d}_repeat"
        else:
            task = {
                'task_id': f"task_{i:04d}",
                'insight': insights[i % len(insights)],
                'coords': [0.1 * (i % 10), 0.2 * (i % 5), 0.3 * (i % 3)],
                'modes': [1 + (i % 3), 2 + (i % 2), 3]
            }
        tasks.append(task)
    
    return tasks


def run_load_test(num_tasks: int = 100, num_workers: int = 4, with_cache: bool = True):
    """
    Run a complete load test.
    
    Args:
        num_tasks: Number of tasks to process
        num_workers: Number of parallel workers
        with_cache: Enable caching
    """
    print("=" * 70)
    print("DREDGE Batch Unified Inference Load Test")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Tasks: {num_tasks}")
    print(f"  Workers: {num_workers}")
    print(f"  Caching: {'Enabled' if with_cache else 'Disabled'}")
    print("=" * 70)
    
    # Initialize pipeline
    pipeline = BatchInferencePipeline(use_cache=with_cache, num_workers=num_workers)
    
    # Generate test data
    print(f"\n📝 Generating {num_tasks} test tasks...")
    tasks = generate_test_tasks(num_tasks, with_repeats=with_cache)
    print(f"✓ Generated {len(tasks)} tasks")
    
    # Run batch processing
    result = pipeline.process_batch(tasks)
    stats = result['stats']
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total Tasks:      {stats['total_tasks']}")
    print(f"Successful:       {stats['successful']}")
    print(f"Failed:           {stats['failed']}")
    print(f"Cache Hits:       {stats['cache_hits']}")
    print(f"Cache Hit Rate:   {stats['cache_hit_rate']:.1%}")
    print(f"\nPerformance:")
    print(f"Total Time:       {stats['total_time']:.2f}s")
    print(f"Throughput:       {stats['throughput']:.2f} tasks/sec")
    print(f"\nLatency (seconds):")
    print(f"  Min:            {stats['latency']['min']:.4f}")
    print(f"  Mean:           {stats['latency']['mean']:.4f}")
    print(f"  Median:         {stats['latency']['median']:.4f}")
    print(f"  Max:            {stats['latency']['max']:.4f}")
    print(f"  P95:            {stats['latency']['p95']:.4f}")
    print(f"  P99:            {stats['latency']['p99']:.4f}")
    print("=" * 70)
    
    # Get server metrics
    metrics_result = pipeline.server.get_metrics()
    if metrics_result['success']:
        print("\nServer Metrics:")
        metrics = metrics_result['metrics']
        print(f"  Counters: {json.dumps(metrics.get('counters', {}), indent=4)}")
    
    # Get cache stats
    if with_cache:
        cache_result = pipeline.server.get_cache_stats()
        if cache_result['success']:
            print("\nCache Statistics:")
            print(f"  {json.dumps(cache_result['cache_stats'], indent=4)}")
    
    return result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='DREDGE Batch Processing Load Test')
    parser.add_argument('--tasks', type=int, default=100, help='Number of tasks (default: 100)')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers (default: 4)')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    
    args = parser.parse_args()
    
    run_load_test(
        num_tasks=args.tasks,
        num_workers=args.workers,
        with_cache=not args.no_cache
    )
