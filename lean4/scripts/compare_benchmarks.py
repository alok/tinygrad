#!/usr/bin/env python3
"""
Compare Lean TinyGrad4 vs Python tinygrad performance.

Runs both implementations on identical workloads and generates a comparison report.
Outputs JSON for CI integration and markdown for readability.

Usage:
    python scripts/compare_benchmarks.py [--output-dir DIR] [--sizes 1000000,4000000]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Force Metal backend
os.environ["METAL"] = "1"


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    name: str
    implementation: str  # "lean" or "python"
    size: int
    time_us: float
    bandwidth_gbs: float
    throughput_gflops: float
    verified: bool


def run_python_benchmark(size: int, iterations: int = 100) -> BenchmarkResult:
    """Run Python tinygrad vector add benchmark."""
    from tinygrad import Tensor, Device

    print(f"  Python tinygrad (size={size:,})")

    # Create data
    a_data = np.array([(i % 1000) / 1000.0 for i in range(size)], dtype=np.float32)
    b_data = np.array([((i + 500) % 1000) / 1000.0 for i in range(size)], dtype=np.float32)

    a = Tensor(a_data)
    b = Tensor(b_data)

    # Warmup
    for _ in range(3):
        out = a + b
        out.realize()

    # Benchmark
    start = time.perf_counter_ns()
    for _ in range(iterations):
        out = a + b
        out.realize()
    end = time.perf_counter_ns()

    time_us = (end - start) / 1e3 / iterations  # microseconds per iteration
    bandwidth = (3.0 * size * 4.0 / time_us) * 1e6 / 1e9
    throughput = (size / time_us) * 1e6 / 1e9

    # Verify
    result = out.numpy()
    expected = a_data + b_data
    verified = np.max(np.abs(result - expected)) < 0.0001

    return BenchmarkResult(
        name="vector_add",
        implementation="python",
        size=size,
        time_us=time_us,
        bandwidth_gbs=bandwidth,
        throughput_gflops=throughput,
        verified=verified,
    )


def run_lean_benchmark(size: int) -> Optional[BenchmarkResult]:
    """Run Lean TinyGrad4 benchmark via executable or JSON output."""
    # Determine script directory to find lean4 project
    script_dir = Path(__file__).parent
    lean4_dir = script_dir.parent  # lean4/scripts -> lean4

    # Check if benchmark executable exists
    exe_path = lean4_dir / ".lake/build/bin/tg4_bench"
    if not exe_path.exists():
        # Try building
        print("  Building Lean benchmark...")
        result = subprocess.run(
            ["lake", "build", "tg4_bench"],
            cwd=lean4_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  Failed to build: {result.stderr}")
            return None

    # Check for pre-existing results
    results_path = lean4_dir / "benchmark_results/latest.json"
    if results_path.exists():
        try:
            with open(results_path) as f:
                data = json.load(f)
            # Find matching size
            for result in data.get("benchmarks", []):
                if result.get("spec", {}).get("size") == size:
                    stats = result.get("stats", {})
                    return BenchmarkResult(
                        name="vector_add",
                        implementation="lean",
                        size=size,
                        time_us=stats.get("mean_us", 0),
                        bandwidth_gbs=result.get("bandwidth_gb_s", 0),
                        throughput_gflops=result.get("throughput_gflops", 0),
                        verified=result.get("verified", False),
                    )
        except (json.JSONDecodeError, KeyError):
            pass

    # Fall back to estimating from last run log
    print(f"  Lean TinyGrad4 (size={size:,}) - using cached results if available")
    return None


def generate_comparison_report(results: list[BenchmarkResult]) -> str:
    """Generate markdown comparison report."""
    lines = [
        "# TinyGrad4 Performance Comparison",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Vector Add Benchmark",
        "",
        "| Implementation | Size | Time (μs) | Bandwidth (GB/s) | Throughput (GFLOP/s) | Status |",
        "|----------------|------|-----------|------------------|----------------------|--------|",
    ]

    # Group by size
    sizes = sorted(set(r.size for r in results))
    for size in sizes:
        for impl in ["python", "lean"]:
            matching = [r for r in results if r.size == size and r.implementation == impl]
            if matching:
                r = matching[0]
                status = "✓" if r.verified else "✗"
                lines.append(
                    f"| {impl.capitalize():12} | {size:,} | {r.time_us:,.2f} | {r.bandwidth_gbs:.2f} | {r.throughput_gflops:.2f} | {status} |"
                )

    # Add comparison
    lines.extend([
        "",
        "## Performance Comparison",
        "",
    ])

    for size in sizes:
        python_results = [r for r in results if r.size == size and r.implementation == "python"]
        lean_results = [r for r in results if r.size == size and r.implementation == "lean"]

        if python_results and lean_results:
            py = python_results[0]
            ln = lean_results[0]
            speedup = py.time_us / ln.time_us if ln.time_us > 0 else 0
            lines.append(f"### Size: {size:,}")
            if speedup > 1:
                lines.append(f"- Lean is **{speedup:.2f}x faster** than Python")
            elif speedup > 0:
                lines.append(f"- Python is **{1/speedup:.2f}x faster** than Lean")
            lines.append(f"- Bandwidth improvement: {ln.bandwidth_gbs / py.bandwidth_gbs:.2f}x")
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare Lean and Python tinygrad performance")
    parser.add_argument("--sizes", default="1000000", help="Comma-separated list of sizes")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    args = parser.parse_args()

    # Determine default output dir relative to script
    if args.output_dir is None:
        script_dir = Path(__file__).parent
        args.output_dir = str(script_dir.parent / "benchmark_results")

    sizes = [int(s.strip()) for s in args.sizes.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TinyGrad4 Benchmark Comparison")
    print("=" * 60)
    print()

    results = []

    # Run Python benchmarks
    print("Running Python tinygrad benchmarks...")
    for size in sizes:
        try:
            result = run_python_benchmark(size)
            results.append(result)
            print(f"    Time: {result.time_us:,.2f} μs, BW: {result.bandwidth_gbs:.2f} GB/s")
        except Exception as e:
            print(f"    Failed: {e}")

    # Run/load Lean benchmarks
    print("\nRunning/loading Lean TinyGrad4 benchmarks...")
    for size in sizes:
        result = run_lean_benchmark(size)
        if result:
            results.append(result)
            print(f"    Time: {result.time_us:,.2f} μs, BW: {result.bandwidth_gbs:.2f} GB/s")
        else:
            print("    (No cached results available)")

    # Generate report
    print("\n" + "=" * 60)

    if not args.json:
        report = generate_comparison_report(results)
        print(report)

        # Save report
        report_path = output_dir / "comparison.md"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")

    # Save JSON
    json_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results": [
            {
                "name": r.name,
                "implementation": r.implementation,
                "size": r.size,
                "time_us": r.time_us,
                "bandwidth_gbs": r.bandwidth_gbs,
                "throughput_gflops": r.throughput_gflops,
                "verified": bool(r.verified),  # Convert numpy bool to Python bool
            }
            for r in results
        ],
    }
    json_path = output_dir / "comparison.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
