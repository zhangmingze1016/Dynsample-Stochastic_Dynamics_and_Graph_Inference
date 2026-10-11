"""Compare cached and uncached fixed-drift offset fitting."""

import csv
import gc
from pathlib import Path
from time import perf_counter
import tracemalloc

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, solve_triangular
from scipy.optimize import OptimizeResult

from dynsample.core.linear_model import LinearSDE
from dynsample.core.state import State
from dynsample.core.trajectory import Trajectory
from dynsample.estimation.linear import (
    fit_linear_offset,
    linear_negative_log_likelihood,
)
from dynsample.simulation.linear import (
    linear_transition,
    simulate_linear,
)


N_INTERVALS = 256
TIMING_REPEATS = 5
MEMORY_REPEATS = 3


# Reproduce the earlier uncached calculation for this controlled benchmark.
def _fit_offset_uncached(
    trajectory: Trajectory,
    drift: np.ndarray,
    diffusion: np.ndarray,
) -> OptimizeResult:
    """Fit the offset without interval caching or likelihood reuse."""
    n = trajectory.n_nodes

    model = LinearSDE(
        drift=drift,
        offset=np.zeros(n),
        diffusion=diffusion,
    )

    intervals = np.diff(trajectory.times)
    values = trajectory.values[:, :, 0]

    integral_block = np.zeros((2 * n, 2 * n))
    integral_block[:n, :n] = model.drift
    integral_block[:n, n:] = np.eye(n)

    design_blocks = []
    response_blocks = []

    for i, dt in enumerate(intervals, start=1):
        transition, _, covariance = linear_transition(
            model=model,
            dt=float(dt),
        )

        block_exponential = expm(integral_block * dt)
        integrated_transition = block_exponential[:n, n:]

        response = values[i] - transition @ values[i - 1]
        lower = np.linalg.cholesky(covariance)

        design_blocks.append(
            solve_triangular(
                lower,
                integrated_transition,
                lower=True,
            )
        )
        response_blocks.append(
            solve_triangular(
                lower,
                response,
                lower=True,
            )
        )

    design = np.vstack(design_blocks)
    response = np.concatenate(response_blocks)

    offset, _, rank, singular_values = np.linalg.lstsq(
        design,
        response,
        rcond=None,
    )

    if rank < n:
        raise ValueError("reference offset design is rank deficient")

    if not np.all(np.isfinite(offset)):
        raise FloatingPointError("reference offset is non-finite")

    fitted_model = LinearSDE(
        drift=model.drift,
        offset=offset,
        diffusion=model.diffusion,
    )

    # Intentionally recompute transitions and factors, as in the old path.
    score = linear_negative_log_likelihood(
        trajectory=trajectory,
        model=fitted_model,
    )

    return OptimizeResult(
        offset=offset,
        model=fitted_model,
        fun=float(score),
        success=True,
        rank=int(rank),
        singular_values=singular_values,
    )


# Use equal interval counts and durations with different amounts of repetition.
def _build_schedules():
    regular = np.full(N_INTERVALS, 0.5)

    repeated = np.tile(
        [0.25, 0.75],
        N_INTERVALS // 2,
    )

    # Binary-exact fractions avoid accidental rounding differences in dt.
    distinct = (
        np.arange(N_INTERVALS, dtype=np.float64) + 0.5
    ) / N_INTERVALS

    # Keep all intervals distinct without ordering them from small to large.
    distinct = np.random.default_rng(123).permutation(distinct)

    return {
        "Regular": regular,
        "Repeated irregular": repeated,
        "All distinct": distinct,
    }


# Construct a small coupled system with positive-definite noise covariance.
def _build_model():
    model = LinearSDE(
        drift=np.array([
            [-0.7, 0.0, 0.0],
            [0.3, -1.0, 0.0],
            [0.0, 0.2, -0.5],
        ]),
        offset=np.array([0.4, -0.2, 0.1]),
        diffusion=np.array([
            [0.6, 0.0, 0.0],
            [0.1, 0.5, 0.0],
            [0.0, 0.1, 0.4],
        ]),
    )

    initial_state = State(
        time=0.0,
        values=np.array([[2.0], [-1.0], [0.5]]),
    )

    return model, initial_state


# Fail the benchmark if an optimization changes the fitted result materially.
def _assert_equivalent(cached, uncached):
    np.testing.assert_allclose(
        cached.offset,
        uncached.offset,
        rtol=1e-9,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        cached.fun,
        uncached.fun,
        rtol=1e-9,
        atol=1e-9,
    )


# Measure execution time without memory tracing.
def _measure_seconds(function, trajectory, model):
    gc.collect()

    start = perf_counter()

    result = function(
        trajectory=trajectory,
        drift=model.drift,
        diffusion=model.diffusion,
    )

    elapsed = perf_counter() - start
    return result, elapsed


# Measure traced allocations separately so tracing does not distort timings.
def _measure_peak_bytes(function, trajectory, model):
    gc.collect()
    tracemalloc.start()

    try:
        result = function(
            trajectory=trajectory,
            drift=model.drift,
            diffusion=model.diffusion,
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    return result, peak


# Benchmark both implementations on exactly the same observations.
def _benchmark_case(name, trajectory, model):
    implementations = {
        "Cached": fit_linear_offset,
        "Uncached": _fit_offset_uncached,
    }

    # Warm up both paths and validate before collecting performance results.
    warm_results = {
        method: function(
            trajectory=trajectory,
            drift=model.drift,
            diffusion=model.diffusion,
        )
        for method, function in implementations.items()
    }
    _assert_equivalent(
        warm_results["Cached"],
        warm_results["Uncached"],
    )

    timing = {method: [] for method in implementations}
    memory = {method: [] for method in implementations}
    raw_records = []

    for repeat in range(TIMING_REPEATS):
        # Alternate order to reduce systematic first/second-run effects.
        order = (
            ["Uncached", "Cached"]
            if repeat % 2 == 0
            else ["Cached", "Uncached"]
        )

        results = {}

        for method in order:
            result, seconds = _measure_seconds(
                implementations[method],
                trajectory,
                model,
            )
            results[method] = result
            timing[method].append(seconds)

            raw_records.append({
                "schedule": name,
                "measurement": "time",
                "repeat": repeat + 1,
                "method": method,
                "seconds": seconds,
                "traced_peak_bytes": "",
            })

        _assert_equivalent(results["Cached"], results["Uncached"])

    for repeat in range(MEMORY_REPEATS):
        order = (
            ["Cached", "Uncached"]
            if repeat % 2 == 0
            else ["Uncached", "Cached"]
        )

        results = {}

        for method in order:
            result, peak = _measure_peak_bytes(
                implementations[method],
                trajectory,
                model,
            )
            results[method] = result
            memory[method].append(peak)

            raw_records.append({
                "schedule": name,
                "measurement": "memory",
                "repeat": repeat + 1,
                "method": method,
                "seconds": "",
                "traced_peak_bytes": peak,
            })

        _assert_equivalent(results["Cached"], results["Uncached"])

    cached_seconds = float(np.median(timing["Cached"]))
    uncached_seconds = float(np.median(timing["Uncached"]))

    summary = {
        "schedule": name,
        "nodes": trajectory.n_nodes,
        "observations": trajectory.n_steps,
        "intervals": trajectory.n_steps - 1,
        "unique_intervals": int(
            np.unique(np.diff(trajectory.times)).size
        ),
        "duration": float(
            trajectory.times[-1] - trajectory.times[0]
        ),
        "timing_repeats": TIMING_REPEATS,
        "memory_repeats": MEMORY_REPEATS,
        "cached_seconds_median": cached_seconds,
        "uncached_seconds_median": uncached_seconds,
        "speedup": uncached_seconds / cached_seconds,
        "cached_traced_peak_bytes_median": float(
            np.median(memory["Cached"])
        ),
        "uncached_traced_peak_bytes_median": float(
            np.median(memory["Uncached"])
        ),
        "max_abs_offset_difference": float(
            np.max(np.abs(
                warm_results["Cached"].offset
                - warm_results["Uncached"].offset
            ))
        ),
        "absolute_nll_difference": float(
            abs(
                warm_results["Cached"].fun
                - warm_results["Uncached"].fun
            )
        ),
    }

    return summary, raw_records


# Save summary and individual measurements as plain CSV.
def _write_csv(path, records):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(records[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)


# Compare runtime and traced allocation peaks without implying total RSS.
def _plot_results(records, output_path):
    labels = [row["schedule"] for row in records]
    positions = np.arange(len(records))
    width = 0.36

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        constrained_layout=True,
    )
    fig.suptitle(
        "Fixed-K,B offset fitting: cached versus uncached\n"
        "Identical observations within each comparison"
    )

    for shift, method, color in [
        (-width / 2, "uncached", "tab:orange"),
        (width / 2, "cached", "tab:blue"),
    ]:
        seconds = [
            row[f"{method}_seconds_median"]
            for row in records
        ]
        peak_kib = [
            row[f"{method}_traced_peak_bytes_median"] / 1024.0
            for row in records
        ]

        axes[0].bar(
            positions + shift,
            seconds,
            width=width,
            label=method.capitalize(),
            color=color,
        )
        axes[1].bar(
            positions + shift,
            peak_kib,
            width=width,
            label=method.capitalize(),
            color=color,
        )

    axes[0].set(
        title="Median execution time",
        ylabel="Seconds",
    )
    axes[1].set(
        title="Median traced peak allocations",
        ylabel="KiB; not total process memory",
    )

    for ax in axes:
        ax.set_xticks(positions, labels)
        ax.tick_params(axis="x", labelrotation=15)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend()

    fig.savefig(output_path, dpi=180)


# Run the controlled benchmark and save all measurements.
def main() -> None:
    if tracemalloc.is_tracing():
        raise RuntimeError(
            "Run this benchmark without an existing tracemalloc session"
        )

    model, initial_state = _build_model()
    schedules = _build_schedules()

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "experiments" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    raw_records = []

    print("LINEAR OFFSET CACHE BENCHMARK", flush=True)
    print(f"Nodes: {model.n_nodes}")
    print(f"Intervals per schedule: {N_INTERVALS}")
    print(f"Timing repeats: {TIMING_REPEATS}")
    print(f"Memory repeats: {MEMORY_REPEATS}")
    print("Data generation is excluded from measurements.")

    for name, intervals in schedules.items():
        times = np.concatenate(([0.0], np.cumsum(intervals)))

        trajectory = simulate_linear(
            initial_state=initial_state,
            times=times,
            model=model,
            rng=np.random.default_rng(42),
        )

        print(
            f"\nBenchmarking {name}: "
            f"{np.unique(np.diff(times)).size} unique intervals...",
            flush=True,
        )

        summary, raw = _benchmark_case(name, trajectory, model)
        summaries.append(summary)
        raw_records.extend(raw)

        # Preserve completed cases if a later case fails.
        _write_csv(
            output_dir / "linear_cache_summary.csv",
            summaries,
        )
        _write_csv(
            output_dir / "linear_cache_measurements.csv",
            raw_records,
        )

        print(
            f"Cached:   {summary['cached_seconds_median']:.6f} s\n"
            f"Uncached: {summary['uncached_seconds_median']:.6f} s\n"
            f"Speedup:  {summary['speedup']:.2f}x\n"
            f"Max offset difference: "
            f"{summary['max_abs_offset_difference']:.3e}\n"
            f"Absolute NLL difference: "
            f"{summary['absolute_nll_difference']:.3e}",
            flush=True,
        )

    _plot_results(
        summaries,
        output_dir / "linear_cache.png",
    )

    print(f"\nOutputs saved under: {output_dir}")
    print(
        "Memory measurements use tracemalloc: "
        "they do not capture all native-library allocations or total RSS."
    )
    print(
        "Both implementations use the current linear_transition. "
        "This benchmark measures caching and likelihood reuse, "
        "not the earlier long-horizon numerical fix."
    )
    print(
        "The uncached function is a controlled experiment reference, "
        "not a second public fitting API."
    )
    print(
        "No minimum speedup is required; report the measured trade-off."
    )

    if plt.get_backend().lower() != "agg":
        plt.show()

    plt.close("all")


if __name__ == "__main__":
    main()