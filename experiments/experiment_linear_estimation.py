"""Evaluate linear estimation across starts, seeds, and durations."""

import csv
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from dynsample.core.linear_model import LinearSDE
from dynsample.core.state import State
from dynsample.estimation.linear import (
    fit_linear_drift,
    linear_negative_log_likelihood,
)
from dynsample.simulation.linear import simulate_linear


PARAMETER_NAMES = ("k11", "k21", "k22", "b1", "b2")


# Keep the reference system identical across all experimental conditions.
def _build_reference_model():
    model = LinearSDE(
        drift=np.array([
            [-0.7, 0.0],
            [0.5, -1.0],
        ]),
        offset=np.array([0.4, -0.2]),
        diffusion=np.array([
            [0.6, 0.0],
            [0.0, 0.5],
        ]),
    )

    initial_state = State(
        time=0.0,
        values=np.array([
            [2.0],
            [-1.0],
        ]),
    )

    mask = np.array([
        [True, False],
        [True, True],
    ])

    return model, initial_state, mask


# Generate observations on a supplied irregular sampling schedule.
def _simulate_case(model, initial_state, intervals, seed):
    times = np.concatenate((
        [initial_state.time],
        initial_state.time + np.cumsum(intervals),
    ))

    return simulate_linear(
        initial_state=initial_state,
        times=times,
        model=model,
        rng=np.random.default_rng(seed),
    )


# Record one attempt, including finite candidates from unsuccessful fits.
def _fit_case(
    trajectory,
    model,
    mask,
    seed,
    case,
    initial_drift=None,
):
    duration = float(trajectory.times[-1] - trajectory.times[0])

    record = {
        "seed": int(seed),
        "duration": duration,
        "observations": trajectory.n_steps,
        "case": case,
        "success": False,
        "is_stable": None,
        "seconds": np.nan,
        "initial_nll": np.nan,
        "final_nll": np.nan,
        "drift_error": np.nan,
        "offset_error": np.nan,
        "invalid_evaluations": None,
        "message": "",
        "last_invalid_reason": "",
    }

    for name in PARAMETER_NAMES:
        record[name] = np.nan

    print(
        f"Starting: seed={seed}, duration={duration:.0f}, "
        f"case={case}",
        flush=True,
    )

    start = perf_counter()
    result = None

    try:
        result = fit_linear_drift(
            trajectory=trajectory,
            diffusion=model.diffusion,
            drift_mask=mask,
            initial_drift=initial_drift,
            maxiter=150,
        )

        record.update({
            "success": bool(result.success),
            "is_stable": bool(result.is_stable),
            "initial_nll": float(result.initial_fun),
            "final_nll": float(result.fun),
            "drift_error": float(
                np.linalg.norm(result.drift - model.drift)
            ),
            "offset_error": float(
                np.linalg.norm(result.offset - model.offset)
            ),
            "invalid_evaluations": int(result.invalid_evaluations),
            "message": str(result.message),
            "last_invalid_reason": result.last_invalid_reason or "",
            "k11": float(result.drift[0, 0]),
            "k21": float(result.drift[1, 0]),
            "k22": float(result.drift[1, 1]),
            "b1": float(result.offset[0]),
            "b2": float(result.offset[1]),
        })

    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        record["message"] = f"{type(exc).__name__}: {exc}"

    record["seconds"] = perf_counter() - start

    print(
        f"Finished: success={record['success']}, "
        f"NLL={record['final_nll']:.6f}, "
        f"seconds={record['seconds']:.2f}",
        flush=True,
    )

    if not record["success"]:
        print(f"  Reason: {record['message']}", flush=True)

    return record, result


# Summarize successful estimates while explicitly counting all failures.
def _summarize_runs(records, true_parameters):
    true_parameters = np.asarray(true_parameters, dtype=np.float64)
    if true_parameters.shape != (len(PARAMETER_NAMES),):
        raise ValueError("true_parameters must contain five values")
    if not np.all(np.isfinite(true_parameters)):
        raise ValueError("true_parameters must be finite")

    summaries = []
    durations = sorted({row["duration"] for row in records})

    for duration in durations:
        rows = [
            row for row in records
            if row["duration"] == duration
        ]
        successful = [row for row in rows if row["success"]]

        summary = {
            "duration": duration,
            "attempted": len(rows),
            "successful": len(successful),
            "failed": len(rows) - len(successful),
            "success_rate": len(successful) / len(rows),
            "mean_seconds_all_attempts": float(
                np.mean([row["seconds"] for row in rows])
            ),
            "median_drift_error_successful": np.nan,
            "median_offset_error_successful": np.nan,
        }

        for name, truth in zip(PARAMETER_NAMES, true_parameters):
            summary[f"{name}_mean"] = np.nan
            summary[f"{name}_bias"] = np.nan
            summary[f"{name}_rmse"] = np.nan

            if successful:
                estimates = np.array([
                    row[name] for row in successful
                ])
                errors = estimates - truth

                summary[f"{name}_mean"] = float(np.mean(estimates))
                summary[f"{name}_bias"] = float(np.mean(errors))
                summary[f"{name}_rmse"] = float(
                    np.sqrt(np.mean(errors**2))
                )

        if successful:
            summary["median_drift_error_successful"] = float(
                np.median([
                    row["drift_error"] for row in successful
                ])
            )
            summary["median_offset_error_successful"] = float(
                np.median([
                    row["offset_error"] for row in successful
                ])
            )

        summaries.append(summary)

    return summaries


# Check summary arithmetic and failure handling before expensive fitting.
def _check_summary():
    truth = np.zeros(len(PARAMETER_NAMES))

    def example(success, value, seconds):
        row = {
            "duration": 100.0,
            "success": success,
            "seconds": seconds,
            "drift_error": abs(value),
            "offset_error": abs(value),
        }
        row.update({name: value for name in PARAMETER_NAMES})
        return row

    # Successful errors 1 and 3 give mean/bias 2 and RMSE sqrt(5).
    rows = [example(True, 1.0, 1.0), example(True, 3.0, 3.0)]
    summary = _summarize_runs(rows, truth)[0]
    for name in PARAMETER_NAMES:
        np.testing.assert_allclose(summary[f"{name}_mean"], 2.0)
        np.testing.assert_allclose(summary[f"{name}_bias"], 2.0)
        np.testing.assert_allclose(summary[f"{name}_rmse"], np.sqrt(5.0))
    np.testing.assert_allclose(summary["median_drift_error_successful"], 2.0)
    np.testing.assert_allclose(summary["median_offset_error_successful"], 2.0)

    # A failed finite candidate is counted but excluded from accuracy statistics.
    rows.append(example(False, 1000.0, 5.0))
    summary = _summarize_runs(rows, truth)[0]
    np.testing.assert_equal(
        [summary["attempted"], summary["successful"], summary["failed"]],
        [3, 2, 1],
    )
    np.testing.assert_allclose(summary["success_rate"], 2.0 / 3.0)
    np.testing.assert_allclose(summary["mean_seconds_all_attempts"], 3.0)
    for name in PARAMETER_NAMES:
        np.testing.assert_allclose(summary[f"{name}_rmse"], np.sqrt(5.0))

    # No successful estimates must yield unavailable accuracy, not zero error.
    summary = _summarize_runs([example(False, np.nan, 1.0)], truth)[0]
    np.testing.assert_equal(summary["successful"], 0)
    np.testing.assert_equal(summary["failed"], 1)
    for name in PARAMETER_NAMES:
        for suffix in ("mean", "bias", "rmse"):
            np.testing.assert_equal(np.isnan(summary[f"{name}_{suffix}"]), True)
    for field in ("median_drift_error_successful", "median_offset_error_successful"):
        np.testing.assert_equal(np.isnan(summary[field]), True)
    np.testing.assert_equal(_summarize_runs([], truth), [])


# Write plain numeric and diagnostic records without requiring pandas.
def _write_csv(path, rows):
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


# Preserve the original single-trajectory coefficient comparison.
def _plot_reference(trajectory, model, results, output_path):
    default = results["Default"]

    if default is None:
        print("Reference figure skipped: default fit raised an error.")
        return

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 9),
        constrained_layout=True,
    )

    fig.suptitle(
        "Known-structure linear estimation: seed 42\n"
        f"Default optimizer success: {default.success}; fixed diffusion"
    )

    for node in range(2):
        axes[0, 0].plot(
            trajectory.times,
            trajectory.values[:, node, 0],
            label=f"Node {node + 1}",
            linewidth=1.0,
        )

    axes[0, 0].set(
        title="Simulated observations",
        xlabel="Time",
        ylabel="State",
    )
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    positions = np.arange(2)
    axes[0, 1].scatter(
        positions,
        model.offset,
        marker="x",
        color="black",
        s=90,
        label="Generating b",
        zorder=5,
    )

    for shift, (name, result) in zip(
        [-0.12, 0.0, 0.12],
        results.items(),
    ):
        if result is not None:
            label = name if result.success else f"{name} (failed)"
            axes[0, 1].scatter(
                positions + shift,
                result.offset,
                label=label,
            )

    axes[0, 1].set_xticks(positions, ["Node 1", "Node 2"])
    axes[0, 1].set(title="Offset estimates", ylabel="b")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    limit = max(
        float(np.max(np.abs(model.drift))),
        float(np.max(np.abs(default.drift))),
        1e-12,
    )

    for ax, matrix, title in [
        (axes[1, 0], model.drift, "Generating K"),
        (axes[1, 1], default.drift, "Estimated K: default"),
    ]:
        heatmap = ax.imshow(
            matrix,
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        ax.set(
            title=title,
            xlabel="Source node",
            ylabel="Receiving node",
        )
        ax.set_xticks([0, 1], ["Node 1", "Node 2"])
        ax.set_yticks([0, 1], ["Node 1", "Node 2"])

        for row in range(2):
            for column in range(2):
                ax.text(
                    column,
                    row,
                    f"{matrix[row, column]:.3f}",
                    ha="center",
                    va="center",
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.7,
                        "edgecolor": "none",
                    },
                )

    fig.colorbar(
        heatmap,
        ax=[axes[1, 0], axes[1, 1]],
        label="Drift coefficient",
        shrink=0.8,
    )
    fig.savefig(output_path, dpi=180)


# Show empirical parameter spread; these are not confidence intervals.
def _plot_recovery(records, true_parameters, output_path):
    durations = sorted({row["duration"] for row in records})

    fig, axes = plt.subplots(
        2, 3,
        figsize=(13, 8),
        constrained_layout=True,
    )
    fig.suptitle(
        "Parameter recovery across seeds and observation durations\n"
        "Successful fits only; empirical spread is not a confidence interval"
    )

    for ax, name, truth in zip(
        axes.flat,
        PARAMETER_NAMES,
        true_parameters,
    ):
        ax.axhline(
            truth,
            color="black",
            linestyle="--",
            label="Generating value",
        )

        for position, duration in enumerate(durations):
            rows = [
                row for row in records
                if row["duration"] == duration and row["success"]
            ]

            if rows:
                jitter = np.linspace(-0.12, 0.12, len(rows))
                ax.scatter(
                    position + jitter,
                    [row[name] for row in rows],
                    alpha=0.8,
                )

        ax.set_xticks(
            range(len(durations)),
            [f"{duration:.0f}" for duration in durations],
        )
        ax.set(
            title=name,
            xlabel="Observation duration",
            ylabel="Estimate",
        )
        ax.grid(alpha=0.3)

    axes.flat[0].legend(fontsize=8)

    # Show failure counts so excluded estimates remain visible.
    ax = axes.flat[5]

    successful_counts = []
    failed_counts = []

    for duration in durations:
        rows = [
            row for row in records
            if row["duration"] == duration
        ]
        count = sum(row["success"] for row in rows)
        successful_counts.append(count)
        failed_counts.append(len(rows) - count)

    positions = np.arange(len(durations))

    ax.bar(
        positions,
        successful_counts,
        label="Successful",
    )
    ax.bar(
        positions,
        failed_counts,
        bottom=successful_counts,
        label="Failed",
        color="tab:red",
    )

    ax.set_xticks(
        positions,
        [f"{duration:.0f}" for duration in durations],
    )
    ax.set(
        title="All attempts",
        xlabel="Observation duration",
        ylabel="Count",
    )
    ax.legend()

    fig.savefig(output_path, dpi=180)


# Run the reference example and the repeated-sample evaluation.
def main() -> None:
    _check_summary()
    print("Summary checks passed: arithmetic, failed fits, and no successes.", flush=True)

    model, initial_state, mask = _build_reference_model()

    true_parameters = np.array([
        model.drift[0, 0],
        model.drift[1, 0],
        model.drift[1, 1],
        model.offset[0],
        model.offset[1],
    ])

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "experiments" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("REFERENCE EXPERIMENT: seed 42", flush=True)

    reference_trajectory = _simulate_case(
        model=model,
        initial_state=initial_state,
        intervals=np.tile([0.3, 0.7], 100),
        seed=42,
    )

    starting_points = {
        "Default": None,
        "Diagonal": np.array([
            [-0.4, 0.0],
            [0.0, -0.4],
        ]),
        "Opposite coupling": np.array([
            [-1.5, 0.0],
            [-0.4, -1.2],
        ]),
    }

    reference_records = []
    reference_results = []

    for name, initial_drift in starting_points.items():
        record, result = _fit_case(
            trajectory=reference_trajectory,
            model=model,
            mask=mask,
            seed=42,
            case=name,
            initial_drift=initial_drift,
        )
        reference_records.append(record)
        reference_results.append((name, result))

    reference_results = dict(reference_results)

    true_score = linear_negative_log_likelihood(
        trajectory=reference_trajectory,
        model=model,
    )
    print(f"\nReference NLL at generating parameters: {true_score:.8f}")

    for name, result in reference_results.items():
        if result is not None:
            print(f"\n{name}: success={result.success}")
            print("Estimated K:")
            print(result.drift)
            print("Estimated b:")
            print(result.offset)
            print(f"NLL: {result.fun:.8f}")
            print(f"Message: {result.message}")

    _write_csv(
        output_dir / "linear_estimation_reference.csv",
        reference_records,
    )
    _plot_reference(
        reference_trajectory,
        model,
        reference_results,
        output_dir / "linear_estimation.png",
    )

    print("\nREPEATED-SAMPLE EXPERIMENT", flush=True)

    records = []
    seeds = range(10)
    pair_counts = (100, 300)
    runs_path = output_dir / "linear_estimation_runs.csv"

    for pair_count in pair_counts:
        intervals = np.tile([0.3, 0.7], pair_count)

        for seed in seeds:
            trajectory = _simulate_case(
                model=model,
                initial_state=initial_state,
                intervals=intervals,
                seed=seed,
            )

            record, _ = _fit_case(
                trajectory=trajectory,
                model=model,
                mask=mask,
                seed=seed,
                case="Default",
            )
            records.append(record)

            # Save progress after every attempt.
            _write_csv(runs_path, records)

    summaries = _summarize_runs(records, true_parameters)

    print("\nSUMMARY")
    print("Bias and RMSE below are conditional on successful fits.")

    for summary in summaries:
        print(
            f"\nDuration={summary['duration']:.0f}, "
            f"attempted={summary['attempted']}, "
            f"successful={summary['successful']}, "
            f"failed={summary['failed']}"
        )
        print(
            f"Mean seconds per attempt: "
            f"{summary['mean_seconds_all_attempts']:.2f}"
        )
        print(
            f"{'Parameter':<12}"
            f"{'Mean':>12}"
            f"{'Bias':>12}"
            f"{'RMSE':>12}"
        )

        for name in PARAMETER_NAMES:
            print(
                f"{name:<12}"
                f"{summary[f'{name}_mean']:>12.6f}"
                f"{summary[f'{name}_bias']:>12.6f}"
                f"{summary[f'{name}_rmse']:>12.6f}"
            )

    _write_csv(
        output_dir / "linear_estimation_summary.csv",
        summaries,
    )
    _plot_recovery(
        records,
        true_parameters,
        output_dir / "linear_estimation_recovery.png",
    )

    print(f"\nOutputs saved under: {output_dir}")
    print(
        "Ten seeds provide an initial diagnostic, not a guarantee "
        "of parameter accuracy or confidence-interval coverage."
    )

    if "agg" != plt.get_backend().lower():
        plt.show()

    plt.close("all")


if __name__ == "__main__":
    main()