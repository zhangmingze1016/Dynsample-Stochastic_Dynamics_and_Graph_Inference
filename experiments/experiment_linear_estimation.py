"""Evaluate linear estimation and chronological prediction."""

import csv
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from dynsample.core.linear_model import LinearSDE
from dynsample.core.state import State
from dynsample.core.trajectory import Trajectory
from dynsample.estimation.linear import (
    fit_linear_drift,
    linear_negative_log_likelihood,
)
from dynsample.inference.prediction import predict_linear
from dynsample.metrics.prediction import prediction_metrics
from dynsample.simulation.linear import simulate_linear


PARAMETER_NAMES = ("k11", "k21", "k22", "b1", "b2")

# Keep the existing repeated-sample experiment enabled.
RUN_RECOVERY = True
RUN_NO_EDGE = True

# Define the same generating model for every experiment.
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
        values=np.array([[2.0], [-1.0]]),
    )

    mask = np.array([
        [True, False],
        [True, True],
    ])

    return model, initial_state, mask


# Generate a trajectory at irregular observation times.
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


# Record successful fits, failed fits, and numerical exceptions.
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
    record.update({name: np.nan for name in PARAMETER_NAMES})

    print(
        f"Starting: seed={seed}, duration={duration:.1f}, "
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


# Calculate recovery statistics using successful fits only.
def _summarize_runs(records, true_parameters):
    true_parameters = np.asarray(true_parameters, dtype=np.float64)

    if true_parameters.shape != (len(PARAMETER_NAMES),):
        raise ValueError("true_parameters must contain five values")

    if not np.all(np.isfinite(true_parameters)):
        raise ValueError("true_parameters must be finite")

    summaries = []

    for duration in sorted({row["duration"] for row in records}):
        rows = [row for row in records if row["duration"] == duration]
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
            for field in ("drift_error", "offset_error"):
                summary[f"median_{field}_successful"] = float(
                    np.median([row[field] for row in successful])
                )

        summaries.append(summary)

    return summaries


# Check summary arithmetic before running expensive experiments.
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

    rows = [example(True, 1.0, 1.0), example(True, 3.0, 3.0)]
    summary = _summarize_runs(rows, truth)[0]

    for name in PARAMETER_NAMES:
        np.testing.assert_allclose(summary[f"{name}_mean"], 2.0)
        np.testing.assert_allclose(summary[f"{name}_bias"], 2.0)
        np.testing.assert_allclose(
            summary[f"{name}_rmse"], np.sqrt(5.0)
        )

    for field in ("drift_error", "offset_error"):
        np.testing.assert_allclose(
            summary[f"median_{field}_successful"], 2.0
        )

    rows.append(example(False, 1000.0, 5.0))
    summary = _summarize_runs(rows, truth)[0]

    np.testing.assert_equal(
        [summary["attempted"], summary["successful"], summary["failed"]],
        [3, 2, 1],
    )
    np.testing.assert_allclose(summary["success_rate"], 2.0 / 3.0)
    np.testing.assert_allclose(summary["mean_seconds_all_attempts"], 3.0)

    for name in PARAMETER_NAMES:
        np.testing.assert_allclose(
            summary[f"{name}_rmse"], np.sqrt(5.0)
        )

    summary = _summarize_runs(
        [example(False, np.nan, 1.0)], truth
    )[0]

    np.testing.assert_equal(summary["successful"], 0)
    np.testing.assert_equal(summary["failed"], 1)

    for name in PARAMETER_NAMES:
        for suffix in ("mean", "bias", "rmse"):
            assert np.isnan(summary[f"{name}_{suffix}"])

    for field in ("drift_error", "offset_error"):
        assert np.isnan(summary[f"median_{field}_successful"])

    assert _summarize_runs([], truth) == []


# Save numeric records without adding a pandas dependency.
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


# Plot the original full-data fit and its estimated coefficients.
def _plot_reference(trajectory, model, results, output_path):
    default = results["Default"]

    if default is None:
        print("Reference figure skipped: default fit raised an error.")
        return

    fig, axes = plt.subplots(
        2, 2, figsize=(12, 9), constrained_layout=True
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
        title="Simulated observations", xlabel="Time", ylabel="State"
    )
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    positions = np.arange(2)
    axes[0, 1].scatter(
        positions, model.offset,
        marker="x", color="black", s=90,
        label="Generating b", zorder=5,
    )

    for shift, (name, result) in zip(
        [-0.12, 0.0, 0.12], results.items()
    ):
        if result is not None:
            label = name if result.success else f"{name} (failed)"
            axes[0, 1].scatter(
                positions + shift, result.offset, label=label
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
            matrix, cmap="coolwarm", vmin=-limit, vmax=limit
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
                    column, row, f"{matrix[row, column]:.3f}",
                    ha="center", va="center",
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


# Plot recovery across seeds and explicitly display failed attempts.
def _plot_recovery(records, true_parameters, output_path):
    durations = sorted({row["duration"] for row in records})

    fig, axes = plt.subplots(
        2, 3, figsize=(13, 8), constrained_layout=True
    )
    fig.suptitle(
        "Parameter recovery across seeds and observation durations\n"
        "Successful fits only; empirical spread is not a confidence interval"
    )

    for ax, name, truth in zip(
        axes.flat, PARAMETER_NAMES, true_parameters
    ):
        ax.axhline(
            truth, color="black", linestyle="--",
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

    successful_counts = []
    failed_counts = []

    for duration in durations:
        rows = [row for row in records if row["duration"] == duration]
        count = sum(row["success"] for row in rows)
        successful_counts.append(count)
        failed_counts.append(len(rows) - count)

    ax = axes.flat[5]
    positions = np.arange(len(durations))
    ax.bar(positions, successful_counts, label="Successful")
    ax.bar(
        positions, failed_counts,
        bottom=successful_counts,
        label="Failed", color="tab:red",
    )
    ax.set_xticks(
        positions, [f"{duration:.0f}" for duration in durations]
    )
    ax.set(title="All attempts", ylabel="Count", xlabel="Observation duration")
    ax.legend()

    fig.savefig(output_path, dpi=180)


# Preserve the original multi-start reference experiment.
def _run_reference(model, initial_state, mask, output_dir):
    trajectory = _simulate_case(
        model,
        initial_state,
        np.tile([0.3, 0.7], 100),
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

    records = []
    results = {}

    print("\nREFERENCE EXPERIMENT: seed 42", flush=True)

    for name, initial_drift in starting_points.items():
        record, result = _fit_case(
            trajectory, model, mask,
            seed=42,
            case=name,
            initial_drift=initial_drift,
        )
        records.append(record)
        results[name] = result

    true_score = linear_negative_log_likelihood(trajectory, model)
    print(f"\nReference NLL at generating parameters: {true_score:.8f}")

    for name, result in results.items():
        if result is not None:
            print(f"\n{name}: success={result.success}")
            print("Estimated K:")
            print(result.drift)
            print("Estimated b:")
            print(result.offset)
            print(f"NLL: {result.fun:.8f}")
            print(f"Message: {result.message}")

    _write_csv(output_dir / "linear_estimation_reference.csv", records)
    _plot_reference(
        trajectory, model, results,
        output_dir / "linear_estimation.png",
    )

    return trajectory


# Preserve repeated-sample recovery at two observation durations.
def _run_recovery(model, initial_state, mask, output_dir):
    true_parameters = np.array([
        model.drift[0, 0],
        model.drift[1, 0],
        model.drift[1, 1],
        model.offset[0],
        model.offset[1],
    ])

    records = []
    print("\nREPEATED-SAMPLE EXPERIMENT", flush=True)

    for pair_count in (100, 300):
        for seed in range(10):
            trajectory = _simulate_case(
                model,
                initial_state,
                np.tile([0.3, 0.7], pair_count),
                seed,
            )
            record, _ = _fit_case(
                trajectory, model, mask, seed, "Default"
            )
            records.append(record)

            _write_csv(
                output_dir / "linear_estimation_runs.csv", records
            )

    summaries = _summarize_runs(records, true_parameters)

    print("\nSUMMARY")
    print("Bias and RMSE are conditional on successful fits.")

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
            f"{'Parameter':<12}{'Mean':>12}"
            f"{'Bias':>12}{'RMSE':>12}"
        )

        for name in PARAMETER_NAMES:
            print(
                f"{name:<12}"
                f"{summary[f'{name}_mean']:>12.6f}"
                f"{summary[f'{name}_bias']:>12.6f}"
                f"{summary[f'{name}_rmse']:>12.6f}"
            )

    _write_csv(
        output_dir / "linear_estimation_summary.csv", summaries
    )
    _plot_recovery(
        records, true_parameters,
        output_dir / "linear_estimation_recovery.png",
    )


# Split observations chronologically without overlapping scored targets.
def _split_trajectory(trajectory, train_fraction=0.7):
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie between zero and one")

    split = int(np.floor(trajectory.n_steps * train_fraction))

    if split < 2 or split >= trajectory.n_steps:
        raise ValueError(
            "split must leave at least two training observations "
            "and one validation observation"
        )

    training = Trajectory(
        times=trajectory.times[:split],
        values=trajectory.values[:split],
    )
    validation = Trajectory(
        times=trajectory.times[split:],
        values=trajectory.values[split:],
    )

    return training, validation, split


# Predict each validation state using only the preceding observed state.
def _one_step_predictions(trajectory, split, model):
    times = trajectory.times[split:]
    means = np.empty_like(trajectory.values[split:])

    for position, index in enumerate(range(split, trajectory.n_steps)):
        previous_state = trajectory.state_at(index - 1)

        prediction, _ = predict_linear(
            initial_state=previous_state,
            times=trajectory.times[index:index + 1],
            model=model,
        )

        means[position] = prediction.values[0]

    return Trajectory(times=times, values=means)


# Convert metric arrays into CSV rows for overall and per-node scores.
def _score_predictions(observed, predictions):
    records = []

    for name, predicted in predictions.items():
        scores = prediction_metrics(observed, predicted)

        records.append({
            "method": name,
            "scope": "overall",
            "rmse": float(scores["rmse"]),
            "mae": float(scores["mae"]),
        })

        for node in range(observed.n_nodes):
            records.append({
                "method": name,
                "scope": f"node_{node + 1}",
                "rmse": float(scores["rmse_per_node"][node]),
                "mae": float(scores["mae_per_node"][node]),
            })

    return records


# Plot fixed-origin and observation-updated predictions separately.
def _plot_predictions(
    trajectory,
    split,
    predictions,
    fixed_covariances,
    output_path,
):
    times = trajectory.times[split:]
    origin_time = trajectory.times[split - 1]

    # Marginal process uncertainty under fixed fitted parameters.
    variances = np.diagonal(
        fixed_covariances, axis1=1, axis2=2
    )
    standard_deviations = np.sqrt(np.maximum(variances, 0.0))

    fig, axes = plt.subplots(
        trajectory.n_nodes,
        2,
        figsize=(14, 4 * trajectory.n_nodes),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(
        "Chronological validation: parameters fitted on training data only\n"
        "Left: fixed origin; right: previous observation available"
    )

    # Include a short section of training data for visual context.
    start = max(0, split - 20)

    for node in range(trajectory.n_nodes):
        for column in range(2):
            ax = axes[node, column]

            ax.plot(
                trajectory.times[start:],
                trajectory.values[start:, node, 0],
                color="black",
                linewidth=1.0,
                label="Observed",
            )
            ax.axvline(
                origin_time,
                color="gray",
                linestyle=":",
                label="Last training observation",
            )
            ax.set(
                xlabel="Time",
                ylabel=f"Node {node + 1} state",
            )
            ax.grid(alpha=0.3)

        ax = axes[node, 0]
        mean = predictions["Fixed-origin model"].values[:, node, 0]
        width = 1.96 * standard_deviations[:, node]

        ax.fill_between(
            times,
            mean - width,
            mean + width,
            alpha=0.18,
            color="tab:blue",
            label="Pointwise 95% process interval; fitted parameters fixed",
        )
        ax.plot(
            times, mean,
            color="tab:blue",
            label="Fixed-origin mean",
        )
        ax.plot(
            times,
            predictions["Fixed-origin persistence"].values[:, node, 0],
            color="tab:orange",
            linestyle="--",
            label="Fixed-origin persistence",
        )
        ax.set_title(f"Node {node + 1}: no validation updates")
        ax.legend(fontsize=7)

        ax = axes[node, 1]
        ax.plot(
            times,
            predictions["One-step model"].values[:, node, 0],
            color="tab:blue",
            label="One-step mean",
        )
        ax.plot(
            times,
            predictions["One-step persistence"].values[:, node, 0],
            color="tab:orange",
            linestyle="--",
            label="Previous observation",
        )
        ax.set_title(f"Node {node + 1}: updated one-step predictions")
        ax.legend(fontsize=8)

    fig.savefig(output_path, dpi=180)


# Fit on training observations and evaluate on later observations only.
def _run_prediction_experiment(trajectory, model, mask, output_dir):
    training, validation, split = _split_trajectory(trajectory)

    print("\nCHRONOLOGICAL PREDICTION EXPERIMENT", flush=True)
    print(f"Training observations: {training.n_steps}")
    print(f"Validation observations: {validation.n_steps}")
    print(f"Last training time: {training.times[-1]:.4f}")
    print(f"First validation time: {validation.times[0]:.4f}")
    print("Diffusion and connection mask are supplied and fixed.")

    # This is a fresh fit. Full-data reference estimates are not reused.
    record, result = _fit_case(
        trajectory=training,
        model=model,
        mask=mask,
        seed=42,
        case="Training only",
    )
    _write_csv(
        output_dir / "linear_prediction_fit.csv", [record]
    )

    if result is None or not result.success:
        print("Prediction evaluation skipped: training fit was unsuccessful.")
        return

    fitted_model = result.model
    origin = training.state_at(training.n_steps - 1)

    # All future means are conditioned on the same last training state.
    fixed_prediction, fixed_covariances = predict_linear(
        initial_state=origin,
        times=validation.times,
        model=fitted_model,
    )

    # Each mean uses the latest observation, without refitting parameters.
    one_step_prediction = _one_step_predictions(
        trajectory, split, fitted_model
    )

    # A fair fixed-origin baseline never observes validation states.
    fixed_persistence = Trajectory(
        times=validation.times,
        values=np.repeat(
            origin.values[np.newaxis, :, :],
            validation.n_steps,
            axis=0,
        ),
    )

    # A fair one-step baseline uses exactly the preceding observation.
    one_step_persistence = Trajectory(
        times=validation.times,
        values=trajectory.values[split - 1:-1],
    )

    predictions = {
        "Fixed-origin model": fixed_prediction,
        "Fixed-origin persistence": fixed_persistence,
        "One-step model": one_step_prediction,
        "One-step persistence": one_step_persistence,
    }

    scores = _score_predictions(validation, predictions)
    _write_csv(
        output_dir / "linear_prediction_metrics.csv", scores
    )

    print("\nVALIDATION SCORES")
    print(f"{'Method':<28}{'Scope':<12}{'RMSE':>12}{'MAE':>12}")

    for row in scores:
        print(
            f"{row['method']:<28}"
            f"{row['scope']:<12}"
            f"{row['rmse']:>12.6f}"
            f"{row['mae']:>12.6f}"
        )

    # Include the last training state as the conditioning state.
    # Only transitions into validation observations contribute to this NLL.
    validation_path = Trajectory(
        times=trajectory.times[split - 1:],
        values=trajectory.values[split - 1:],
    )

    nll_rows = []
    for name, candidate in [
        ("Training fit", fitted_model),
        ("Generating model", model),
    ]:
        score = linear_negative_log_likelihood(
            trajectory=validation_path,
            model=candidate,
        )
        nll_rows.append({
            "model": name,
            "validation_transitions": validation.n_steps,
            "conditional_nll": float(score),
            "nll_per_transition": float(score / validation.n_steps),
        })

    _write_csv(
        output_dir / "linear_prediction_nll.csv", nll_rows
    )

    print("\nVALIDATION CONDITIONAL PATH NLL")
    for row in nll_rows:
        print(
            f"{row['model']:<20}"
            f"NLL={row['conditional_nll']:.6f}, "
            f"per transition={row['nll_per_transition']:.6f}"
        )

    # Save observations and every predicted mean for reproducibility.
    prediction_rows = []

    for index, time in enumerate(validation.times):
        for node in range(validation.n_nodes):
            row = {
                "time": float(time),
                "node": node + 1,
                "observed": float(validation.values[index, node, 0]),
            }
            for name, predicted in predictions.items():
                row[name] = float(predicted.values[index, node, 0])

            row["fixed_origin_variance"] = float(
                fixed_covariances[index, node, node]
            )
            prediction_rows.append(row)

    _write_csv(
        output_dir / "linear_prediction_values.csv",
        prediction_rows,
    )
    _plot_predictions(
        trajectory,
        split,
        predictions,
        fixed_covariances,
        output_dir / "linear_prediction.png",
    )

    print(
        "\nCompare each model with the persistence baseline "
        "using the same information."
    )
    print(
        "Prediction intervals include process noise only; "
        "parameter uncertainty is not included."
    )
    print(
        "One validation trajectory does not establish "
        "general predictive superiority."
    )


# Summarize unrestricted estimates of coefficients whose true values are zero.
def _summarize_no_edge_runs(records):
    summaries = []

    for duration in sorted({row["duration"] for row in records}):
        rows = [
            row for row in records
            if row["duration"] == duration
        ]
        successful = [
            row for row in rows
            if row["success"]
        ]

        for coefficient, direction in [
            ("k12", "Node 2 -> Node 1"),
            ("k21", "Node 1 -> Node 2"),
        ]:
            summary = {
                "duration": duration,
                "coefficient": coefficient,
                "direction": direction,
                "true_value": 0.0,
                "attempted": len(rows),
                "successful": len(successful),
                "failed": len(rows) - len(successful),
                "mean": np.nan,
                "rmse": np.nan,
                "mean_absolute": np.nan,
                "max_absolute": np.nan,
            }

            if successful:
                estimates = np.array([
                    row[coefficient] for row in successful
                ])

                # The true value is zero, so each estimate is also its error.
                summary["mean"] = float(np.mean(estimates))
                summary["rmse"] = float(
                    np.sqrt(np.mean(estimates**2))
                )
                summary["mean_absolute"] = float(
                    np.mean(np.abs(estimates))
                )
                summary["max_absolute"] = float(
                    np.max(np.abs(estimates))
                )

            summaries.append(summary)

    return summaries


# Show off-diagonal estimates and retain visible failure counts.
def _plot_no_edge_runs(records, output_path):
    durations = sorted({
        row["duration"] for row in records
    })

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4.5),
        constrained_layout=True,
    )
    fig.suptitle(
        "Independent-node control: unrestricted drift fitting\n"
        "True cross-node coefficients are zero; no edge selection is applied"
    )

    for ax, coefficient, direction in [
        (axes[0], "k12", "Node 2 -> Node 1"),
        (axes[1], "k21", "Node 1 -> Node 2"),
    ]:
        ax.axhline(
            0.0,
            color="black",
            linestyle="--",
            label="True coefficient",
        )

        for position, duration in enumerate(durations):
            successful = [
                row for row in records
                if row["duration"] == duration and row["success"]
            ]

            if successful:
                jitter = np.linspace(
                    -0.12, 0.12, len(successful)
                )

                ax.scatter(
                    position + jitter,
                    [row[coefficient] for row in successful],
                    alpha=0.8,
                )

        ax.set_xticks(
            range(len(durations)),
            [f"{duration:.0f}" for duration in durations],
        )
        ax.set(
            title=f"{coefficient}: {direction}",
            xlabel="Observation duration",
            ylabel="Estimated drift coefficient",
        )
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

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

    axes[2].bar(
        positions,
        successful_counts,
        label="Successful",
    )
    axes[2].bar(
        positions,
        failed_counts,
        bottom=successful_counts,
        color="tab:red",
        label="Failed",
    )
    axes[2].set_xticks(
        positions,
        [f"{duration:.0f}" for duration in durations],
    )
    axes[2].set(
        title="All fitting attempts",
        xlabel="Observation duration",
        ylabel="Count",
    )
    axes[2].legend()

    fig.savefig(output_path, dpi=180)


# Fit a dense drift matrix to data generated without cross-node coupling.
def _run_no_edge_experiment(output_dir):
    model = LinearSDE(
        drift=np.array([
            [-0.7, 0.0],
            [0.0, -1.0],
        ]),
        offset=np.array([0.4, -0.2]),
        diffusion=np.array([
            [0.6, 0.0],
            [0.0, 0.5],
        ]),
    )

    initial_state = State(
        time=0.0,
        values=np.array([[2.0], [-1.0]]),
    )

    records = []
    runs_path = output_dir / "linear_no_edge_runs.csv"

    print("\nINDEPENDENT-NODE CONTROL", flush=True)
    print("Generating K and B are diagonal.")
    print("Fitting allows every drift coefficient: mask=None.")
    print("Diffusion B is supplied and fixed.")
    print("No threshold or sparse edge selection is applied.")

    for pair_count in (100, 300):
        intervals = np.tile([0.3, 0.7], pair_count)

        for seed in range(10):
            trajectory = _simulate_case(
                model=model,
                initial_state=initial_state,
                intervals=intervals,
                seed=seed,
            )

            record, result = _fit_case(
                trajectory=trajectory,
                model=model,
                mask=None,
                seed=seed,
                case="Independent nodes, unrestricted fit",
            )

            # The original reference records k21 but not k12.
            # Add the other direction for this unrestricted experiment.
            record["k12"] = (
                float(result.drift[0, 1])
                if result is not None
                else np.nan
            )

            records.append(record)

            # Preserve every completed attempt, including failed fits.
            _write_csv(runs_path, records)

    summaries = _summarize_no_edge_runs(records)

    print("\nNO-EDGE SUMMARY")
    print("Coefficient statistics use successful fits only.")
    print(
        f"{'Duration':>10}"
        f"{'Term':>8}"
        f"{'Success':>10}"
        f"{'Failed':>9}"
        f"{'Mean':>12}"
        f"{'RMSE':>12}"
        f"{'Mean abs':>12}"
        f"{'Max abs':>12}"
    )

    for row in summaries:
        print(
            f"{row['duration']:>10.0f}"
            f"{row['coefficient']:>8}"
            f"{row['successful']:>10}"
            f"{row['failed']:>9}"
            f"{row['mean']:>12.6f}"
            f"{row['rmse']:>12.6f}"
            f"{row['mean_absolute']:>12.6f}"
            f"{row['max_absolute']:>12.6f}"
        )

    _write_csv(
        output_dir / "linear_no_edge_summary.csv",
        summaries,
    )
    _plot_no_edge_runs(
        records,
        output_dir / "linear_no_edge.png",
    )

    print(
        "\nNonzero estimates are not automatically detected edges. "
        "This experiment measures coefficient error under a no-edge model."
    )
    print(
        "Ten seeds provide an initial control, not a calibrated "
        "threshold or a false-positive rate."
    )
    print(
        "Paired seeds at the two durations extend the same simulated paths; "
        "the duration comparisons are dependent."
    )



# Run estimation, prediction, recovery, and the independent-node control.
def main() -> None:
    _check_summary()
    print(
        "Summary checks passed: arithmetic, failed fits, and no successes.",
        flush=True,
    )

    model, initial_state, mask = _build_reference_model()

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "experiments" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory = _run_reference(
        model, initial_state, mask, output_dir
    )

    _run_prediction_experiment(
        trajectory, model, mask, output_dir
    )

    if RUN_RECOVERY:
        _run_recovery(model, initial_state, mask, output_dir)

    if RUN_NO_EDGE:
        _run_no_edge_experiment(output_dir)

    print(f"\nOutputs saved under: {output_dir}")
    print(
        "Diffusion is supplied and fixed. The reference fit uses a mask; "
        "the no-edge control fits an unrestricted drift matrix. "
        "Neither experiment performs sparse edge selection."
    )
    print(
        "Ten recovery seeds provide an initial diagnostic, "
        "not a guarantee of accuracy or interval coverage."
    )

    if plt.get_backend().lower() != "agg":
        plt.show()

    plt.close("all")


if __name__ == "__main__":
    main()