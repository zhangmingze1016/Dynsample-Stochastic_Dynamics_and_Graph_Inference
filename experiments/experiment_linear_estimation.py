"""Recover a two-node linear drift under a known connection mask."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dynsample.core.linear_model import LinearSDE
from dynsample.core.state import State
from dynsample.estimation.linear import (
    fit_linear_drift,
    linear_negative_log_likelihood,
)
from dynsample.simulation.linear import simulate_linear


# Simulate a known system, estimate its coefficients, and compare starts.
def main() -> None:
    # 1. Define the data-generating model.
    true_drift = np.array([
        [-0.7, 0.0],
        [0.5, -1.0],
    ])
    true_offset = np.array([0.4, -0.2])
    diffusion = np.array([
        [0.6, 0.0],
        [0.0, 0.5],
    ])

    true_model = LinearSDE(
        drift=true_drift,
        offset=true_offset,
        diffusion=diffusion,
    )

    # Rows are receiving nodes; columns are source nodes.
    drift_mask = np.array([
        [True, False],
        [True, True],
    ])

    # 2. Generate 201 observations over 100 time units.
    intervals = np.tile([0.3, 0.7], 100)
    times = np.concatenate(([0.0], np.cumsum(intervals)))

    initial_state = State(
        time=float(times[0]),
        values=np.array([
            [2.0],
            [-1.0],
        ]),
    )

    trajectory = simulate_linear(
        initial_state=initial_state,
        times=times,
        model=true_model,
        rng=np.random.default_rng(42),
    )
    values = trajectory.values[:, :, 0]

    # 3. Use the same observations with three different starting points.
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

    results = {}

    print("LINEAR DRIFT ESTIMATION")
    print("Allowed direct connection: node 1 -> node 2")
    print("Diffusion B is supplied and fixed.")
    print(f"Observations: {times.size}")
    print(f"Duration: {times[-1] - times[0]:.4f}")
    print(flush=True)

    for name, initial_drift in starting_points.items():
        print(f"Fitting: {name} ...", flush=True)

        result = fit_linear_drift(
            trajectory=trajectory,
            diffusion=diffusion,
            drift_mask=drift_mask,
            initial_drift=initial_drift,
            maxiter=150,
        )
        results[name] = result

    # 4. Print numerical comparisons.
    true_score = linear_negative_log_likelihood(
        trajectory=trajectory,
        model=true_model,
    )

    print("\nTRUE COEFFICIENTS")
    print("K:")
    print(true_drift)
    print("b:")
    print(true_offset)
    print(f"NLL at generating parameters: {true_score:.8f}")

    print("\nSTARTING-POINT COMPARISON")
    print(
        f"{'Case':<20}"
        f"{'Initial NLL':>15}"
        f"{'Final NLL':>15}"
        f"{'Success':>10}"
        f"{'Stable':>10}"
    )

    for name, result in results.items():
        print(
            f"{name:<20}"
            f"{result.initial_fun:>15.6f}"
            f"{result.fun:>15.6f}"
            f"{str(result.success):>10}"
            f"{str(result.is_stable):>10}"
        )

    default_result = results["Default"]

    print("\nCOEFFICIENTS AND DIAGNOSTICS")
    for name, result in results.items():
        print(f"\n{name}")
        print("Estimated K:")
        print(result.drift)
        print("Estimated b:")
        print(result.offset)
        print(f"Stopping reason: {result.message}")
        print(f"Spectral abscissa: {result.spectral_abscissa:.8f}")
        print(f"Invalid trial evaluations: {result.invalid_evaluations}")

        if result.last_invalid_reason is not None:
            print(f"Last invalid trial: {result.last_invalid_reason}")

        drift_error = np.linalg.norm(result.drift - true_drift)
        offset_error = np.linalg.norm(result.offset - true_offset)
        difference_from_default = np.linalg.norm(
            result.drift - default_result.drift
        )

        print(f"Drift Frobenius error: {drift_error:.8f}")
        print(f"Offset Euclidean error: {offset_error:.8f}")
        print(
            "Drift difference from default fit: "
            f"{difference_from_default:.8f}"
        )
        print(
            "NLL difference from default fit: "
            f"{result.fun - default_result.fun:.8f}"
        )

    # 5. Plot the default fit, even if its status needs inspection.
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
        constrained_layout=True,
    )

    status = "success" if default_result.success else "inspect stopping message"
    fig.suptitle(
        "Linear SDE estimation with a known connection mask\n"
        f"Default fit: {status}; diffusion fixed"
    )

    # Observed paths.
    ax = axes[0, 0]

    for node in range(2):
        ax.plot(
            times,
            values[:, node],
            linewidth=1.0,
            label=f"Node {node + 1}",
        )

    ax.set_title("Simulated observations")
    ax.set_xlabel("Time")
    ax.set_ylabel("State")
    ax.grid(alpha=0.3)
    ax.legend()

    # Compare offsets from every starting point.
    ax = axes[0, 1]
    positions = np.arange(2)

    ax.scatter(
        positions,
        true_offset,
        color="black",
        marker="x",
        s=90,
        label="Generating b",
        zorder=5,
    )

    shifts = [-0.12, 0.0, 0.12]

    for shift, (name, result) in zip(shifts, results.items()):
        ax.scatter(
            positions + shift,
            result.offset,
            label=name,
        )

    ax.set_xticks(positions, ["Node 1", "Node 2"])
    ax.set_title("Constant offset estimates")
    ax.set_ylabel("Offset b")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # Use the same color range for both drift matrices.
    color_limit = max(
        float(np.max(np.abs(true_drift))),
        float(np.max(np.abs(default_result.drift))),
        1e-12,
    )

    matrices = [
        (axes[1, 0], true_drift, "Generating drift K"),
        (axes[1, 1], default_result.drift, "Estimated drift K: default fit"),
    ]

    for ax, matrix, title in matrices:
        heatmap = ax.imshow(
            matrix,
            cmap="coolwarm",
            vmin=-color_limit,
            vmax=color_limit,
        )

        ax.set_title(title)
        ax.set_xlabel("Source node")
        ax.set_ylabel("Receiving node")
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

    # Save a reproducible figure inside the project.
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "experiments" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "linear_estimation.png"
    fig.savefig(output_path, dpi=180)

    print(f"\nFigure saved to: {output_path}")
    print(
        "This experiment estimates strengths under a supplied mask; "
        "it does not discover unknown connections."
    )

    plt.show()


if __name__ == "__main__":
    main()