"""Visualize a three-node linear SDE and check its transition moments."""

import matplotlib.pyplot as plt
import numpy as np
from dynsample.core.linear_model import LinearSDE
from dynsample.core.state import State
from dynsample.simulation.linear import (
    linear_step,
    linear_transition,
    simulate_linear,
)

def main() -> None:
    model = LinearSDE(
        drift = np.array([
            [-1.0, 0.0, 0.0],
            [0.8, -1.0, 0.0],
            [0.0, 0.8, -1.0],
        ]),
        offset = np.zeros(3),
        diffusion = 0.15 * np.eye(3),
    )

    initial_state = State(
        time = 0.0,
        values = np.array([
            [3.0],
            [0.0],
            [0.0],
        ]),
    )

    intervals = np.tile([0.03, 0.07], 80)
    times = np.concatenate(([0.0], np.cumsum(intervals)))

    trajectory = simulate_linear(
        initial_state = initial_state,
        times = times,
        model = model,
        rng = np.random.default_rng(42),
    )

    elapsed = times - initial_state.time
    decay = np.exp(-elapsed)

    theoretical_means = np.column_stack(
        (
            3.0 * decay,
            2.4 * elapsed * decay,
            0.96 * elapsed ** 2 * decay,
        )
    )

    standard_deviations = np.empty((times.size, model.n_nodes))

    for i, dt in enumerate(elapsed):
        _, _, covariance = linear_transition(
            model = model,
            dt = float(dt)
        )
        standard_deviations[i] = np.sqrt(
            np.maximum(np.diag(covariance), 0.0)
        )

    horizon = 2.0
    n_samples = 5000
    samples = np.empty((n_samples, model.n_nodes))
    rng = np.random.default_rng(42)

    for i in range(n_samples):
        result = linear_step(
            state = initial_state,
            next_time = initial_state.time + horizon,
            model = model,
            rng = rng,
        )
        samples[i] = result.values[:, 0]

    transition, offset, expected_covariance = linear_transition(
        model = model,
        dt = horizon,
    )

    expected_mean = transition @ initial_state.values[:, 0] + offset

    sample_mean =np.mean(samples, axis = 0)
    sample_covariance = np.cov(samples, rowvar = False, ddof =1)

    print("LINEAR SDE: DIRECTED CHAIN")
    print("Direct connections: node 1 -> node 2 -> node 3")
    print(f"Observation times: {times.size}")
    print(f"Duration: {elapsed[-1]:.4f}")
    print(f"\nMoment comparison at elapsed time {horizon:.2f}")
    print(f"Independent samples: {n_samples}")
    print(f"{'Node':<10}{'Theory mean':>16}{'Sample mean':>16}")


    for node in range(model.n_nodes):
        print(
            f"{node + 1:<10}"
            f"{expected_mean[node]:>16.6f}"
            f"{sample_mean[node]:>16.6f}"
        )

    print("\nTheoretical covariance:")
    print(expected_covariance)
    print("\nSample covariance:")
    print(sample_covariance)
    print(
        "\nMaximum absolute covariance difference:",
        float(np.max(np.abs(sample_covariance - expected_covariance))),
    )



    fig, axes = plt.subplots(
        4,
        1,
        figsize=(11, 11),
        sharex=True,
        constrained_layout=True,
    )

    colors = ["tab:blue", "tab:orange", "tab:green"]

    for node in range(model.n_nodes):
        ax = axes[node]
        mean = theoretical_means[:, node]
        std = standard_deviations[:, node]

        ax.fill_between(
            times,
            mean - 1.96 * std,
            mean + 1.96 * std,
            color=colors[node],
            alpha=0.15,
            label="Pointwise 95% conditional interval",
        )
        ax.plot(
            times,
            trajectory.values[:, node, 0],
            color=colors[node],
            linewidth=1.2,
            label="Sampled path",
        )
        ax.plot(
            times,
            mean,
            color="black",
            linestyle="--",
            label="Analytical conditional mean",
        )

        ax.set_title(f"Node {node + 1}")
        ax.set_ylabel("State")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    for node in range(model.n_nodes):
        axes[3].plot(
            times,
            theoretical_means[:, node],
            color=colors[node],
            label=f"Node {node + 1}",
        )

    axes[3].set_title(
        "Propagation of an initial perturbation under a fixed graph"
    )
    axes[3].set_xlabel("Time")
    axes[3].set_ylabel("Conditional mean")
    axes[3].grid(alpha=0.25)
    axes[3].legend()

    fig.suptitle("Linear SDE simulation: node 1 → node 2 → node 3")
    plt.show()


if __name__ == "__main__":
    main()
    