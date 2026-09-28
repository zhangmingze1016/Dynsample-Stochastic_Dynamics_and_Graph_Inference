import numpy as np
import matplotlib.pyplot as plt

from dynsample.core.state import State
from dynsample.simulation.ou import simulate_ou, ou_transition
from dynsample.estimation.ou import (
    _initial_ou_parameters,
    ou_negative_log_likelihood,
    fit_ou,
)

def main() -> None:
    true_alpha = 0.7
    true_mu = 10.0
    true_sigma = 1.5

    intervals = np.tile([0.05, 0.15], 500)
    times = np.concatenate(([0.0], np.cumsum(intervals)))

    initial_state = State(
        time = 0.0,
        values = np.array([[14.0]]),
    )

    trajectory = simulate_ou(
        initial_state = initial_state,
        times = times,
        mean_reversion = true_alpha,
        long_run_mean = true_mu,
        volatility = true_sigma,
        rng = np.random.default_rng(42),
    )

    values = trajectory.values[:, 0, 0]

    initial_parameters = _initial_ou_parameters(trajectory)

    parameter_bounds = np.array([
        [0.001, 5.0],
        [-20.0, 20.0],
        [0.01, 10.0],
    ])

    result = fit_ou(
        trajectory = trajectory,
        initial_parameters = None,
        parameter_bounds = parameter_bounds,
    )

    estimated_alpha = float(np.exp(result.x[0]))
    estimated_mu = float(result.x[1])
    estimated_sigma = float(np.exp(result.x[2]))

    initial_score = ou_negative_log_likelihood(
        trajectory=trajectory,
        mean_reversion=initial_parameters[0],
        long_run_mean=initial_parameters[1],
        volatility=initial_parameters[2],
    )

    print("\nOU PARAMETER ESTIMATION")
    print("-" * 56)
    print(
        f"{'Parameter':<12}"
        f"{'True':>12}"
        f"{'Initial':>16}"
        f"{'Estimated':>16}"
    )

    names = ["alpha", "mu", "sigma"]
    true_values = [true_alpha, true_mu, true_sigma]
    estimated_values = [
        estimated_alpha,
        estimated_mu,
        estimated_sigma,
    ]

    for i in range(3):
        print(
            f"{names[i]:<12}"
            f"{true_values[i]:>12.4f}"
            f"{initial_parameters[i]:>16.4f}"
            f"{estimated_values[i]:>16.4f}"
    )

    print()
    print(f"Number of observations: {times.size}")
    print(f"Time span: {times[-1] - times[0]:.4f}")
    print(f"Initial negative log-likelihood: {initial_score:.6f}")
    print(f"Final negative log-likelihood: {result.fun:.6f}")
    print(f"Optimization success: {result.success}")
    print(f"Stopping reason: {result.message}")
    print("Parameter bounds:")
    print(parameter_bounds)


    search_bounds = parameter_bounds.copy()
    search_bounds[[0, 2], :] = np.log(
        parameter_bounds[[0, 2], :]
    )

    near_lower = np.isclose(
        result.x,
        search_bounds[:, 0],
        rtol=0.0,
        atol=1e-5,
    )
    near_upper = np.isclose(
        result.x,
        search_bounds[:, 1],
        rtol=0.0,
        atol=1e-5,
    )
    near_boundary = near_lower | near_upper

    print(
        "Near a search boundary (alpha, mu, sigma):",
        near_boundary,
    )

    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(
            "The fit did not finish successfully. "
            "Inspect the printed result before plotting."
        )

    if np.any(near_boundary):
        print(
            "A fitted parameter is close to a search boundary. "
            "Check sensitivity to the chosen bounds."
        )

    # 5. Calculate fitted one-step means and standardized residuals.
    predicted = np.empty(times.size - 1)
    standardized_residuals = np.empty(times.size - 1)

    for i in range(1, times.size):
        transition, offset, variance = ou_transition(
            dt=times[i] - times[i - 1],
            mean_reversion=estimated_alpha,
            long_run_mean=estimated_mu,
            volatility=estimated_sigma,
        )

        predicted[i - 1] = (
            transition * values[i - 1] + offset
        )

        standardized_residuals[i - 1] = (
            values[i] - predicted[i - 1]
        ) / np.sqrt(variance)

    print(
        f"Standardized residual mean: "
        f"{np.mean(standardized_residuals):.4f}"
    )
    print(
        f"Standardized residual variance: "
        f"{np.var(standardized_residuals):.4f}"
    )

    # 6. Generate new paths under the fitted model.
    n_paths = 5
    simulated_paths = np.empty((n_paths, times.size))
    simulation_rng = np.random.default_rng(2026)

    for path_index in range(n_paths):
        new_trajectory = simulate_ou(
            initial_state=initial_state,
            times=times,
            mean_reversion=estimated_alpha,
            long_run_mean=estimated_mu,
            volatility=estimated_sigma,
            rng=simulation_rng,
        )

        simulated_paths[path_index] = (
            new_trajectory.values[:, 0, 0]
        )

    # Conditional mean given only the initial state.
    elapsed = times - initial_state.time
    initial_value = initial_state.values[0, 0]

    fitted_mean_from_start = (
        estimated_mu
        + np.exp(-estimated_alpha * elapsed)
        * (initial_value - estimated_mu)
    )

    # 7. Plot the experiment.
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 11),
        sharex=True,
    )

    axes[0].plot(
        times,
        values,
        color="tab:blue",
        linewidth=1.0,
        label="Observed trajectory",
    )
    axes[0].plot(
        times[1:],
        predicted,
        color="tab:orange",
        linewidth=1.0,
        alpha=0.8,
        label="Fitted one-step conditional mean",
    )
    axes[0].axhline(
        true_mu,
        color="black",
        linestyle=":",
        label="True long-run mean",
    )
    axes[0].set_title("Observed data and fitted one-step means")
    axes[0].set_ylabel("State")
    axes[0].legend()

    axes[1].plot(
        times[1:],
        standardized_residuals,
        color="tab:purple",
        linewidth=0.8,
    )
    axes[1].axhline(
        0.0,
        color="black",
        linestyle="--",
    )
    axes[1].set_title("In-sample standardized residuals")
    axes[1].set_ylabel("Residual")

    for path_index in range(n_paths):
        label = None
        if path_index == 0:
            label = "New paths under fitted parameters"

        axes[2].plot(
            times,
            simulated_paths[path_index],
            color="tab:blue",
            linewidth=0.8,
            alpha=0.35,
            label=label,
        )

    axes[2].plot(
        times,
        fitted_mean_from_start,
        color="tab:orange",
        linewidth=2.0,
        label="Fitted conditional mean given initial state",
    )
    axes[2].axhline(
        estimated_mu,
        color="black",
        linestyle="--",
        label="Estimated long-run mean",
    )
    axes[2].set_title(
        "New simulations with fixed fitted parameters"
    )
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("State")
    axes[2].legend()

    for ax in axes:
        ax.grid(alpha=0.25)

    fig.suptitle("OU simulation and parameter estimation")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()





