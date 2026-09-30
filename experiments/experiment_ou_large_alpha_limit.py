import matplotlib.pyplot as plt
import numpy as np

from dynsample.core.state import State
from dynsample.estimation.ou import (
    _profile_ou_mu,
    _profile_ou_sigma,
    _profile_ou_negative_log_likelihood,
)
from dynsample.simulation.ou import simulate_ou

def main() -> None:
    intervals = np.tile([0.05, 0.15], 500)
    times = np.concatenate(([0.0], np.cumsum(intervals)))

    trajectory = simulate_ou(
        initial_state = State(
            time = 0.0,
            values = np.array([[14.0]]),
        ),
        times = times,
        mean_reversion = 0.7,
        long_run_mean = 10.0,
        volatility = 1.5,
        rng = np.random.default_rng(42),
    )

    values = trajectory.values[:, 0, 0]
    targets = values[1:]
    n = targets.size

    limit_mu = float(np.mean(targets))
    limit_variance = float(
        np.mean((targets - limit_mu) ** 2)
    )

    if not np.isfinite(limit_variance) or limit_variance <= 0.0:
        raise ValueError(
            "the Gaussian boundary requires finite positive variance"
        )

    limit_score = float(
        0.5 * n * (
            np.log(2.0 * np.pi)
            + np.log(limit_variance)
            + 1.0
        )
    )

    alphas = np.array([1.0, 3.0, 10.0, 30.0, 100.0, 300.0])

    fitted_means = np.empty(alphas.size)
    variance_ratios = np.empty(alphas.size)
    scores = np.empty(alphas.size)

    print("\nINDEPENDENT GAUSSIAN BOUNDARY")
    print(f"mu:   {limit_mu:.8f}")
    print(f"tau2: {limit_variance:.8f}")
    print(f"NLL:  {limit_score:.8f}")

    print("\nOU PROFILE AS ALPHA INCREASES")
    print(
        f"{'alpha':>12}"
        f"{'mu':>14}"
        f"{'sigma':>14}"
        f"{'sigma2/(2alpha)':>18}"
        f"{'NLL difference':>18}"
    )

    for i, alpha in enumerate(alphas):
        mu = _profile_ou_mu(trajectory, alpha)
        sigma = _profile_ou_sigma(trajectory, alpha)
        score = _profile_ou_negative_log_likelihood(
            trajectory,
            alpha,
        )
        fitted_means[i] = mu
        variance_ratios[i] = sigma ** 2 / (2.0 * alpha)
        scores[i] = score

        print(
            f"{alpha:>12.2f}"
            f"{mu:>14.8f}"
            f"{sigma:>14.8f}"
            f"{variance_ratios[i]:>18.8f}"
            f"{score - limit_score:>18.8f}"
        )

    fig, axes = plt.subplots(
        3,
        1,
        figsize = (10, 9),
        sharex = True,
        constrained_layout = True,
    )


    axes[0].semilogx(
        alphas,
        fitted_means,
        marker="o",
        label="OU fitted mu",
    )
    axes[0].axhline(
        limit_mu,
        color="black",
        linestyle="--",
        label="Gaussian boundary mean",
    )
    axes[0].set_ylabel("Mean")
    axes[0].legend()

    axes[1].semilogx(
        alphas,
        variance_ratios,
        marker="o",
        label="OU fitted sigma squared / (2 alpha)",
    )
    axes[1].axhline(
        limit_variance,
        color="black",
        linestyle="--",
        label="Gaussian boundary variance",
    )
    axes[1].set_ylabel("Stationary variance")
    axes[1].legend()

    axes[2].semilogx(
        alphas,
        scores - limit_score,
        marker="o",
        label="OU profile NLL minus boundary NLL",
    )
    axes[2].axhline(
        0.0,
        color="black",
        linestyle="--",
    )
    axes[2].set_ylabel("NLL difference")
    axes[2].set_xlabel("Alpha (log scale)")
    axes[2].legend()

    for ax in axes:
        ax.grid(alpha=0.3)

    fig.suptitle("OU profile likelihood approaching the large-alpha boundary")
    plt.show()


if __name__ == "__main__":
    main()