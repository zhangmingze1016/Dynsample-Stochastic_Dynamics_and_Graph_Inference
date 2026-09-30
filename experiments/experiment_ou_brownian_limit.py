import matplotlib.pyplot as plt
import numpy as np
from dynsample.core.state import State
from dynsample.estimation.brownian import fit_brownian_drift
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
        long_run_mean = 10,
        volatility = 1.5,
        rng = np.random.default_rng(42,)
    )

    brownian = fit_brownian_drift(trajectory)

    alphas = np.logspace(-1, -6, 6)

    effective_drifts = np.empty(alphas.size)
    volatilities = np. empty(alphas.size)
    scores = np.empty(alphas.size)

    print("\nBROWNIAN BOUNDARY FIT")
    print(f"drift: {brownian.drift:.8f}")
    print(f"sigma: {brownian.volatility:.8f}")
    print(f"NLL:   {brownian.fun:.8f}")

    print("\nOU PROFILE AS ALPHA APPROACHES ZERO")
    print(
        f"{'alpha':>12}"
        f"{'mu':>16}"
        f"{'alpha * mu':>16}"
        f"{'sigma':>14}"
        f"{'NLL difference':>18}"
    )

    for i, alpha in enumerate(alphas):
        mu = _profile_ou_mu(trajectory, alpha)
        sigma = _profile_ou_sigma(trajectory, alpha)
        score = _profile_ou_negative_log_likelihood(
            trajectory,
            alpha,
        )

        effective_drifts[i] = alpha * mu
        volatilities[i] = sigma
        scores[i] = score

        print(
            f"{alpha:>12.1e}"
            f"{mu:>16.6f}"
            f"{effective_drifts[i]:>16.8f}"
            f"{sigma:>14.8f}"
            f"{score - brownian.fun:>18.8f}"
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
        effective_drifts,
        marker = "o",
        label = "OU alpha * fitted mu",
    )
    axes[0].axhline(
        brownian.drift,
        color = "black",
        linestyle = "--",
        label = "Brownian fitted drift",
    )
    axes[0].set_ylabel("Drift coefficient")
    axes[0].legend()

    axes[1].semilogx(
        alphas,
        volatilities,
        marker = "o",
        label = "OU fitted Sigma",
    )
    axes[1].axhline(
        brownian.volatility,
        color="black",
        linestyle="--",
        label="Brownian fitted sigma",
    )
    axes[1].set_ylabel("Volatility")
    axes[1].legend()

    axes[2].semilogx(
        alphas,
        scores - brownian.fun,
        marker="o",
        label="OU profile NLL minus Brownian NLL",
    )
    axes[2].axhline(
        0.0,
        color="black",
        linestyle="--",
    )
    axes[2].set_ylabel("NLL difference")
    axes[2].set_xlabel("Alpha — smaller values toward the right")
    axes[2].legend()

    axes[2].set_xlim(alphas[0], alphas[-1])

    for ax in axes:
        ax.grid(alpha=0.3)

    fig.suptitle("OU profile likelihood approaching the Brownian boundary")
    plt.show()


if __name__ == "__main__":
    main()