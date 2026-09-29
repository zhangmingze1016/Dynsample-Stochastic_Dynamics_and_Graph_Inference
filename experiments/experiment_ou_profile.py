"""Compare profile fitting with joint fitting and inspect the alpha score curve."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dynsample.core.state import State
from dynsample.estimation.ou import (
    _profile_ou_negative_log_likelihood,
    fit_ou,
    fit_ou_profile,
)
from dynsample.simulation.ou import simulate_ou


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=Path, help="Optional output image path")
    parser.add_argument("--no-show", action="store_true", help="Do not open a plot window")
    args = parser.parse_args()

    # Alternate short and long intervals; keep the original timestamps.
    true_parameters = np.array([0.7, 10.0, 1.5])
    times = np.concatenate(([0.0], np.cumsum(np.tile([0.05, 0.15], 500))))
    trajectory = simulate_ou(
        initial_state=State(time=0.0, values=np.array([[14.0]])),
        times=times,
        mean_reversion=true_parameters[0],
        long_run_mean=true_parameters[1],
        volatility=true_parameters[2],
        rng=np.random.default_rng(42),
    )

    alpha_bounds = (0.001, 5.0)
    profile = fit_ou_profile(trajectory, alpha_bounds=alpha_bounds)
    joint = fit_ou(
        trajectory,
        initial_parameters=None,
        parameter_bounds=(alpha_bounds, (-20.0, 20.0), (0.01, 10.0)),
    )
    if not joint.success:
        raise RuntimeError(f"Joint fit failed: {joint.message}")

    fitted = np.array([
        profile.mean_reversion, profile.long_run_mean, profile.volatility
    ])
    joint_parameters = np.array([np.exp(joint.x[0]), joint.x[1], np.exp(joint.x[2])])
    print("OU PROFILE FIT: irregular observations")
    print(f"{'Parameter':<12}{'True':>12}{'Profile':>12}{'Joint':>12}")
    for name, truth, estimate, reference in zip(
        ("alpha", "mu", "sigma"), true_parameters, fitted, joint_parameters
    ):
        print(f"{name:<12}{truth:>12.4f}{estimate:>12.4f}{reference:>12.4f}")
    print(f"Profile NLL: {profile.fun:.8f}")
    print(f"Joint NLL:   {joint.fun:.8f}")
    print(f"Absolute NLL difference: {abs(profile.fun - joint.fun):.3g}")
    print(f"Alpha search bounds: {alpha_bounds}")
    print(f"Profile optimizer status: {profile.message}")
    print("Joint fitting also restricts mu and sigma; profile fitting does not.")

    # Scan for visualization only; this is not proof of global optimality.
    alpha_grid = np.geomspace(*alpha_bounds, 180)
    scores = np.array([
        _profile_ou_negative_log_likelihood(trajectory, alpha)
        for alpha in alpha_grid
    ])
    print(f"Best displayed grid NLL: {scores.min():.8f}")
    if scores.min() < profile.fun - 1e-6:
        print("WARNING: the displayed grid contains a better score than the optimizer.")

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    axes[0].plot(times, trajectory.values[:, 0, 0], linewidth=0.8, label="Observed trajectory")
    axes[0].axhline(true_parameters[1], color="black", linestyle=":", label="True long-run mean")
    axes[0].axhline(fitted[1], color="tab:orange", linestyle="--", label="Estimated long-run mean")
    axes[0].set(title="OU observations at irregular times", xlabel="Time", ylabel="State")
    axes[0].legend()

    axes[1].semilogx(alpha_grid, scores - profile.fun, label="Profile NLL minus fitted NLL")
    axes[1].axvline(true_parameters[0], color="black", linestyle=":", label="True alpha")
    axes[1].axvline(fitted[0], color="tab:orange", linestyle="--", label="Estimated alpha")
    axes[1].scatter([fitted[0]], [0.0], color="tab:orange", zorder=3)
    axes[1].set(xlabel="Alpha (log scale)", ylabel="NLL difference", title="Profile likelihood within supplied bounds")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.25)

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=160)
        print(f"Saved figure: {args.save.resolve()}")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
