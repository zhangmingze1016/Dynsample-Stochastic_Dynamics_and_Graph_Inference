"""Compare eight input combinations of the public OU fitting interface."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import dynsample as ds
from dynsample.simulation.ou import simulate_ou


# Simulate one trajectory and compare the supported fitting options.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save",
        type=Path,
        help="Optional output image path",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a plot window",
    )
    args = parser.parse_args()

    true_parameters = np.array([0.7, 10.0, 1.5])

    intervals = np.tile([0.05, 0.15], 500)
    times = np.concatenate(([0.0], np.cumsum(intervals)))

    trajectory = simulate_ou(
        initial_state=ds.State(
            time=0.0,
            values=np.array([[14.0]]),
        ),
        times=times,
        mean_reversion=true_parameters[0],
        long_run_mean=true_parameters[1],
        volatility=true_parameters[2],
        rng=np.random.default_rng(42),
    )

    initial = (0.5, 9.0, 1.2)
    bounds = (
        (0.01, 5.0),
        (-20.0, 20.0),
        (0.1, 5.0),
    )
    alpha_bounds = (0.1, 2.0)

    cases = [
        ("Default", {}),
        (
            "Initial",
            {"initial_parameters": initial},
        ),
        (
            "Bounds",
            {"parameter_bounds": bounds},
        ),
        (
            "Alpha bounds",
            {"alpha_bounds": alpha_bounds},
        ),
        (
            "Initial + bounds",
            {
                "initial_parameters": initial,
                "parameter_bounds": bounds,
            },
        ),
        (
            "Initial + alpha",
            {
                "initial_parameters": initial,
                "alpha_bounds": alpha_bounds,
            },
        ),
        (
            "Both bounds",
            {
                "parameter_bounds": bounds,
                "alpha_bounds": alpha_bounds,
            },
        ),
        (
            "All options",
            {
                "initial_parameters": initial,
                "parameter_bounds": bounds,
                "alpha_bounds": alpha_bounds,
            },
        ),
    ]

    results = []

    print("\nOU PUBLIC INTERFACE COMPARISON")
    print(
        f"True parameters: alpha={true_parameters[0]:.4f}, "
        f"mu={true_parameters[1]:.4f}, "
        f"sigma={true_parameters[2]:.4f}"
    )
    print(f"Observations: {times.size}")
    print(f"Duration: {times[-1] - times[0]:.4f}")

    print(
        f"\n{'Case':<22}"
        f"{'Method':<10}"
        f"{'Alpha':>10}"
        f"{'Mu':>10}"
        f"{'Sigma':>10}"
        f"{'NLL':>15}"
        f"{'Success':>10}"
    )

    for label, kwargs in cases:
        result = ds.fit_ou(
            trajectory,
            **kwargs,
        )
        results.append(result)

        print(
            f"{label:<22}"
            f"{result.method:<10}"
            f"{result.mean_reversion:>10.5f}"
            f"{result.long_run_mean:>10.5f}"
            f"{result.volatility:>10.5f}"
            f"{result.fun:>15.8f}"
            f"{str(bool(result.success)):>10}"
        )

    estimates = np.array([
        [
            result.mean_reversion,
            result.long_run_mean,
            result.volatility,
        ]
        for result in results
    ])

    scores = np.array([result.fun for result in results])
    reference_score = scores[0]
    score_differences = scores - reference_score

    print("\nNLL DIFFERENCE FROM DEFAULT FIT")

    for (label, _), difference in zip(cases, score_differences):
        print(f"{label:<22}{difference:>16.10f}")

    print("\nSTOPPING MESSAGES")

    for (label, _), result in zip(cases, results):
        print(f"{label}: {result.message}")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
        constrained_layout=True,
    )

    labels = [label for label, _ in cases]
    positions = np.arange(len(cases))
    parameter_names = ["Alpha", "Mu", "Sigma"]

    colors = [
        "tab:blue" if result.success else "tab:red"
        for result in results
    ]

    for column, ax in enumerate(axes.flat):
        if column < 3:
            ax.scatter(
                positions,
                estimates[:, column],
                c=colors,
                zorder=3,
            )
            ax.axhline(
                true_parameters[column],
                color="black",
                linestyle="--",
                label="Generating parameter",
            )
            ax.set_title(parameter_names[column])
            ax.set_ylabel("Estimated value")
            ax.legend()
        else:
            ax.scatter(
                positions,
                score_differences,
                c=colors,
                zorder=3,
            )
            ax.axhline(
                0.0,
                color="black",
                linestyle="--",
            )
            ax.set_title("NLL difference from default fit")
            ax.set_ylabel("NLL minus default NLL")
            ax.ticklabel_format(
                axis="y",
                style="sci",
                scilimits=(0, 0),
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(
            labels,
            rotation=35,
            ha="right",
        )
        ax.grid(alpha=0.25)

    fig.suptitle(
        "OU fitting with eight input combinations\n"
        "Blue: optimizer reports success; red: inspect stopping message"
    )

    if args.save is not None:
        args.save.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        fig.savefig(
            args.save,
            dpi=180,
        )
        print(f"\nSaved figure to: {args.save}")

    if not args.no_show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()