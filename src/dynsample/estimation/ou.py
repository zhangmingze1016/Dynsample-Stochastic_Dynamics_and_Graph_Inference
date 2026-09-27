import numpy as np

from dynsample.core.trajectory import Trajectory
from dynsample.simulation.ou import ou_transition

def ou_negative_log_likelihood(
        trajectory: Trajectory,
        mean_reversion: float,
        long_run_mean: float,
        volatility: float,
) -> float:
    if trajectory.values.shape[1:] != (1, 1):
        raise ValueError(
            "trajectory must contain one node and one feature"
        )

    if trajectory.times.size < 2:
        raise ValueError(
            "trajectory must contain at least two time points"
        )

    parameters = (mean_reversion, long_run_mean, volatility)
    if not np.all(np.isfinite(parameters)):
        raise ValueError("all parameters must be finite")

    if mean_reversion <= 0.0:
        raise ValueError("mean_reversion must be positive")

    if volatility <= 0.0:
        raise ValueError("volatility must be positive")

    times = trajectory.times
    values = trajectory.values[:, 0, 0]
    total = 0.0

    for i in range(1, times.size):
        dt = times[i] - times[i - 1]

        transition, offset, variance = ou_transition(
            dt = dt,
            mean_reversion = mean_reversion,
            long_run_mean = long_run_mean,
            volatility = volatility,
        )

        if not np.isfinite(variance) or variance <= 0.0:
            raise ValueError("transition variance must be finite and positive")

        mean = transition * values[i - 1] + offset
        residual = values[i] - mean

        total += 0.5 * (
            np.log(2.0 * np.pi)
            + np.log(variance)
            + residual**2 / variance
        )

    return float(total)