import numpy as np
from scipy.optimize import OptimizeResult
from dynsample.core.trajectory import Trajectory

def fit_brownian_drift(
        trajectory: Trajectory,
) -> OptimizeResult:

    if trajectory.values.shape[1:] != (1, 1):
        raise ValueError(
            "trajectory must contain one node and one feature"
        )

    if trajectory.times.size < 3:
        raise ValueError(
            "trajectory must contain at least three time points"
        )

    times = trajectory.times
    values = trajectory.values[:, 0, 0]

    dt = np.diff(times)
    increments = np.diff(values)


    if not np.all(np.isfinite(dt)) or np.any(dt <= 0.0):
        raise ValueError(
            "time intervals must be finite and positive"
        )

    duration = float(times[-1] - times[0])

    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError(
            "time span must be finite and positive"
        )

    drift = float(values[-1] - values[0]) / duration

    if not np.isfinite(drift):
        raise ValueError(
            "drift calculation produced a non-finite value"
        )

    residuals = increments - drift * dt
    scaled_resiuals = residuals / np.sqrt(dt)

    sigma_squared = float(np.mean(scaled_resiuals ** 2))

    
    if not np.isfinite(sigma_squared):
        raise ValueError(
            "variance calculation produced a non-finite value"
        )

    if sigma_squared <= 0.0:
        raise ValueError(
            "cannot estimate positive volatility from zero residual variance"
        )

    volatility = float(np.sqrt(sigma_squared))
    standardized_residuals = scaled_resiuals / volatility

    score = float(
        0.5 * np.sum(
            np.log(2.0 * np.pi)
            + np.log(sigma_squared)
            + np.log(dt)
            + standardized_residuals ** 2
        )
    )

    if not np.isfinite(score):
        raise ValueError(
            "negative log-likelihood must be finite"
        )

    return OptimizeResult(
        x = np.array([drift, volatility]),
        drift = drift,
        volatility = volatility,
        fun = score,
        success = True,
        message="Closed-form conditional maximum likelihood estimate.",
    )