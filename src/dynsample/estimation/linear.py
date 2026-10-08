import numpy as np
from scipy.linalg import solve_triangular

from dynsample.core.linear_model import LinearSDE
from dynsample.core.trajectory import Trajectory
from dynsample.simulation.linear import linear_transition

# Score complete observations using the condition Gaussian likelihood

def linear_negative_log_likelihood(
        trajectory: Trajectory,
        model: LinearSDE,
) -> float:
    """Return the negative log-likelihood conditional on the first state."""

    if trajectory.n_steps < 2:
        raise ValueError("trajectory must contain at least two time steps")

    if trajectory.values.shape[1:] != (model.n_nodes, 1):
        raise ValueError(
            "trajectory must contain matching nodes and one feature per node"
        )

    with np.errstate(over = "ignore", invalid = "ignore"):
        intervals = np.diff(trajectory.times)

    if not np.all(np.isfinite(intervals)):
        raise ValueError("time intervals must be finite")

    if np.any(intervals <= 0):
        raise ValueError("time must be strictly increasing")

    values = trajectory.values[:, :, 0]
    normalizing_constant = model.n_nodes * np.log(2.0 * np.pi)
    total = 0.0

    for i, dt in enumerate(intervals, start = 1):
        transition, offset, covariance = linear_transition(
            model = model,
            dt = float(dt)
        )

        with np.errstate(over = "ignore", invalid = "ignore"):
            mean = transition @ values[i - 1] + offset
            residual = values[i] - mean

        if not np.all(np.isfinite(residual)):
            raise FloatingPointError(
                f"non-finite residual at transition {i}")

        try:
            lower = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"transition {i} requires a positive-definite covariance"
            ) from exc

        standardized_residual = solve_triangular(
            lower,
            residual,
            lower = True
        )

        with np.errstate(over="ignore", invalid="ignore"):
            log_determinant = 2.0 * np.sum(
                np.log(np.diag(lower))
            )

            quadratic = float(
                standardized_residual @ standardized_residual
            )

            contribution = 0.5 * (
                normalizing_constant
                + log_determinant
                + quadratic
            )

            total = total + float(contribution)

            if not np.isfinite(total):
                raise FloatingPointError(
                    f"non-finite likelihood at transition {i}"
            )

    return float(total)