import numpy as np
from numpy.typing import NDArray

from dynsample.core.trajectory import Trajectory

# Score aligned prediction means against complete observed trajectories.
def prediction_metrics(
        observed: Trajectory,
        predicted: Trajectory
) -> dict[str, float | NDArray[np.float64]]:
    """Return equally weighted overall and per-node RMSE and MAE."""

    # Both trajectories must describe the same time-node-feature grid.
    if observed.values.shape != predicted.values.shape:
        raise ValueError(
            "observed and predicted values must have matching shapes"
        )

    if not np.array_equal(observed.times, predicted.times):
        raise ValueError(
            "observed and predicted times must match exactly"
        )

    if observed.values.size == 0:
        raise ValueError(
            "trajectories must contain non-empty values"
        )

    if (
        not np.all(np.isfinite(observed.values))
        or not np.all(np.isfinite(predicted.values))
    ):
        raise ValueError(
            "observed and predicted values must be finite"
        )

    # Compute errors without silently accepting numerical overflow.
    with np.errstate(over = "ignore", invalid = "ignore"):
        errors = predicted.values - observed.values
        absolute_errors = np.abs(errors)
        squared_errors = errors ** 2

    if (
        not np.all(np.isfinite(absolute_errors))
        or not np.all(np.isfinite(squared_errors))
    ):
        raise FloatingPointError(
            "prediction errors or squared errors are non-finite"
        )

    # Overall scores average across times, nodes, and features.
    # Per-node scores average across times and features only.
    with np.errstate(over = "ignore", invalid = "ignore"):
        rmse = float(np.sqrt(np.mean(squared_errors)))
        mae = float(np.mean(absolute_errors))

        rmse_per_node = np.sqrt(
            np.mean(squared_errors, axis = (0, 2))
        )
        mae_per_node = np.mean(
            absolute_errors,
            axis = (0, 2),
        )

    if (
        not np.isfinite(rmse)
        or not np.isfinite(mae)
        or not np.all(np.isfinite(rmse_per_node))
        or not np.all(np.isfinite(mae_per_node))
    ):
        raise FloatingPointError(
            "prediction metrics are non-finite"
        )

    return {
        "rmse": rmse,
        "mae": mae,
        "rmse_per_node": rmse_per_node,
        "mae_per_node": mae_per_node,
    }