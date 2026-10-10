import numpy as np
from numpy.typing import NDArray

from dynsample.core.linear_model import LinearSDE
from dynsample.core.state import State
from dynsample.core.trajectory import Trajectory
from dynsample.simulation.linear import linear_transition

# Predict conditional means and marginal covariances from one exact state.
def predict_linear(
        initial_state: State,
        times: NDArray[np.float64],
        model: LinearSDE,
) -> tuple[Trajectory, NDArray[np.float64]]:
    """Return conditional means and process covariances at query times."""
    # Own the query-time array without changing the caller's input.
    times = np.array(
        times,
        dtype = np.float64,
        copy = True,
    )


    if times.ndim != 1:
        raise ValueError("times must be a one-dimensional array")

    if times.size == 0:
        raise ValueError("times must contain at least one time")

    if not np.all(np.isfinite(times)):
        raise ValueError("times must contain only finite values")

    if not np.isfinite(initial_state.time):
        raise ValueError("initial state time must be finite")

    if initial_state.values.shape != (model.n_nodes, 1):
        raise ValueError(
            "initial state must contain matching nodes "
            "and one feature per node"
        )

    if not np.all(np.isfinite(initial_state.values)):
        raise ValueError("initial state values must be finite")

    # Horizons are measured from the same initial time for every prediction.
    with np.errstate(over = "ignore", invalid = "ignore"):
        intervals =np.diff(times)
        horizons = times - initial_state.time
        if not np.all(np.isfinite(intervals)):
            raise ValueError("time intervals must be finite")

    if np.any(intervals <= 0.0):
        raise ValueError("times must be strictly increasing")

    if not np.all(np.isfinite(horizons)):
        raise ValueError("prediction horizons must be finite")

    if np.any(horizons < 0.0):
        raise ValueError(
            "prediction times must not precede the initial state"
        )

    n = model.n_nodes
    initial_values = initial_state.values[:, 0]

    means = np.empty(
        (times.size, n, 1),
        dtype = np.float64,
    )
    covariances = np.empty(
        (times.size, n, n),
        dtype = np.float64,
    )

    for i, horizon in enumerate(horizons):
        transition, offset, covariance = linear_transition(
            model = model,
            dt = float(horizon),
        )

        with np.errstate(over = "ignore", invalid = "ignore"):
            mean = transition @ initial_values + offset

            if not np.all(np.isfinite(mean)):
                raise FloatingPointError(
                    f"non-finite predicted mean at query index {i}"
                )

            means[i, :, 0] = mean
            covariances[i] = covariance

    mean_trajectory = Trajectory(
        times = times,
        values = means,
    )

    return mean_trajectory, covariances