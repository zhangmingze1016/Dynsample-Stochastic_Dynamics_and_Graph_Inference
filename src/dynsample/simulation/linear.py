import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm
from dynsample.core.linear_model import LinearSDE
from dynsample.core.state import State
from dynsample.core.trajectory import Trajectory

def linear_transition(
        model: LinearSDE,
        dt:float
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],

]:
    """Return (transition, offset, covariance) for a constant linear SDE."""
    if not np.isfinite(dt):
        raise ValueError("dt must be finite")

    if dt < 0:
        raise ValueError("dt must be non-negative")

    n = model.n_nodes

    if dt == 0:
        return(
            np.eye(n, dtype = np.float64),
            np.zeros(n, dtype = np.float64),
            np.zeros((n, n), dtype = np.float64),
        )

    drift = model.drift
    diffusion = model.diffusion

    # Augment the drift to integrate the constant offset without inverting it.

    mean_block = np.zeros((n + 1, n + 1), dtype = np.float64)
    mean_block[:n, :n] = drift
    mean_block[:n, n] = model.offset


    # Construct the Van Loan block for the integrated noise covariance.
    with np.errstate(over="ignore", invalid="ignore"):
        noise_rate = diffusion @ diffusion.T

        covariance_block = np.zeros((2 * n, 2 * n), dtype = np.float64)
        covariance_block[:n, :n] = drift
        covariance_block[:n, n:] = noise_rate
        covariance_block[n:, n:] = -drift.T

        scaled_mean_block = mean_block * dt
        scaled_covariance_block = covariance_block * dt


    if (
        not np.all(np.isfinite(scaled_mean_block))
        or not np.all(np.isfinite(scaled_covariance_block))
    ):
        raise FloatingPointError("transition blocks contain non-finite values")

    with np.errstate(over = "ignore", invalid = "ignore"):
        mean_exponential = expm(scaled_mean_block)
        covariance_exponential = expm(scaled_covariance_block)

    if (
        not np.all(np.isfinite(mean_exponential))
        or not np.all(np.isfinite(covariance_exponential))
    ):
        raise FloatingPointError("matrix exponential contains non-finite values")

    transition = mean_exponential[:n, :n].copy()
    offset = mean_exponential[:n, n].copy()

    with np.errstate(over = "ignore", invalid = "ignore"):
        covariance = covariance_exponential[:n, n:] @ transition.T
        covariance = 0.5 * covariance + 0.5 * covariance.T

    if not np.all(np.isfinite(covariance)):
        raise FloatingPointError("transition covariance contains non-finite values")

    return transition, offset, covariance

def linear_step(
        state: State,
        next_time: float,
        model: LinearSDE,
        rng: np.random.Generator,    
)-> State:
    """Sample the next state under a constant linear SDE."""

    if not np.isfinite(next_time):
        raise ValueError("next_time must be finite")

    if next_time <= state.time:
        raise ValueError("next_time must be greater than the current state time")

    if state.values.shape != (model.n_nodes, 1):
        raise ValueError(
            "state values must have shape (model.n_nodes, 1)"
        )

    with np.errstate(over = "ignore", invalid = "ignore"):
        dt = next_time - state.time

    if not np.isfinite(dt):
        raise ValueError("time interval must be finite")

    transition, offset, covariance = linear_transition(model, dt)

    current_values = state.values[:, 0]

    with np.errstate(over = "ignore", invalid = "ignore"):
        mean = transition @ current_values + offset

    if not np.all(np.isfinite(mean)):
        raise FloatingPointError("conditional mean contains non-finite values")

    # Factor a positive-semidefinite covariance, including singular cases.

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    if (
        not np.all(np.isfinite(eigenvalues))
        or not np.all(np.isfinite(eigenvectors))
    ):
        raise FloatingPointError(
            "covariance decomposition contains non-finite values"
        )

    scale = float(np.max(np.abs(eigenvalues)))
    tolerance = (
        100
        * np.finfo(np.float64).eps
        * model.n_nodes
        * scale
    )

    if np.min(eigenvalues) < -tolerance:
        raise FloatingPointError(
            "transition covariance is not positive semidefinite"
        )

    eigenvalues = np.maximum(eigenvalues, 0.0)
    standard_noise = rng.standard_normal(size = model.n_nodes)

    with np.errstate(over = "ignore", invalid = "ignore"):
        noise = eigenvectors @ (
            np.sqrt(eigenvalues) * standard_noise
        )
        next_values = mean + noise

    if not np.all(np.isfinite(next_values)):
        raise FloatingPointError("next state contains non-finite values")

    eigenvalues = np.max(eigenvalues) * standard_noise

    return State(
        time = next_time,
        values = next_values[:, None],
    )

# Simulate a coupled linear SDE at supplied observation times.
def simulate_linear(
        initial_state: State,
        times: NDArray[np.float64],
        model: LinearSDE,
        rng: np.random.Generator
) -> Trajectory:
    """Return a trajectory with one feature per node."""

    times = np.asarray(times, dtype = np.float64)

    if times.ndim != 1:
        raise ValueError("times must be one-dimensional")

    if times.size == 0:
        raise ValueError("times must contain at least one time")

    if not np.all(np.isfinite(times)):
        raise ValueError("times must contain only finite values")

    if times[0] != initial_state.time:
        raise ValueError(
            "the first time must equal initial_state.time"
        )

    with np.errstate(over = "ignore", invalid = "ignore"):
        intervals = np.diff(times)

    if not np.all(np.isfinite(intervals)):
        raise ValueError("time intervals must be finite")

    if np.any(intervals <= 0):
        raise ValueError("times must be strictly increasing")

    if initial_state.values.shape != (model.n_nodes, 1):
        raise ValueError(
            "initial state values must have shape (model.n_nodes, 1)"
        )

    values = np.empty(
        (times.size, model.n_nodes, 1),
        dtype = np.float64,
    )

    values[0] = initial_state.values
    current_state = initial_state

    for i in range(1, times.size):
        current_state = linear_step(
            state = current_state,
            next_time = float(times[i]),
            model = model,
            rng = rng
        )
        values[i] = current_state.values

    return Trajectory(
        times = times,
        values = values,
    )