"""Check drifted Brownian MLE against hand calculations and direct densities."""

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.stats import norm

from dynsample.core.trajectory import Trajectory
from dynsample.estimation.brownian import fit_brownian_drift


@pytest.mark.parametrize(
    "times, values, drift, variance, transition_variances",
    [
        ([0.0, 1.0, 2.0], [1.0, 4.0, 3.0], 1.0, 4.0, [4.0, 4.0]),
        ([2.0, 3.0, 5.0], [1.0, 4.0, 4.0], 1.0, 3.0, [3.0, 6.0]),
        ([2.0, 3.0, 5.0], [-1.0, -4.0, -4.0], -1.0, 3.0, [3.0, 6.0]),
    ],
)
def test_brownian_fit_matches_hand_calculation(
    times, values, drift, variance, transition_variances
) -> None:
    trajectory = Trajectory(
        times=np.array(times),
        values=np.array(values)[:, None, None],
    )
    original_times = trajectory.times.copy()
    original_values = trajectory.values.copy()

    result = fit_brownian_drift(trajectory)

    # In these examples the two residuals have magnitude 2.
    q = np.array(transition_variances)
    expected_score = 0.5 * np.sum(np.log(2.0 * np.pi * q) + 4.0 / q)
    assert result.success
    np.testing.assert_allclose(result.drift, drift, rtol=1e-12)
    np.testing.assert_allclose(result.volatility, np.sqrt(variance), rtol=1e-12)
    np.testing.assert_allclose(result.x, [drift, np.sqrt(variance)], rtol=1e-12)
    np.testing.assert_allclose(result.fun, expected_score, rtol=1e-12)
    np.testing.assert_array_equal(trajectory.times, original_times)
    np.testing.assert_array_equal(trajectory.values, original_values)


def test_brownian_fit_matches_independent_density_optimization() -> None:
    times = np.array([0.0, 0.2, 0.9, 2.4, 3.0, 5.0])
    values = np.array([2.0, 2.4, 1.1, 3.2, 2.8, 4.1])
    trajectory = Trajectory(times=times, values=values[:, None, None])
    dt = np.diff(times)

    # Evaluate Gaussian densities independently of the implementation's NLL.
    def objective(parameters):
        drift, log_sigma = parameters
        return -np.sum(norm.logpdf(
            np.diff(values), loc=drift * dt,
            scale=np.exp(log_sigma) * np.sqrt(dt),
        ))

    reference = minimize(
        objective, x0=np.array([0.0, 0.0]), method="Nelder-Mead",
        options={"xatol": 1e-9, "fatol": 1e-10, "maxiter": 2000},
    )
    actual = fit_brownian_drift(trajectory)

    assert reference.success, reference.message
    np.testing.assert_allclose(
        actual.x, [reference.x[0], np.exp(reference.x[1])],
        rtol=1e-6, atol=1e-7,
    )
    np.testing.assert_allclose(actual.fun, reference.fun, rtol=0.0, atol=1e-8)


@pytest.mark.parametrize("values", [[0.0, 0.0, 0.0], [1.0, 3.0, 7.0]])
def test_brownian_fit_rejects_zero_residual_variance(values) -> None:
    trajectory = Trajectory(
        times=np.array([0.0, 1.0, 3.0]),
        values=np.array(values)[:, None, None],
    )
    with pytest.raises(ValueError, match="zero residual variance"):
        fit_brownian_drift(trajectory)


@pytest.mark.parametrize("steps", [1, 2])
def test_brownian_fit_rejects_insufficient_observations(steps) -> None:
    trajectory = Trajectory(
        times=np.arange(steps, dtype=float),
        values=np.zeros((steps, 1, 1)),
    )
    with pytest.raises(ValueError, match="at least three"):
        fit_brownian_drift(trajectory)


@pytest.mark.parametrize("shape", [(3, 2, 1), (3, 1, 2)])
def test_brownian_fit_rejects_multiple_nodes_or_features(shape) -> None:
    trajectory = Trajectory(
        times=np.arange(3, dtype=float), values=np.zeros(shape),
    )
    with pytest.raises(ValueError, match="one node and one feature"):
        fit_brownian_drift(trajectory)
