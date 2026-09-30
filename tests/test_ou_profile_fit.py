"""End-to-end checks against an independently derived regular-grid MLE."""

import numpy as np
import pytest

from dynsample.core.trajectory import Trajectory
from dynsample.estimation import ou


def test_profile_fit_matches_regular_grid_mle() -> None:
    # Generate AR(1) observations without using the OU simulator.
    rng = np.random.default_rng(42)
    values = np.empty(401)
    values[0] = -3.0
    for i in range(1, values.size):
        values[i] = 0.8 * values[i - 1] - 0.6 + 0.5 * rng.standard_normal()
    dt = 0.25
    trajectory = Trajectory(
        times=np.arange(values.size) * dt,
        values=values[:, None, None],
    )

    # Regular-grid Gaussian conditional MLE is an AR(1) least-squares fit.
    design = np.column_stack((values[:-1], np.ones(values.size - 1)))
    phi, intercept = np.linalg.lstsq(design, values[1:], rcond=None)[0]
    residual = values[1:] - design @ np.array([phi, intercept])
    q = np.mean(residual**2)
    assert 0.0 < phi < 1.0
    alpha = -np.log(phi) / dt
    expected = [alpha, intercept / (1.0 - phi), np.sqrt(2.0 * alpha * q / (1.0 - phi**2))]
    expected_score = 0.5 * residual.size * (np.log(2.0 * np.pi * q) + 1.0)
    original_values = trajectory.values.copy()
    original_times = trajectory.times.copy()

    result = ou.fit_ou_profile(trajectory, alpha_bounds=(0.01, 5.0))

    assert result.success
    np.testing.assert_allclose(
        [result.mean_reversion, result.long_run_mean, result.volatility],
        expected, rtol=1e-5, atol=1e-6,
    )
    np.testing.assert_allclose(result.fun, expected_score, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(result.x, result.mean_reversion)
    np.testing.assert_allclose(np.exp(result.log_alpha), result.mean_reversion)
    np.testing.assert_allclose(
        result.fun,
        ou.ou_negative_log_likelihood(
            trajectory, result.mean_reversion, result.long_run_mean, result.volatility
        ),
    )
    assert result.alpha_bounds == (0.01, 5.0)
    np.testing.assert_array_equal(trajectory.values, original_values)
    np.testing.assert_array_equal(trajectory.times, original_times)


@pytest.mark.parametrize(
    "bounds",
    [(0.0, 1.0), (-1.0, 1.0), (2.0, 1.0), (1.0, 1.0),
     (np.nan, 1.0), (0.1, np.inf), (0.1,), (0.1, 1.0, 2.0)],
)
def test_profile_fit_rejects_invalid_bounds(bounds) -> None:
    trajectory = Trajectory(
        times=np.array([0.0, 1.0, 2.0]),
        values=np.array([14.0, 13.0, 10.0])[:, None, None],
    )
    with pytest.raises(ValueError):
        ou.fit_ou_profile(trajectory, alpha_bounds=bounds)


def test_profile_fit_rejects_degenerate_data() -> None:
    trajectory = Trajectory(
        times=np.array([0.0, 0.3, 2.0]),
        values=np.zeros((3, 1, 1)),
    )
    with pytest.raises(ValueError, match="zero residual variance"):
        ou.fit_ou_profile(trajectory, alpha_bounds=(0.01, 5.0))

@pytest.mark.parametrize(
    "bounds, expected_lower, expected_upper",
    [
        ((0.01, 5.0), False, False),
        ((0.01, 0.3), False, True),
        ((2.0, 5.0), True, False),
    ],
)
def test_profile_fit_boundary_flags(
    bounds,
    expected_lower,
    expected_upper,
) -> None:
    rng = np.random.default_rng(42)

    values = np.empty(401)
    values[0] = -3.0

    for i in range(1, values.size):
        values[i] = (
            0.8 * values[i - 1]
            - 0.6
            + 0.5 * rng.standard_normal()
        )

    trajectory = Trajectory(
        times=np.arange(values.size) * 0.25,
        values=values[:, None, None],
    )

    result = ou.fit_ou_profile(
        trajectory=trajectory,
        alpha_bounds=bounds,
    )

    assert result.success
    assert result.near_lower_bound == expected_lower
    assert result.near_upper_bound == expected_upper
    assert 0.0 <= result.alpha_search_position <= 1.0
    assert result.boundary_fraction == 0.01