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

def test_initial_alpha_bounds_match_time_scales() -> None:
    trajectory = Trajectory(
        times=np.array([0.0, 0.5, 2.0, 5.0]),
        values=np.zeros((4, 1, 1)),
    )

    lower, upper = ou._initial_ou_alpha_bounds(trajectory)

    np.testing.assert_allclose(
        [lower, upper],
        [0.002, 9.210340371976184],
        rtol=1e-12,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "scale, shift",
    [
        (1.0, 100.0),
        (60.0, 0.0),
        (0.5, 10.0),
    ],
)
def test_initial_alpha_bounds_follow_time_units(
    scale: float,
    shift: float,
) -> None:
    times = np.array([0.0, 0.5, 2.0, 5.0])
    values = np.zeros((4, 1, 1))

    original = Trajectory(
        times=times,
        values=values,
    )
    transformed = Trajectory(
        times=scale * times + shift,
        values=values,
    )

    original_bounds = np.array(
        ou._initial_ou_alpha_bounds(original)
    )
    transformed_bounds = np.array(
        ou._initial_ou_alpha_bounds(transformed)
    )

    np.testing.assert_allclose(
        transformed_bounds,
        original_bounds / scale,
        rtol=1e-12,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "times, shape, message",
    [
        ([0.0], (1, 1, 1), "at least two time points"),
        ([0.0, 1.0], (2, 2, 1), "one node and one feature"),
        ([0.0, 1.0], (2, 1, 2), "one node and one feature"),
        (
            [0.0, 1e-308],
            (2, 1, 1),
            "initial alpha bounds must be finite",
        ),
    ],
)
def test_initial_alpha_bounds_reject_invalid_inputs(
    times,
    shape,
    message,
) -> None:
    trajectory = Trajectory(
        times=np.array(times),
        values=np.zeros(shape),
    )

    with pytest.raises(ValueError, match=message):
        ou._initial_ou_alpha_bounds(trajectory)


def test_profile_scan_rejects_all_failed_evaluations(
    monkeypatch,
) -> None:
    trajectory = Trajectory(
        times=np.array([0.0, 1.0, 2.0]),
        values=np.array([1.0, 4.0, 3.0])[:, None, None],
    )

    def objective(trajectory, alpha):
        raise ValueError("deliberate test failure")

    monkeypatch.setattr(
        ou,
        "_profile_ou_negative_log_likelihood",
        objective,
    )

    with pytest.raises(
        RuntimeError,
        match="all profile grid evaluations failed",
    ):
        ou._scan_ou_profile(
            trajectory,
            alpha_bounds=(0.1, 10.0),
            grid_size=5,
        )

def test_profile_scan_finds_multiple_candidate_intervals(
    monkeypatch,
) -> None:
    trajectory = Trajectory(
        times=np.array([0.0, 1.0, 2.0]),
        values=np.array([1.0, 4.0, 3.0])[:, None, None],
    )

    def objective(trajectory, alpha):
        u = np.log(alpha)
        return (u**2 - 1.0)**2

    monkeypatch.setattr(
        ou,
        "_profile_ou_negative_log_likelihood",
        objective,
    )

    result = ou._scan_ou_profile(
        trajectory,
        alpha_bounds=(np.exp(-2.0), np.exp(2.0)),
        grid_size=5,
    )

    np.testing.assert_allclose(
        np.log(result.alphas),
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        atol=1e-14,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.scores,
        [9.0, 0.0, 1.0, 0.0, 9.0],
        atol=1e-14,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.candidate_intervals,
        [
            [np.exp(-2.0), 1.0],
            [1.0, np.exp(2.0)],
        ],
        atol=1e-14,
        rtol=0.0,
    )

    assert result.best_index in (1, 3)
    assert result.failures == {}


def test_profile_scan_records_failure_without_crossing_it(
    monkeypatch,
) -> None:
    trajectory = Trajectory(
        times=np.array([0.0, 1.0, 2.0]),
        values=np.array([1.0, 4.0, 3.0])[:, None, None],
    )

    def objective(trajectory, alpha):
        u = np.log(alpha)

        if abs(u) < 0.1:
            raise ValueError("deliberate test failure")

        return u**2

    monkeypatch.setattr(
        ou,
        "_profile_ou_negative_log_likelihood",
        objective,
    )

    result = ou._scan_ou_profile(
        trajectory,
        alpha_bounds=(np.exp(-2.0), np.exp(2.0)),
        grid_size=5,
    )

    assert np.isnan(result.scores[2])
    assert result.failures == {2: "deliberate test failure"}
    assert result.best_index in (1, 3)
    assert result.candidate_intervals == []