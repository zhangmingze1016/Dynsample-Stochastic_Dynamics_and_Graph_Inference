import numpy as np
import pytest

from dynsample.core.trajectory import Trajectory
from dynsample.estimation import ou
from dynsample.estimation.brownian import fit_brownian_drift

@pytest.fixture
def irregular_trajectory() -> Trajectory:
    times = np.array([0.0, 0.2, 0.9, 2.4, 3.0, 5.0])
    values = np.array([2.0, 2.4, 1.1, 3.2, 2.8, 4.1])

    return Trajectory(
        times=times,
        values=values[:, None, None],
    )

def test_ou_small_alpha_matches_brownian_boundary(
    irregular_trajectory: Trajectory,
) -> None:
    trajectory = irregular_trajectory
    boundary = fit_brownian_drift(trajectory)
    alpha = 1e-7

    mu = ou._profile_ou_mu(trajectory, alpha)
    sigma = ou._profile_ou_sigma(trajectory, alpha)
    score = ou._profile_ou_negative_log_likelihood(
        trajectory, alpha
    )

    np.testing.assert_allclose(
        alpha * mu,
        boundary.drift,
        rtol = 0.0,
        atol = 1e-6,
    )
    np.testing.assert_allclose(
        sigma,
        boundary.volatility,
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        score,
        boundary.fun,
        rtol=0.0,
        atol=1e-6,
    )



def test_ou_large_alpha_matches_gaussian_boundary(
    irregular_trajectory: Trajectory,
) -> None:
    trajectory = irregular_trajectory
    targets = trajectory.values[1:, 0, 0]

    expected_mu = np.mean(targets)
    expected_variance = np.mean(
        (targets - expected_mu) ** 2
    )
    expected_score = 0.5 * targets.size * (
        np.log(2.0 * np.pi)
        + np.log(expected_variance)
        + 1.0
    )

    alpha = 1000.0

    mu = ou._profile_ou_mu(trajectory, alpha)
    sigma = ou._profile_ou_sigma(trajectory, alpha)
    score = ou._profile_ou_negative_log_likelihood(
        trajectory, alpha
    )

    np.testing.assert_allclose(
        mu,
        expected_mu,
        rtol=0.0,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        sigma**2 / (2.0 * alpha),
        expected_variance,
        rtol=0.0,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        score,
        expected_score,
        rtol=0.0,
        atol=1e-8,
    )

def test_ou_boundary_scores_match_hand_calculation() -> None:
    trajectory = Trajectory(
        times=np.array([0.0, 1.0, 2.0]),
        values=np.array([1.0, 4.0, 3.0])[:, None, None],
    )

    results = ou._ou_boundary_scores(trajectory)

    brownian = results["brownian"]
    gaussian = results["gaussian"]

    expected_brownian_score = np.log(2.0 * np.pi * 4.0) + 1.0
    expected_gaussian_score = np.log(2.0 * np.pi * 0.25) + 1.0

    np.testing.assert_allclose(
        [brownian.drift, brownian.volatility, brownian.fun],
        [1.0, 2.0, expected_brownian_score],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        [gaussian.mean, gaussian.variance, gaussian.fun],
        [3.5, 0.25, expected_gaussian_score],
        rtol=1e-12,
        atol=1e-12,
    )

    assert brownian.success
    assert gaussian.success


@pytest.mark.parametrize(
    "values, message",
    [
        ([1.0, 2.0, 3.0], "zero residual variance"),
        (
            [1.0, 3.0, 3.0],
            "Gaussian boundary variance must be finite and positive",
        ),
    ],
)
def test_ou_boundary_scores_reject_degenerate_data(
    values,
    message,
) -> None:
    trajectory = Trajectory(
        times=np.array([0.0, 1.0, 2.0]),
        values=np.array(values)[:, None, None],
    )

    with pytest.raises(ValueError, match=message):
        ou._ou_boundary_scores(trajectory)

def test_profile_scan_finds_multiple_candidate_intervals(
    monkeypatch,
) -> None:
    trajectory = Trajectory(
        times=np.array([0.0, 1.0, 2.0]),
        values=np.array([1.0, 4.0, 3.0])[:, None, None],
    )

    def objective(trajectory, alpha):
        u = np.log(alpha)
        return (u**2 - 1.0) ** 2

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
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.scores,
        [9.0, 0.0, 1.0, 0.0, 9.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.candidate_intervals,
        [
            [np.exp(-2.0), 1.0],
            [1.0, np.exp(2.0)],
        ],
        atol=1e-12,
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

