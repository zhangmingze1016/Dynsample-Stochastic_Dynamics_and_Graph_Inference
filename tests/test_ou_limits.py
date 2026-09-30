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