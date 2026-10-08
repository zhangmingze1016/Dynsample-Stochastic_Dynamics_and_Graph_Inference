"""Check linear likelihoods against scalar and multivariate references."""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from dynsample.core.linear_model import LinearSDE
from dynsample.core.trajectory import Trajectory
from dynsample.estimation.linear import linear_negative_log_likelihood
from dynsample.estimation.ou import ou_negative_log_likelihood


# One-dimensional linear dynamics must reproduce the scalar OU likelihood.
def test_linear_nll_matches_scalar_ou():
    alpha = 0.7
    mu = 2.0
    sigma = 1.5

    trajectory = Trajectory(
        times=np.array([0.0, 0.1, 0.4, 1.2]),
        values=np.array([3.0, 2.8, 2.4, 2.1]).reshape(-1, 1, 1),
    )
    model = LinearSDE(
        drift=np.array([[-alpha]]),
        offset=np.array([alpha * mu]),
        diffusion=np.array([[sigma]]),
    )

    actual = linear_negative_log_likelihood(trajectory, model)

    expected = ou_negative_log_likelihood(
        trajectory=trajectory,
        mean_reversion=alpha,
        long_run_mean=mu,
        volatility=sigma,
    )

    assert isinstance(actual, float)
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-11,
        atol=1e-12,
    )


# Compare irregular correlated increments with SciPy's Gaussian density.
def test_linear_nll_matches_multivariate_gaussian_reference():
    times = np.array([0.0, 0.2, 0.9])
    observations = np.array([
        [1.0, -0.5],
        [1.1, -0.7],
        [0.8, -0.2],
    ])
    trajectory = Trajectory(
        times=times,
        values=observations[:, :, None],
    )
    model = LinearSDE(
        drift=np.zeros((2, 2)),
        offset=np.array([0.3, -0.2]),
        diffusion=np.array([
            [1.0, 0.0],
            [0.6, 0.8],
        ]),
    )

    expected = 0.0
    noise_rate = np.array([
        [1.0, 0.6],
        [0.6, 1.0],
    ])

    for i, dt in enumerate(np.diff(times), start=1):
        mean = observations[i - 1] + model.offset * dt
        covariance = noise_rate * dt

        expected -= multivariate_normal.logpdf(
            observations[i],
            mean=mean,
            cov=covariance,
        )

    actual = linear_negative_log_likelihood(trajectory, model)

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-11,
        atol=1e-12,
    )


# A rectangular diffusion can produce a positive-definite transition covariance.
def test_linear_nll_accepts_noise_spread_by_coupling():
    dt = 0.5
    initial = np.array([1.0, 2.0])
    observed = np.array([2.4, 3.2])

    trajectory = Trajectory(
        times=np.array([0.0, dt]),
        values=np.stack([initial, observed])[:, :, None],
    )
    model = LinearSDE(
        drift=np.array([
            [0.0, 1.0],
            [0.0, 0.0],
        ]),
        offset=np.array([0.0, 2.0]),
        diffusion=np.array([
            [0.0],
            [1.0],
        ]),
    )

    expected_mean = np.array([
        initial[0] + dt * initial[1] + dt**2,
        initial[1] + 2.0 * dt,
    ])
    expected_covariance = np.array([
        [dt**3 / 3.0, dt**2 / 2.0],
        [dt**2 / 2.0, dt],
    ])

    expected = -multivariate_normal.logpdf(
        observed,
        mean=expected_mean,
        cov=expected_covariance,
    )

    actual = linear_negative_log_likelihood(trajectory, model)

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-11,
        atol=1e-12,
    )


# Conditional scores must add when splitting at an observed state.
def test_linear_nll_is_additive_over_transitions():
    trajectory = Trajectory(
        times=np.array([0.0, 0.2, 0.7]),
        values=np.array([
            [[1.0], [0.0]],
            [[0.8], [0.2]],
            [[0.5], [0.4]],
        ]),
    )
    model = LinearSDE(
        drift=np.array([[-0.7, 0.0], [0.3, -0.5]]),
        offset=np.array([0.2, -0.1]),
        diffusion=np.eye(2),
    )

    left = Trajectory(
        times=trajectory.times[:2],
        values=trajectory.values[:2],
    )
    right = Trajectory(
        times=trajectory.times[1:],
        values=trajectory.values[1:],
    )

    whole_score = linear_negative_log_likelihood(trajectory, model)
    split_score = (
        linear_negative_log_likelihood(left, model)
        + linear_negative_log_likelihood(right, model)
    )

    np.testing.assert_allclose(
        whole_score,
        split_score,
        rtol=1e-12,
        atol=1e-12,
    )


# A narrow Gaussian can have density above one and therefore negative NLL.
def test_linear_nll_can_be_negative():
    trajectory = Trajectory(
        times=np.array([0.0, 1.0]),
        values=np.zeros((2, 1, 1)),
    )
    model = LinearSDE(
        drift=np.zeros((1, 1)),
        offset=np.zeros(1),
        diffusion=np.array([[0.1]]),
    )

    actual = linear_negative_log_likelihood(trajectory, model)
    expected = 0.5 * np.log(2.0 * np.pi * 0.01)

    assert actual < 0.0
    np.testing.assert_allclose(actual, expected, atol=1e-12)


# One observation contains no transition to score.
def test_linear_nll_rejects_single_observation():
    trajectory = Trajectory(
        times=np.array([0.0]),
        values=np.zeros((1, 1, 1)),
    )
    model = LinearSDE(
        drift=np.zeros((1, 1)),
        offset=np.zeros(1),
        diffusion=np.ones((1, 1)),
    )

    with pytest.raises(ValueError, match="at least two time steps"):
        linear_negative_log_likelihood(trajectory, model)


# Keep the current scalar-feature contract explicit.
@pytest.mark.parametrize(
    "shape",
    [
        (2, 3, 1),
        (2, 2, 2),
    ],
)
def test_linear_nll_rejects_incompatible_shape(shape):
    trajectory = Trajectory(
        times=np.array([0.0, 0.5]),
        values=np.zeros(shape),
    )
    model = LinearSDE(
        drift=np.zeros((2, 2)),
        offset=np.zeros(2),
        diffusion=np.eye(2),
    )

    with pytest.raises(ValueError, match="matching nodes and one feature"):
        linear_negative_log_likelihood(trajectory, model)


# Ordinary multivariate density scoring does not support singular covariance.
@pytest.mark.parametrize(
    "diffusion",
    [
        np.zeros((2, 1)),
        np.array([[1.0], [0.0]]),
    ],
)
def test_linear_nll_rejects_singular_covariance(diffusion):
    trajectory = Trajectory(
        times=np.array([0.0, 0.5]),
        values=np.zeros((2, 2, 1)),
    )
    model = LinearSDE(
        drift=np.zeros((2, 2)),
        offset=np.zeros(2),
        diffusion=diffusion,
    )

    with pytest.raises(ValueError, match="positive-definite covariance"):
        linear_negative_log_likelihood(trajectory, model)