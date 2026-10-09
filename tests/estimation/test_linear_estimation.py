"""Check linear likelihoods against scalar and multivariate references."""

import numpy as np
import pytest
from scipy.stats import multivariate_normal
from scipy.optimize import OptimizeResult, minimize_scalar
import dynsample.estimation.linear as linear_estimation

from dynsample.estimation.linear import fit_linear_drift

from dynsample.core.linear_model import LinearSDE
from dynsample.core.trajectory import Trajectory
from dynsample.estimation.linear import (
    fit_linear_offset,
    linear_negative_log_likelihood,
)
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

# With zero K, the fitted offset equals total displacement / total duration.
def test_fit_linear_offset_matches_brownian_drift():
    trajectory = Trajectory(
        times=np.array([0.0, 0.2, 0.9, 2.0]),
        values=np.array([
            [1.0, -1.0],
            [1.3, -0.8],
            [0.9, 0.2],
            [2.5, 0.5],
        ])[:, :, None],
    )
    drift = np.zeros((2, 2))
    diffusion = np.array([
        [1.0, 0.0],
        [0.6, 0.8],
    ])

    result = fit_linear_offset(
        trajectory=trajectory,
        drift=drift,
        diffusion=diffusion,
    )

    expected = np.array([0.75, 0.75])

    np.testing.assert_allclose(
        result.offset, expected, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(result.x, result.offset)
    np.testing.assert_allclose(result.model.offset, result.offset)
    np.testing.assert_array_equal(result.model.drift, drift)
    np.testing.assert_array_equal(result.model.diffusion, diffusion)

    assert result.success
    assert result.rank == 2
    assert np.isfinite(result.fun)


# Compare the scalar fit with an independently derived weighted formula.
def test_fit_linear_offset_matches_scalar_formula():
    times = np.array([0.0, 0.1, 0.4, 1.2])
    values = np.array([3.0, 2.8, 2.4, 2.1])
    alpha = 0.7
    sigma = 1.5

    trajectory = Trajectory(
        times=times,
        values=values[:, None, None],
    )

    dt = np.diff(times)
    transition = np.exp(-alpha * dt)
    integrated_transition = -np.expm1(-alpha * dt) / alpha
    variance = (
        sigma**2 * -np.expm1(-2.0 * alpha * dt)
        / (2.0 * alpha)
    )
    response = values[1:] - transition * values[:-1]

    expected = (
        np.sum(integrated_transition * response / variance)
        / np.sum(integrated_transition**2 / variance)
    )

    result = fit_linear_offset(
        trajectory=trajectory,
        drift=np.array([[-alpha]]),
        diffusion=np.array([[sigma]]),
    )

    np.testing.assert_allclose(
        result.offset, [expected], rtol=1e-11, atol=1e-12
    )

    expected_score = ou_negative_log_likelihood(
        trajectory=trajectory,
        mean_reversion=alpha,
        long_run_mean=expected / alpha,
        volatility=sigma,
    )
    np.testing.assert_allclose(
        result.fun, expected_score, rtol=1e-11, atol=1e-12
    )


# A singular coupled K must work without a matrix inverse.
def test_fit_linear_offset_handles_singular_coupled_drift():
    dt = 0.5
    initial = np.array([1.0, 2.0])
    expected_offset = np.array([0.3, -0.4])

    drift = np.array([
        [0.0, 1.0],
        [0.0, 0.0],
    ])

    # Since K squared is zero, these expressions are exact.
    transition = np.eye(2) + drift * dt
    integrated_transition = (
        np.eye(2) * dt + drift * dt**2 / 2.0
    )
    final = (
        transition @ initial
        + integrated_transition @ expected_offset
    )

    trajectory = Trajectory(
        times=np.array([0.0, dt]),
        values=np.stack([initial, final])[:, :, None],
    )

    result = fit_linear_offset(
        trajectory=trajectory,
        drift=drift,
        diffusion=np.eye(2),
    )

    np.testing.assert_allclose(
        result.offset, expected_offset, rtol=1e-11, atol=1e-12
    )


# At least one transition is required to estimate an offset.
def test_fit_linear_offset_rejects_single_observation():
    trajectory = Trajectory(
        times=np.array([0.0]),
        values=np.zeros((1, 1, 1)),
    )

    with pytest.raises(ValueError, match="at least two time steps"):
        fit_linear_offset(
            trajectory=trajectory,
            drift=np.zeros((1, 1)),
            diffusion=np.ones((1, 1)),
        )


# The current estimator supports one feature per node.
def test_fit_linear_offset_rejects_multiple_features():
    trajectory = Trajectory(
        times=np.array([0.0, 1.0]),
        values=np.zeros((2, 2, 2)),
    )

    with pytest.raises(ValueError, match="one feature per node"):
        fit_linear_offset(
            trajectory=trajectory,
            drift=np.zeros((2, 2)),
            diffusion=np.eye(2),
        )


# Ordinary Gaussian likelihood requires positive-definite transition Q.
def test_fit_linear_offset_rejects_singular_covariance():
    trajectory = Trajectory(
        times=np.array([0.0, 1.0]),
        values=np.zeros((2, 2, 1)),
    )

    with pytest.raises(ValueError, match="positive-definite covariance"):
        fit_linear_offset(
            trajectory=trajectory,
            drift=np.zeros((2, 2)),
            diffusion=np.array([
                [1.0],
                [0.0],
            ]),
        )
# Compare scalar drift fitting with an independent scalar likelihood.
def test_fit_linear_drift_matches_scalar_reference():
    times = np.arange(9, dtype=np.float64) * 0.5
    values = np.array([
        1.5, 0.8, 0.6, 0.1, -0.2,
        0.2, 0.7, 0.3, -0.1,
    ])
    sigma = 0.7
    dt = 0.5

    trajectory = Trajectory(
        times=times,
        values=values[:, None, None],
    )

    # On a regular grid, the optimal transition offset is mean(y - F*x).
    def reference_score(k):
        transition = np.exp(k * dt)
        response = values[1:] - transition * values[:-1]
        transition_offset = np.mean(response)

        if k == 0.0:
            variance = sigma**2 * dt
        else:
            variance = (
                sigma**2 * np.expm1(2.0 * k * dt) / (2.0 * k)
            )

        residual = response - transition_offset

        return float(
            0.5 * np.sum(
                np.log(2.0 * np.pi * variance)
                + residual**2 / variance
            )
        )

    reference = minimize_scalar(
        reference_score,
        bounds=(-5.0, -0.01),
        method="bounded",
        options={"xatol": 1e-10},
    )

    assert reference.success
    assert -4.99 < reference.x < -0.02

    result = fit_linear_drift(
        trajectory=trajectory,
        diffusion=np.array([[sigma]]),
        drift_mask=np.array([[True]]),
        initial_drift=np.array([[-1.0]]),
    )

    expected_k = float(reference.x)
    expected_transition = np.exp(expected_k * dt)
    expected_integral = np.expm1(expected_k * dt) / expected_k
    expected_b = np.mean(
        values[1:] - expected_transition * values[:-1]
    ) / expected_integral

    assert result.success, result.message

    np.testing.assert_allclose(
        result.drift,
        [[expected_k]],
        rtol=1e-4,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        result.offset,
        [expected_b],
        rtol=1e-4,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        result.fun,
        reference.fun,
        rtol=0.0,
        atol=1e-7,
    )


# Only permitted entries may change, and input arrays must stay unchanged.
def test_fit_linear_drift_respects_mask_and_preserves_inputs():
    trajectory = Trajectory(
        times=np.array([0.0, 0.2, 0.5, 0.9, 1.4]),
        values=np.array([
            [1.0, -0.5],
            [0.8, -0.2],
            [0.9, 0.1],
            [0.4, -0.1],
            [0.2, 0.3],
        ])[:, :, None],
    )
    mask = np.array([
        [True, False],
        [True, True],
    ])
    initial = np.array([
        [-1.0, 0.0],
        [0.2, -0.5],
    ])
    diffusion = np.array([
        [0.7, 0.0],
        [0.2, 0.6],
    ])

    original_mask = mask.copy()
    original_initial = initial.copy()
    original_diffusion = diffusion.copy()

    initial_fit = fit_linear_offset(
        trajectory=trajectory,
        drift=initial,
        diffusion=diffusion,
    )

    result = fit_linear_drift(
        trajectory=trajectory,
        diffusion=diffusion,
        drift_mask=mask,
        initial_drift=initial,
        maxiter=20,
    )

    assert np.isfinite(result.fun)
    assert result.fun <= initial_fit.fun + 1e-10
    assert result.drift[0, 1] == 0.0

    np.testing.assert_array_equal(result.x, result.drift[mask])
    np.testing.assert_array_equal(mask, original_mask)
    np.testing.assert_array_equal(initial, original_initial)
    np.testing.assert_array_equal(diffusion, original_diffusion)

    np.testing.assert_allclose(
        result.fun,
        linear_negative_log_likelihood(
            trajectory=trajectory,
            model=result.model,
        ),
    )


# With no free drift entries, fit only the Brownian drift offset.
def test_fit_linear_drift_handles_all_false_mask():
    trajectory = Trajectory(
        times=np.array([0.0, 0.2, 0.9, 2.0]),
        values=np.array([
            [1.0, -1.0],
            [1.3, -0.8],
            [0.9, 0.2],
            [2.5, 0.5],
        ])[:, :, None],
    )

    result = fit_linear_drift(
        trajectory=trajectory,
        diffusion=np.eye(2),
        drift_mask=np.zeros((2, 2), dtype=bool),
    )

    np.testing.assert_array_equal(result.drift, np.zeros((2, 2)))
    np.testing.assert_allclose(result.offset, [0.75, 0.75])

    assert result.success
    assert result.x.size == 0
    assert result.optimizer_result is None
    assert result.spectral_abscissa == 0.0
    assert not result.is_stable


# An optimizer failure must remain a failure even with a finite candidate.
def test_fit_linear_drift_preserves_optimizer_failure(monkeypatch):
    trajectory = Trajectory(
        times=np.array([0.0, 0.5, 1.0]),
        values=np.array([1.0, 0.8, 0.3])[:, None, None],
    )

    def fake_minimize(fun, x0, method, options):
        return OptimizeResult(
            x=x0.copy(),
            fun=fun(x0),
            success=False,
            message="Iteration budget exhausted.",
        )

    monkeypatch.setattr(
        linear_estimation,
        "minimize",
        fake_minimize,
    )

    result = fit_linear_drift(
        trajectory=trajectory,
        diffusion=np.array([[0.5]]),
        drift_mask=np.array([[True]]),
        initial_drift=np.array([[-1.0]]),
    )

    assert not result.success
    assert "Iteration budget exhausted" in result.message
    assert np.isfinite(result.fun)

    np.testing.assert_allclose(result.drift, [[-1.0]])
    np.testing.assert_allclose(result.spectral_abscissa, -1.0)
    assert result.is_stable


# Reject malformed masks, conflicting starts, and invalid iteration budgets.
@pytest.mark.parametrize(
    "mask, initial, maxiter, message",
    [
        (
            np.array([True]),
            None,
            10,
            "drift_mask must have shape",
        ),
        (
            np.array([[1]]),
            None,
            10,
            "boolean",
        ),
        (
            np.array([[False]]),
            np.array([[-1.0]]),
            10,
            "zero outside drift_mask",
        ),
        (
            np.array([[True]]),
            None,
            0,
            "positive integer",
        ),
    ],
)
def test_fit_linear_drift_rejects_invalid_configuration(
    mask,
    initial,
    maxiter,
    message,
):
    trajectory = Trajectory(
        times=np.array([0.0, 1.0]),
        values=np.array([1.0, 0.5])[:, None, None],
    )

    with pytest.raises(ValueError, match=message):
        fit_linear_drift(
            trajectory=trajectory,
            diffusion=np.array([[0.5]]),
            drift_mask=mask,
            initial_drift=initial,
            maxiter=maxiter,
        )