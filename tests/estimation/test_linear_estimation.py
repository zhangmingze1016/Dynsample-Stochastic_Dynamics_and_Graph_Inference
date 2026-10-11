"""Check linear likelihoods against scalar and multivariate references."""

import numpy as np
import pytest
from scipy.stats import multivariate_normal
from scipy.optimize import OptimizeResult, minimize_scalar
import dynsample.estimation.linear as linear_estimation
from unittest.mock import patch
from dynsample.core.state import State
from dynsample.estimation.linear import fit_linear_drift
import dynsample.simulation.linear as linear_simulation
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

# Derive the scalar fixed-K,B offset MLE and NLL without matrix routines.
def _scalar_offset_cache_reference(times, values, alpha, sigma):
    dt = np.diff(times)

    transition = np.exp(-alpha * dt)
    integrated_transition = -np.expm1(-alpha * dt) / alpha
    variance = (
        sigma**2
        * -np.expm1(-2.0 * alpha * dt)
        / (2.0 * alpha)
    )

    response = values[1:] - transition * values[:-1]

    fitted_offset = (
        np.sum(integrated_transition * response / variance)
        / np.sum(integrated_transition**2 / variance)
    )

    residual = response - integrated_transition * fitted_offset

    score = 0.5 * np.sum(
        np.log(2.0 * np.pi)
        + np.log(variance)
        + residual**2 / variance
    )

    return float(fitted_offset), float(score)


# Repeated and distinct intervals must preserve the mathematical solution.
@pytest.mark.parametrize(
    "times",
    [
        np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
        np.array([0.0, 0.25, 0.75, 1.0, 1.5]),
        np.array([0.0, 0.125, 0.375, 0.875, 1.875]),
    ],
    ids=["regular", "repeated-irregular", "all-distinct"],
)
def test_fit_linear_offset_cache_matches_scalar_reference(times):
    values = np.array([1.2, 0.7, 1.0, 0.4, 0.8])
    alpha = 0.7
    sigma = 0.6

    trajectory = Trajectory(
        times=times,
        values=values[:, None, None],
    )

    expected_offset, expected_score = _scalar_offset_cache_reference(
        times, values, alpha, sigma
    )

    result = fit_linear_offset(
        trajectory=trajectory,
        drift=np.array([[-alpha]]),
        diffusion=np.array([[sigma]]),
    )

    assert result.success
    np.testing.assert_allclose(
        result.offset,
        [expected_offset],
        rtol=1e-11,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.fun,
        expected_score,
        rtol=1e-11,
        atol=1e-12,
    )


# Count expensive operations and verify exact, rather than rounded, cache keys.
@pytest.mark.parametrize(
    "times, expected_unique",
    [
        (
            np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
            1,
        ),
        (
            np.array([0.0, 0.25, 0.75, 1.0, 1.5]),
            2,
        ),
        (
            np.array([0.0, 1.0, np.nextafter(2.0, np.inf)]),
            2,
        ),
    ],
    ids=["one-interval", "two-intervals", "nearby-distinct-intervals"],
)
def test_fit_linear_offset_cache_reuses_factorizations(
    times, expected_unique
):
    all_values = np.array([
        [1.2, -0.3],
        [0.7, 0.2],
        [1.0, 0.1],
        [0.4, -0.2],
        [0.8, 0.3],
    ])

    trajectory = Trajectory(
        times=times,
        values=all_values[:times.size, :, None],
    )
    drift = np.array([
        [-0.7, 0.0],
        [0.2, -1.0],
    ])
    diffusion = np.array([
        [0.6, 0.0],
        [0.2, 0.5],
    ])

    assert np.unique(np.diff(times)).size == expected_unique

    with (
        patch.object(
            linear_simulation,
            "expm",
            wraps=linear_simulation.expm,
        ) as transition_expm,
        patch.object(
            linear_estimation,
            "expm",
            wraps=linear_estimation.expm,
        ) as integral_expm,
        patch.object(
            np.linalg,
            "cholesky",
            wraps=np.linalg.cholesky,
        ) as cholesky,
    ):
        result = fit_linear_offset(
            trajectory=trajectory,
            drift=drift,
            diffusion=diffusion,
        )

        # Two exponentials inside linear_transition per unique interval.
        assert transition_expm.call_count == 2 * expected_unique

        # One offset-integral exponential per unique interval.
        assert integral_expm.call_count == expected_unique

        # One covariance factorization per unique interval.
        assert cholesky.call_count == expected_unique

    # Run the separate NLL implementation after operation counting ends.
    expected_score = linear_negative_log_likelihood(
        trajectory=trajectory,
        model=result.model,
    )

    np.testing.assert_allclose(
        result.fun,
        expected_score,
        rtol=1e-11,
        atol=1e-11,
    )


# Every fit must rebuild factors when K or B changes between calls.
@pytest.mark.parametrize(
    "second_alpha, second_sigma",
    [
        (1.4, 0.6),
        (0.7, 1.2),
    ],
    ids=["changed-drift", "changed-diffusion"],
)
def test_fit_linear_offset_cache_is_local_to_each_call(
    second_alpha, second_sigma
):
    times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    values = np.array([1.2, 0.7, 1.0, 0.4, 0.8])

    trajectory = Trajectory(
        times=times,
        values=values[:, None, None],
    )

    with patch.object(
        np.linalg,
        "cholesky",
        wraps=np.linalg.cholesky,
    ) as cholesky:
        fit_linear_offset(
            trajectory=trajectory,
            drift=np.array([[-0.7]]),
            diffusion=np.array([[0.6]]),
        )

        first_call_count = cholesky.call_count

        result = fit_linear_offset(
            trajectory=trajectory,
            drift=np.array([[-second_alpha]]),
            diffusion=np.array([[second_sigma]]),
        )

        assert first_call_count == 1
        assert cholesky.call_count == 2

    expected_offset, expected_score = _scalar_offset_cache_reference(
        times,
        values,
        second_alpha,
        second_sigma,
    )

    np.testing.assert_allclose(
        result.offset,
        [expected_offset],
        rtol=1e-11,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.fun,
        expected_score,
        rtol=1e-11,
        atol=1e-12,
    )

# Small positive variances must retain their actual Gaussian density.
@pytest.mark.parametrize("epsilon", [1e-3, 1e-6])
def test_linear_nll_near_singular_diagonal_reference(epsilon):
    times = np.array([0.0, 0.25, 1.0])
    intervals = np.diff(times)

    scales = np.array([1.0, epsilon])

    # These are prescribed standardized increments, not random draws.
    standardized_increments = np.array([
        [0.5, -1.2],
        [-0.8, 0.3],
    ])

    increments = (
        np.sqrt(intervals)[:, None]
        * scales[None, :]
        * standardized_increments
    )

    values = np.vstack((
        np.zeros((1, 2)),
        np.cumsum(increments, axis=0),
    ))

    trajectory = Trajectory(
        times=times,
        values=values[:, :, None],
    )
    model = LinearSDE(
        drift=np.zeros((2, 2)),
        offset=np.zeros(2),
        diffusion=np.diag(scales),
    )

    # Independent scalar-normal formulas for the two coordinates.
    variances = intervals[:, None] * scales[None, :]**2

    expected = 0.5 * np.sum(
        np.log(2.0 * np.pi)
        + np.log(variances)
        + standardized_increments**2
    )

    actual = linear_negative_log_likelihood(
        trajectory=trajectory,
        model=model,
    )

    assert np.isfinite(actual)
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-10,
        atol=1e-10,
    )


# Offset fitting must preserve accuracy across very different noise scales.
@pytest.mark.parametrize("epsilon", [1e-3, 1e-6])
def test_fit_linear_offset_near_singular_diagonal_reference(epsilon):
    times = np.array([0.0, 0.25, 0.75, 1.0])
    intervals = np.diff(times)
    scales = np.array([1.0, epsilon])

    base_values = np.array([
        [0.0, 0.0],
        [0.2, -0.3],
        [-0.1, 0.4],
        [0.6, 0.2],
    ])
    values = base_values * scales[None, :]

    trajectory = Trajectory(
        times=times,
        values=values[:, :, None],
    )

    result = fit_linear_offset(
        trajectory=trajectory,
        drift=np.zeros((2, 2)),
        diffusion=np.diag(scales),
    )

    # With K=0 and constant known B, b_hat is displacement / duration.
    duration = times[-1] - times[0]
    expected_offset = (values[-1] - values[0]) / duration

    residuals = (
        np.diff(values, axis=0)
        - intervals[:, None] * expected_offset[None, :]
    )
    standard_deviations = (
        np.sqrt(intervals)[:, None] * scales[None, :]
    )
    standardized_residuals = residuals / standard_deviations

    expected_score = 0.5 * np.sum(
        np.log(2.0 * np.pi)
        + 2.0 * np.log(standard_deviations)
        + standardized_residuals**2
    )

    assert result.success
    assert result.rank == 2

    # Compare in each coordinate's own scale so the small component matters.
    np.testing.assert_allclose(
        result.offset / scales,
        expected_offset / scales,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        result.fun,
        expected_score,
        rtol=1e-10,
        atol=1e-10,
    )

# Build one coupled trajectory for checking changes of time units.
@pytest.fixture
def linear_time_rescaling_case():
    model = LinearSDE(
        drift=np.array([
            [-0.7, 0.0],
            [0.3, -1.0],
        ]),
        offset=np.array([0.4, -0.2]),
        diffusion=np.diag([0.6, 0.5]),
    )

    times = np.concatenate((
        [0.0],
        np.cumsum(np.tile([0.25, 0.75], 40)),
    ))

    trajectory = linear_simulation.simulate_linear(
        initial_state=State(
            time=0.0,
            values=np.array([[2.0], [-1.0]]),
        ),
        times=times,
        model=model,
        rng=np.random.default_rng(42),
    )

    return trajectory, model


# Changing time units must preserve the conditional density of the states.
@pytest.mark.parametrize("scale", [0.001, 1000.0])
def test_linear_nll_respects_time_rescaling(
    linear_time_rescaling_case,
    scale,
):
    trajectory, model = linear_time_rescaling_case

    scaled_trajectory = Trajectory(
        times=trajectory.times * scale,
        values=trajectory.values,
    )
    scaled_model = LinearSDE(
        drift=model.drift / scale,
        offset=model.offset / scale,
        diffusion=model.diffusion / np.sqrt(scale),
    )

    original_score = linear_negative_log_likelihood(
        trajectory=trajectory,
        model=model,
    )
    scaled_score = linear_negative_log_likelihood(
        trajectory=scaled_trajectory,
        model=scaled_model,
    )

    np.testing.assert_allclose(
        scaled_score,
        original_score,
        rtol=1e-10,
        atol=1e-10,
    )


# The estimated offset must transform as b_new = b / scale.
@pytest.mark.parametrize("scale", [0.001, 1000.0])
def test_fit_linear_offset_respects_time_rescaling(
    linear_time_rescaling_case,
    scale,
):
    trajectory, model = linear_time_rescaling_case

    scaled_trajectory = Trajectory(
        times=trajectory.times * scale,
        values=trajectory.values,
    )

    original = fit_linear_offset(
        trajectory=trajectory,
        drift=model.drift,
        diffusion=model.diffusion,
    )
    scaled = fit_linear_offset(
        trajectory=scaled_trajectory,
        drift=model.drift / scale,
        diffusion=model.diffusion / np.sqrt(scale),
    )

    assert original.success
    assert scaled.success
    assert original.rank == scaled.rank

    # Convert back to the original units before comparing.
    np.testing.assert_allclose(
        scaled.offset * scale,
        original.offset,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        scaled.fun,
        original.fun,
        rtol=1e-10,
        atol=1e-10,
    )


# Drift fitting should recover equivalent candidates in different time units.
@pytest.mark.parametrize("scale", [0.001, 1000.0])
def test_fit_linear_drift_respects_time_rescaling(
    linear_time_rescaling_case,
    scale,
):
    trajectory, model = linear_time_rescaling_case

    mask = np.array([
        [True, False],
        [True, True],
    ])
    initial_drift = np.array([
        [-0.5, 0.0],
        [0.1, -0.8],
    ])

    scaled_trajectory = Trajectory(
        times=trajectory.times * scale,
        values=trajectory.values,
    )

    original = fit_linear_drift(
        trajectory=trajectory,
        diffusion=model.diffusion,
        drift_mask=mask,
        initial_drift=initial_drift,
        maxiter=150,
    )
    scaled = fit_linear_drift(
        trajectory=scaled_trajectory,
        diffusion=model.diffusion / np.sqrt(scale),
        drift_mask=mask,
        initial_drift=initial_drift / scale,
        maxiter=150,
    )

    assert original.success, original.message
    assert scaled.success, scaled.message

    # Numerical optimization needs looser tolerances than direct evaluation.
    np.testing.assert_allclose(
        scaled.drift * scale,
        original.drift,
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        scaled.offset * scale,
        original.offset,
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        scaled.fun,
        original.fun,
        rtol=1e-8,
        atol=1e-8,
    )

    # Changing units must not change the supplied structural restriction.
    np.testing.assert_array_equal(
        original.drift[~mask],
        np.zeros(np.count_nonzero(~mask)),
    )
    np.testing.assert_array_equal(
        scaled.drift[~mask],
        np.zeros(np.count_nonzero(~mask)),
    )