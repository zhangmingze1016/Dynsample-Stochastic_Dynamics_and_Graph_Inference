"""Check linear transitions against analytical solutions and composition."""
from dynsample.core.state import State
from dynsample.simulation.linear import linear_step
import numpy as np
import pytest

from dynsample.core.linear_model import LinearSDE
from dynsample.simulation.linear import linear_transition
from dynsample.simulation.ou import ou_transition


def test_linear_transition_zero_time():
    model = LinearSDE(np.array([[-1.0]]), np.array([2.0]), np.array([[3.0]]))
    transition, offset, covariance = linear_transition(model, 0.0)
    np.testing.assert_array_equal(transition, [[1.0]])
    np.testing.assert_array_equal(offset, [0.0])
    np.testing.assert_array_equal(covariance, [[0.0]])


@pytest.mark.parametrize("dt", [-0.1, np.nan, np.inf, -np.inf])
def test_linear_transition_rejects_invalid_time(dt):
    model = LinearSDE(np.zeros((1, 1)), np.zeros(1), np.ones((1, 1)))
    with pytest.raises(ValueError, match="dt must be"):
        linear_transition(model, dt)


@pytest.mark.parametrize("dt", [1e-8, 0.2, 2.0])
def test_linear_transition_matches_scalar_ou(dt):
    alpha, mu, sigma = 0.7, 2.0, 1.5
    model = LinearSDE(
        np.array([[-alpha]]), np.array([alpha * mu]), np.array([[sigma]])
    )
    transition, offset, covariance = linear_transition(model, dt)
    expected_f, expected_c, expected_q = ou_transition(dt, alpha, mu, sigma)
    np.testing.assert_allclose(transition, [[expected_f]], rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(offset, [expected_c], rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(covariance, [[expected_q]], rtol=1e-10, atol=1e-14)


def test_linear_transition_zero_drift_shared_noise():
    # Rectangular diffusion: both states share one Brownian source.
    model = LinearSDE(np.zeros((2, 2)), np.array([1.0, -2.0]), np.array([[2.0], [1.0]]))
    transition, offset, covariance = linear_transition(model, 0.25)
    np.testing.assert_allclose(transition, np.eye(2))
    np.testing.assert_allclose(offset, [0.25, -0.5])
    np.testing.assert_allclose(covariance, [[1.0, 0.5], [0.5, 0.25]])


def test_linear_transition_singular_directed_drift():
    # K squared is zero, so exp(K*t) = I + K*t; all integrals are polynomials.
    dt = 0.4
    model = LinearSDE(
        np.array([[0.0, 1.0], [0.0, 0.0]]),
        np.array([0.0, 2.0]),
        np.array([[0.0], [1.0]]),
    )
    transition, offset, covariance = linear_transition(model, dt)
    np.testing.assert_allclose(transition, [[1.0, dt], [0.0, 1.0]], atol=1e-14)
    np.testing.assert_allclose(offset, [dt**2, 2 * dt], atol=1e-14)
    np.testing.assert_allclose(
        covariance, [[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]], atol=1e-14
    )


def test_linear_transition_zero_diffusion():
    model = LinearSDE(np.array([[0.5]]), np.array([1.0]), np.zeros((1, 1)))
    transition, offset, covariance = linear_transition(model, 0.4)
    np.testing.assert_allclose(transition, [[np.exp(0.2)]])
    np.testing.assert_allclose(offset, [2 * np.expm1(0.2)])
    np.testing.assert_array_equal(covariance, [[0.0]])


def test_linear_transition_composition_and_covariance():
    model = LinearSDE(
        np.array([[-0.8, 0.4], [-0.2, -0.5]]),
        np.array([0.3, -0.7]),
        np.array([[1.0, 0.2], [0.0, 0.6]]),
    )
    f1, c1, q1 = linear_transition(model, 0.2)
    f2, c2, q2 = linear_transition(model, 0.7)
    transition, offset, covariance = linear_transition(model, 0.9)
    np.testing.assert_allclose(transition, f2 @ f1, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(offset, f2 @ c1 + c2, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(covariance, f2 @ q1 @ f2.T + q2, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-13)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-12


def test_linear_transition_rejects_overflowed_blocks():
    model = LinearSDE(np.zeros((1, 1)), np.zeros(1), np.array([[1e200]]))
    with pytest.raises(FloatingPointError, match="transition blocks"):
        linear_transition(model, 1.0)

# Check the deterministic solution when diffusion is zero.
def test_linear_step_zero_diffusion():
    model = LinearSDE(
        drift=np.zeros((2, 2)),
        offset=np.array([1.0, -2.0]),
        diffusion=np.zeros((2, 1)),
    )
    state = State(
        time=1.0,
        values=np.array([[2.0], [3.0]]),
    )
    original_values = state.values.copy()

    result = linear_step(
        state=state,
        next_time=1.5,
        model=model,
        rng=np.random.default_rng(42),
    )

    assert isinstance(result, State)
    assert result.time == 1.5
    assert result.values.shape == (2, 1)
    np.testing.assert_allclose(result.values, [[2.5], [2.0]])
    np.testing.assert_array_equal(state.values, original_values)
    assert state.time == 1.0


# Reject times that cannot represent a forward step.
@pytest.mark.parametrize(
    "next_time",
    [1.0, 0.5, np.nan, np.inf, -np.inf],
)
def test_linear_step_rejects_invalid_next_time(next_time):
    model = LinearSDE(
        drift=np.zeros((1, 1)),
        offset=np.zeros(1),
        diffusion=np.ones((1, 1)),
    )
    state = State(
        time=1.0,
        values=np.array([[0.0]]),
    )

    with pytest.raises(ValueError, match="next_time must"):
        linear_step(
            state=state,
            next_time=next_time,
            model=model,
            rng=np.random.default_rng(42),
        )


# Require matching node counts and exactly one feature per node.
@pytest.mark.parametrize(
    "values",
    [
        np.zeros((3, 1)),
        np.zeros((2, 2)),
    ],
)
def test_linear_step_rejects_incompatible_state(values):
    model = LinearSDE(
        drift=np.zeros((2, 2)),
        offset=np.zeros(2),
        diffusion=np.eye(2),
    )
    state = State(time=0.0, values=values)

    with pytest.raises(ValueError, match="state values must have shape"):
        linear_step(
            state=state,
            next_time=0.5,
            model=model,
            rng=np.random.default_rng(42),
        )


# The same seed must reproduce the same sampled state.
def test_linear_step_is_reproducible():
    model = LinearSDE(
        drift=np.array([[-0.7, 0.2], [0.0, -0.4]]),
        offset=np.array([0.3, -0.2]),
        diffusion=np.array([[1.0, 0.0], [0.4, 0.8]]),
    )
    state = State(
        time=0.0,
        values=np.array([[1.0], [-1.0]]),
    )

    first = linear_step(
        state=state,
        next_time=0.3,
        model=model,
        rng=np.random.default_rng(42),
    )
    second = linear_step(
        state=state,
        next_time=0.3,
        model=model,
        rng=np.random.default_rng(42),
    )

    np.testing.assert_array_equal(first.values, second.values)


# A rank-one covariance must preserve its exact noise constraint.
def test_linear_step_supports_shared_noise():
    model = LinearSDE(
        drift=np.zeros((2, 2)),
        offset=np.zeros(2),
        diffusion=np.array([[1.0], [2.0]]),
    )
    state = State(
        time=0.0,
        values=np.zeros((2, 1)),
    )
    rng = np.random.default_rng(42)

    for _ in range(20):
        result = linear_step(
            state=state,
            next_time=0.5,
            model=model,
            rng=rng,
        )

        np.testing.assert_allclose(
            result.values[1, 0],
            2.0 * result.values[0, 0],
            rtol=0.0,
            atol=1e-12,
        )


# Repeated independent steps should match the analytical mean and covariance.
def test_linear_step_sample_mean_and_covariance():
    model = LinearSDE(
        drift=np.zeros((2, 2)),
        offset=np.array([0.4, -0.2]),
        diffusion=np.array([[1.0, 0.0], [0.6, 0.8]]),
    )
    state = State(
        time=0.0,
        values=np.array([[1.0], [-2.0]]),
    )

    dt = 0.5
    n_samples = 6000
    rng = np.random.default_rng(2026)
    samples = np.empty((n_samples, 2), dtype=np.float64)

    for i in range(n_samples):
        result = linear_step(
            state=state,
            next_time=dt,
            model=model,
            rng=rng,
        )
        samples[i] = result.values[:, 0]

    expected_mean = np.array([1.2, -2.1])
    expected_covariance = np.array([
        [0.5, 0.3],
        [0.3, 0.5],
    ])

    sample_mean = np.mean(samples, axis=0)
    sample_covariance = np.cov(samples, rowvar=False, ddof=1)

    mean_standard_error = np.sqrt(
        np.diag(expected_covariance) / n_samples
    )
    np.testing.assert_array_less(
        np.abs(sample_mean - expected_mean),
        6.0 * mean_standard_error,
    )

    variances = np.diag(expected_covariance)
    covariance_standard_error = np.sqrt(
        (
            expected_covariance**2
            + np.outer(variances, variances)
        )
        / (n_samples - 1)
    )
    np.testing.assert_array_less(
        np.abs(sample_covariance - expected_covariance),
        6.0 * covariance_standard_error,
    )