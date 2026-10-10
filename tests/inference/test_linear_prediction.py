"""Check linear prediction against independent analytical references."""

import numpy as np
import pytest

from dynsample.core.linear_model import LinearSDE
from dynsample.core.state import State
from dynsample.core.trajectory import Trajectory
from dynsample.inference.prediction import predict_linear


# Scalar linear prediction must match the analytical OU moments.
def test_predict_linear_matches_scalar_ou():
    alpha = 0.7
    mu = 2.0
    sigma = 0.6
    x0 = 3.0

    initial_state = State(
        time=2.0,
        values=np.array([[x0]]),
    )
    times = np.array([2.0, 2.1, 2.7, 4.0])

    model = LinearSDE(
        drift=np.array([[-alpha]]),
        offset=np.array([alpha * mu]),
        diffusion=np.array([[sigma]]),
    )

    means, covariances = predict_linear(
        initial_state=initial_state,
        times=times,
        model=model,
    )

    horizons = times - initial_state.time

    expected_means = (
        mu + np.exp(-alpha * horizons) * (x0 - mu)
    )
    expected_variances = (
        sigma**2
        * -np.expm1(-2.0 * alpha * horizons)
        / (2.0 * alpha)
    )

    assert isinstance(means, Trajectory)
    assert means.values.shape == (4, 1, 1)
    assert covariances.shape == (4, 1, 1)

    np.testing.assert_array_equal(means.times, times)
    np.testing.assert_allclose(
        means.values[:, 0, 0],
        expected_means,
        rtol=1e-11,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        covariances[:, 0, 0],
        expected_variances,
        rtol=1e-11,
        atol=1e-12,
    )


# With K=0, the mean is x0+b*dt and covariance is B*B.T*dt.
def test_predict_linear_matches_zero_drift_reference():
    initial_values = np.array([1.0, -2.0])
    offset = np.array([0.3, -0.4])

    initial_state = State(
        time=5.0,
        values=initial_values[:, None],
    )

    # All query times are later than the initial time.
    times = np.array([5.2, 5.9, 7.0])

    model = LinearSDE(
        drift=np.zeros((2, 2)),
        offset=offset,
        diffusion=np.array([
            [1.0, 0.0],
            [0.6, 0.8],
        ]),
    )

    means, covariances = predict_linear(
        initial_state=initial_state,
        times=times,
        model=model,
    )

    horizons = times - initial_state.time
    noise_rate = np.array([
        [1.0, 0.6],
        [0.6, 1.0],
    ])

    expected_means = (
        initial_values[None, :]
        + horizons[:, None] * offset[None, :]
    )
    expected_covariances = (
        horizons[:, None, None] * noise_rate[None, :, :]
    )

    assert means.values.shape == (3, 2, 1)
    assert covariances.shape == (3, 2, 2)

    np.testing.assert_allclose(
        means.values[:, :, 0],
        expected_means,
        rtol=1e-11,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        covariances,
        expected_covariances,
        rtol=1e-11,
        atol=1e-12,
    )


# Coupling must propagate both state means and process uncertainty.
def test_predict_linear_matches_coupled_reference():
    initial_state = State(
        time=0.0,
        values=np.array([
            [1.0],
            [2.0],
        ]),
    )
    times = np.array([0.2, 0.5, 1.0])

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

    means, covariances = predict_linear(
        initial_state=initial_state,
        times=times,
        model=model,
    )

    # dX2 = 2 dt + dW; dX1 = X2 dt.
    for i, horizon in enumerate(times):
        expected_mean = np.array([
            1.0 + 2.0 * horizon + horizon**2,
            2.0 + 2.0 * horizon,
        ])
        expected_covariance = np.array([
            [horizon**3 / 3.0, horizon**2 / 2.0],
            [horizon**2 / 2.0, horizon],
        ])

        np.testing.assert_allclose(
            means.values[i, :, 0],
            expected_mean,
            rtol=1e-11,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            covariances[i],
            expected_covariance,
            rtol=1e-11,
            atol=1e-12,
        )


# At zero horizon, the exact initial state has zero uncertainty.
def test_predict_linear_preserves_initial_state():
    initial_state = State(
        time=3.0,
        values=np.array([
            [1.5],
            [-0.5],
        ]),
    )
    model = LinearSDE(
        drift=-np.eye(2),
        offset=np.array([0.2, 0.3]),
        diffusion=np.eye(2),
    )

    means, covariances = predict_linear(
        initial_state=initial_state,
        times=np.array([3.0]),
        model=model,
    )

    np.testing.assert_array_equal(
        means.values[0],
        initial_state.values,
    )
    np.testing.assert_array_equal(
        covariances,
        np.zeros((1, 2, 2)),
    )


# Prediction may have singular covariance; no Gaussian density is evaluated.
def test_predict_linear_accepts_singular_covariance():
    initial_state = State(
        time=0.0,
        values=np.zeros((2, 1)),
    )
    model = LinearSDE(
        drift=np.zeros((2, 2)),
        offset=np.zeros(2),
        diffusion=np.array([
            [1.0],
            [0.0],
        ]),
    )

    means, covariances = predict_linear(
        initial_state=initial_state,
        times=np.array([0.5]),
        model=model,
    )

    np.testing.assert_array_equal(
        means.values,
        np.zeros((1, 2, 1)),
    )
    np.testing.assert_allclose(
        covariances[0],
        np.array([
            [0.5, 0.0],
            [0.0, 0.0],
        ]),
        atol=1e-12,
    )


# Adding query times must not change a prediction at an existing horizon.
def test_predict_linear_is_independent_of_query_grid():
    initial_state = State(
        time=0.0,
        values=np.array([
            [1.0],
            [-1.0],
        ]),
    )
    model = LinearSDE(
        drift=np.array([
            [-0.7, 0.0],
            [0.5, -1.0],
        ]),
        offset=np.array([0.4, -0.2]),
        diffusion=np.diag([0.6, 0.5]),
    )

    single_mean, single_covariance = predict_linear(
        initial_state=initial_state,
        times=np.array([2.0]),
        model=model,
    )
    grid_means, grid_covariances = predict_linear(
        initial_state=initial_state,
        times=np.array([0.0, 0.2, 0.9, 2.0]),
        model=model,
    )

    np.testing.assert_allclose(
        grid_means.values[-1],
        single_mean.values[0],
    )
    np.testing.assert_allclose(
        grid_covariances[-1],
        single_covariance[0],
    )


# Returned times must not share storage with the caller's query array.
def test_predict_linear_copies_query_times():
    times = np.array([0.0, 1.0])
    initial_state = State(
        time=0.0,
        values=np.array([[1.0]]),
    )
    model = LinearSDE(
        drift=np.array([[-1.0]]),
        offset=np.array([0.0]),
        diffusion=np.array([[0.5]]),
    )

    means, _ = predict_linear(
        initial_state=initial_state,
        times=times,
        model=model,
    )

    np.testing.assert_array_equal(times, [0.0, 1.0])
    assert not np.shares_memory(means.times, times)

    times[1] = 9.0
    np.testing.assert_array_equal(means.times, [0.0, 1.0])


# Reject invalid query-time arrays before computing transitions.
@pytest.mark.parametrize(
    "times, message",
    [
        (np.array([]), "at least one time"),
        (np.array([[0.0, 1.0]]), "one-dimensional"),
        (np.array([0.0, np.nan]), "finite values"),
        (np.array([0.0, np.inf]), "finite values"),
        (np.array([0.0, 0.0]), "strictly increasing"),
        (np.array([1.0, 0.5]), "strictly increasing"),
        (np.array([-0.1, 0.5]), "must not precede"),
    ],
)
def test_predict_linear_rejects_invalid_times(times, message):
    initial_state = State(
        time=0.0,
        values=np.array([[1.0]]),
    )
    model = LinearSDE(
        drift=np.array([[-1.0]]),
        offset=np.array([0.0]),
        diffusion=np.array([[0.5]]),
    )

    with pytest.raises(ValueError, match=message):
        predict_linear(
            initial_state=initial_state,
            times=times,
            model=model,
        )


# Node counts and feature counts must match the current model contract.
@pytest.mark.parametrize(
    "shape",
    [
        (3, 1),
        (2, 2),
    ],
)
def test_predict_linear_rejects_incompatible_state(shape):
    initial_state = State(
        time=0.0,
        values=np.zeros(shape),
    )
    model = LinearSDE(
        drift=-np.eye(2),
        offset=np.zeros(2),
        diffusion=np.eye(2),
    )

    with pytest.raises(
        ValueError,
        match="matching nodes and one feature",
    ):
        predict_linear(
            initial_state=initial_state,
            times=np.array([0.0, 1.0]),
            model=model,
        )


# Finite timestamps can still produce an overflowing elapsed duration.
def test_predict_linear_rejects_overflowing_horizon():
    initial_state = State(
        time=-1e308,
        values=np.array([[0.0]]),
    )
    model = LinearSDE(
        drift=np.array([[0.0]]),
        offset=np.array([0.0]),
        diffusion=np.array([[1.0]]),
    )

    with pytest.raises(ValueError, match="horizons must be finite"):
        predict_linear(
            initial_state=initial_state,
            times=np.array([1e308]),
            model=model,
        )
