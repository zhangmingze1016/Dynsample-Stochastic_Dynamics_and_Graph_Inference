import numpy as np
import pytest

from dynsample.core.state import State
from dynsample.core.trajectory import Trajectory
from dynsample.simulation.brownian import (
    brownian_step,
    simulate_brownian,
)

def test_simulate_brownian_rejects_shifeted_large_timestamp() -> None:
    initial_state = State(
        time = 1_000_000_000.0,
        values = np.zeros((1, 1)),
    )
    times = np.array([
        1_000_000_001.0,
        1_000_000_002.0,
    ])

    with pytest.raises(
        ValueError,
        match =  "the first time must equal initial_state.time"
    ):
        simulate_brownian(
            initial_state = initial_state,
            times = times,
            volatility = 1.0,
            rng = np.random.default_rng(42),
        )


def test_brownian_step_returns_state() -> None:
    """A Brownian transition shall return a State of the proper form."""

    initial_state = State(
        time=0.0,
        values=np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ]),
    )

    rng = np.random.default_rng(42)

    result = brownian_step(
        state=initial_state,
        next_time=0.5,
        volatility=0.2,
        rng=rng,
    )

    assert isinstance(result, State)
    assert result.time == 0.5
    assert result.values.shape == initial_state.values.shape


def test_brownian_step_zero_volatility() -> None:
    """
    With no volatility, the state shall suffer no stochastic change.
    """

    initial_state = State(
        time=0.0,
        values=np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ]),
    )

    rng = np.random.default_rng(42)

    result = brownian_step(
        state=initial_state,
        next_time=1.0,
        volatility=0.0,
        rng=rng,
    )

    np.testing.assert_array_equal(
        result.values,
        initial_state.values,
    )

    assert result.time == 1.0


def test_brownian_step_rejects_same_time() -> None:
    """A transition cannot proceed unto the time already occupied."""

    initial_state = State(
        time=1.0,
        values=np.array([
            [1.0],
            [2.0],
        ]),
    )

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_step(
            state=initial_state,
            next_time=1.0,
            volatility=0.2,
            rng=rng,
        )


def test_brownian_step_rejects_earlier_time() -> None:
    """A Brownian transition shall not proceed backward in time."""

    initial_state = State(
        time=1.0,
        values=np.array([
            [1.0],
            [2.0],
        ]),
    )

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_step(
            state=initial_state,
            next_time=0.5,
            volatility=0.2,
            rng=rng,
        )


def test_brownian_step_rejects_negative_volatility() -> None:
    """Volatility cannot take a negative value."""

    initial_state = State(
        time=0.0,
        values=np.array([
            [1.0],
            [2.0],
        ]),
    )

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_step(
            state=initial_state,
            next_time=1.0,
            volatility=-0.2,
            rng=rng,
        )


def test_brownian_step_is_reproducible() -> None:
    """
    Equal seeds shall produce equal Brownian increments.
    """

    initial_state = State(
        time=0.0,
        values=np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ]),
    )

    rng_1 = np.random.default_rng(42)
    rng_2 = np.random.default_rng(42)

    result_1 = brownian_step(
        state=initial_state,
        next_time=1.0,
        volatility=0.5,
        rng=rng_1,
    )

    result_2 = brownian_step(
        state=initial_state,
        next_time=1.0,
        volatility=0.5,
        rng=rng_2,
    )

    np.testing.assert_array_equal(
        result_1.values,
        result_2.values,
    )


def test_simulate_brownian_returns_trajectory() -> None:
    """A complete Brownian simulation shall return a Trajectory."""

    initial_state = State(
        time=0.0,
        values=np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]),
    )

    times = np.array([
        0.0,
        0.1,
        0.7,
        2.3,
    ])

    rng = np.random.default_rng(42)

    result = simulate_brownian(
        initial_state=initial_state,
        times=times,
        volatility=0.2,
        rng=rng,
    )

    assert isinstance(result, Trajectory)

    assert result.values.shape == (
        4,
        3,
        2,
    )


def test_simulate_brownian_preserves_times() -> None:
    """
    The simulator shall preserve the irregular times appointed
    by the caller.
    """

    initial_state = State(
        time=0.0,
        values=np.array([
            [1.0],
            [2.0],
        ]),
    )

    times = np.array([
        0.0,
        0.13,
        0.47,
        2.1,
    ])

    rng = np.random.default_rng(42)

    result = simulate_brownian(
        initial_state=initial_state,
        times=times,
        volatility=0.2,
        rng=rng,
    )

    np.testing.assert_array_equal(
        result.times,
        times,
    )


def test_simulate_brownian_preserves_initial_state() -> None:
    """The first state shall remain precisely the state supplied."""

    initial_state = State(
        time=0.0,
        values=np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ]),
    )

    times = np.array([
        0.0,
        0.5,
        1.0,
    ])

    rng = np.random.default_rng(42)

    result = simulate_brownian(
        initial_state=initial_state,
        times=times,
        volatility=0.3,
        rng=rng,
    )

    np.testing.assert_array_equal(
        result.values[0],
        initial_state.values,
    )


def test_simulate_brownian_rejects_non_increasing_times() -> None:
    """Simulation times shall proceed in strictly increasing order."""

    initial_state = State(
        time=0.0,
        values=np.array([
            [1.0],
        ]),
    )

    times = np.array([
        0.0,
        0.5,
        0.5,
        1.0,
    ])

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        simulate_brownian(
            initial_state=initial_state,
            times=times,
            volatility=0.2,
            rng=rng,
        )


def test_brownian_increment_mean_and_variance() -> None:
    """
    Brownian increments shall possess approximately the theoretical
    mean and variance prescribed by the model.
    """

    n_samples = 100_000

    initial_state = State(
        time=0.0,
        values=np.zeros((n_samples, 1)),
    )

    next_time = 2.0
    volatility = 0.5

    rng = np.random.default_rng(42)

    result = brownian_step(
        state=initial_state,
        next_time=next_time,
        volatility=volatility,
        rng=rng,
    )

    increments = (
        result.values
        - initial_state.values
    ).ravel()

    empirical_mean = np.mean(increments)
    empirical_variance = np.var(increments)

    dt = next_time - initial_state.time

    theoretical_mean = 0.0
    theoretical_variance = volatility**2 * dt

    assert np.isclose(
        empirical_mean,
        theoretical_mean,
        atol=0.01,
    )

    assert np.isclose(
        empirical_variance,
        theoretical_variance,
        rtol=0.03,
    )