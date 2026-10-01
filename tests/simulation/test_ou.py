import numpy as np
import pytest
from dynsample.core.state import State
from dynsample.simulation.ou import ou_step
from dynsample.simulation.brownian import brownian_step
from dynsample.core.trajectory import Trajectory
from dynsample.simulation.ou import simulate_ou
import matplotlib.pyplot as plt

def test_ou_step_returns_new_state() -> None:
    values = np.array(
        [
            [14.0, 8.0],
            [0.0, -2.0],
        ]
    )

    state = State(
        time = 0.0,
        values = values
    )

    original_values = state.values.copy()

    result = ou_step(
        state = state,
        next_time = 0.3,
        mean_reversion = 0.7,
        long_run_mean = 10.0,
        volatility = 1.2,
        rng = np.random.default_rng(42),
    )

    assert isinstance(result, State)
    assert result.time == 0.3
    assert result.values.shape == (2,2)
    np.testing.assert_array_almost_equal(state.values, original_values)
    assert not np.shares_memory(result.values, state.values)

def test_ou_step_zero_volatility() -> None:
    state = State(
        time = 0.0,
        values =np.array(
            [
                [14.0],
                [6.0],
            ]
        ),
        
    )

    result = ou_step(
        state = state,
        next_time = 1.0,
        mean_reversion = np.log(2.0),
        long_run_mean = 10.0,
        volatility = 0.0,
        rng = np.random.default_rng(42),
    )

    expected = np.array(
        [
            [12.0],
            [8.0]
        ]
    )

    expected = np.array([[12.0], [8.0]])
    np.testing.assert_allclose(result.values, expected)


def test_ou_step_is_reproducible() -> None:
    state = State(time = 0.0, values = np.zeros((2,3)))
    arguments = dict(
        state = state,
        next_time = 0.5,
        mean_reversion = 0.7,
        long_run_mean = 2.0,
        volatility = 1.2,
    )

    first = ou_step(**arguments, rng = np.random.default_rng(42))
    second = ou_step(**arguments, rng = np.random.default_rng(42))

    np.testing.assert_array_equal(first.values, second.values)

def test_ou_step_matches_brownian_when_reversion_is_zero() -> None:
    state = State(
        time = 0.0,
        values = np.array([[1.0], [2.0]])
    )

    ou_result  = ou_step(
        state = state,
        next_time = 0.5,
        mean_reversion = 0.0,
        long_run_mean = 10.0,
        volatility = 1.2,
        rng = np.random.default_rng(42),
    )

    brownian_result = brownian_step(
        state = state,
        next_time = 0.5,
        volatility = 1.2,
        rng = np.random.default_rng(42),
    )

    np.testing.assert_allclose(ou_result.values, brownian_result.values)

@pytest.mark.parametrize("next_time", [0.0, -1.0, np.nan, np.inf])
def test_ou_step_rejects_invalid_next_time(next_time: float) -> None:
    state = State(time=0.0, values=np.zeros((1, 1)))

    with pytest.raises(ValueError):
        ou_step(
            state=state,
            next_time=next_time,
            mean_reversion=0.7,
            long_run_mean=0.0,
            volatility=1.0,
            rng=np.random.default_rng(42),
        )

def test_simulate_ou_zero_volatility() -> None:
    initial_state = State(
        time=0.0,
        values=np.array([[14.0], [6.0]]),
    )
    times = np.array([0.0, 1.0, 3.0])

    result = simulate_ou(
        initial_state=initial_state,
        times=times,
        mean_reversion=np.log(2.0),
        long_run_mean=10.0,
        volatility=0.0,
        rng=np.random.default_rng(42),
    )

    expected = np.array([
        [[14.0], [6.0]],
        [[12.0], [8.0]],
        [[10.5], [9.5]],
    ])

    assert isinstance(result, Trajectory)
    np.testing.assert_array_equal(result.times, times)
    np.testing.assert_allclose(result.values, expected)

def test_simulate_ou_single_time() -> None:
    initial_state = State(
        time=2.0,
        values=np.array([[14.0], [6.0]]),
    )

    result = simulate_ou(
        initial_state=initial_state,
        times=np.array([2.0]),
        mean_reversion=0.7,
        long_run_mean=10.0,
        volatility=1.2,
        rng=np.random.default_rng(42),
    )

    assert result.values.shape == (1, 2, 1)
    np.testing.assert_array_equal(result.values[0], initial_state.values)
    np.testing.assert_array_equal(result.times, np.array([2.0]))

