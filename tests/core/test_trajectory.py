import numpy as np
import pytest

from dynsample.core.state import State
from dynsample.core.trajectory import Trajectory

def test_trajectory_is_isolated_from_input_mutation() -> None:
    times = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    values = np.array([1.0, 2.0, 3.0], dtype=np.float64)[:, None, None]
    trajectory = Trajectory(times=times, values=values)

    times[1] = -1.0
    values[1, 0, 0] = np.nan

    np.testing.assert_array_equal(trajectory.times, [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(trajectory.values[:, 0, 0], [1.0, 2.0, 3.0])

def test_trajectory_rejects_overflowing_time_intervals() -> None:
    with np.errstate(over="raise", invalid="raise"):
        with pytest.raises(ValueError, match="time intervals must be finite"):
            Trajectory(
                times=np.array([-1e308, 1e308]),
                values=np.zeros((2, 1, 1)),
            )


@pytest.mark.parametrize("times", [[1e308], [-1e308, 0.0, 1e308]])
def test_trajectory_accepts_finite_adjacent_intervals(times) -> None:
    trajectory = Trajectory(
        times=np.array(times), values=np.zeros((len(times), 1, 1))
    )
    np.testing.assert_array_equal(trajectory.times, times)


def test_trajectory_shape() -> None:
    times = np.array([
        0.0,
        0.1,
        0.2,
        0.3,
    ])

    values = np.zeros((4,5,3))

    trajectory = Trajectory(
        times = times,
        values = values,
    )

    assert trajectory.n_steps == 4
    assert trajectory.n_nodes == 5
    assert trajectory.n_features == 3
    assert len(trajectory) == 4

def test_trajectory_state_at() -> None:
    times = np.array([
        0.0,
        0.5,
        1.0,
    ])

    values = np.array([
        [[1.0], [2.0]],
        [[1.5], [2.5]],
        [[2.0], [3.0]]
    ])

    trajectory = Trajectory(
        times = times,
        values = values,
    )



    state = trajectory.state_at(1)

    assert isinstance(state, State)
    assert state.time == 0.5

    np.testing.assert_array_equal(
        state.values,
        np.array([
            [1.5],
            [2.5],
        ])
    )

def test_trajectory_state_at_returns_independent_snapshots() -> None:
    trajectory = Trajectory(
        times=np.array([0.0, 1.0]),
        values=np.array([1.0, 2.0])[:, None, None],
    )
    first = trajectory.state_at(0)
    second = trajectory.state_at(0)
    first.values[0, 0] = np.nan
    assert trajectory.values[0, 0, 0] == 1.0
    assert second.values[0, 0] == 1.0

    trajectory.values[0, 0, 0] = 3.0
    assert second.values[0, 0] == 1.0


def test_trajectory_requires_matching_time_steps() -> None:
    times = np.array([
        0.0,
        0.1,
        0.2
    ])

    values = np.zeros((2,5,2))

    with pytest.raises(ValueError):
        Trajectory(
            times = times,
            values = values,
        )

def test_trajectory_requires_increasing_times() -> None:
    times = np.array([
        0.0,
        0.2,
        0.1
    ])

    values = np.zeros((3, 5, 2))

    with pytest.raises(ValueError):
        Trajectory(
            times = times,
            values = values,
        )

def test_trajectory_requires_three_dimensional_values() -> None:
    times = np.array([
        0.0,
        0.1,
        0.2,
    ])

    values = np.zeros((2,5,2))

    with pytest.raises(ValueError):
        Trajectory(
            times = times,
            values = values,
        )
