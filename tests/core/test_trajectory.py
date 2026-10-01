import numpy as np
import pytest

from dynsample.core.state import State
from dynsample.core.trajectory import Trajectory

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