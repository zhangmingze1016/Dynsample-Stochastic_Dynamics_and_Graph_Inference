import numpy as np
import pytest

from dynsample.core.state import State

def test_state_shape() -> None:
    values = np.zeros((5,3))

    state = State(
        time = 0.0,
        values =values,
    )

    assert state.values.shape == (5,3)
    assert state.n_nodes == 5
    assert state.n_features == 3

def test_state_converts_input_to_float64() -> None:
    state = State(
        time = 0.0,
        values= [
            [1, 2],
            [3, 4],
        ],
    )

    assert isinstance(state.values, np.ndarray)
    assert state.values.dtype == np.float64

def test_state_requires_two_dimensions() -> None:
    values = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        State(
            time = 0.0,
            values = values,
        )

def test_state_requires_finite_time() -> None:
    values = np.zeros((3,2))

    with pytest.raises(ValueError):
        State(
            time = np.nan,
            values = values,
        )