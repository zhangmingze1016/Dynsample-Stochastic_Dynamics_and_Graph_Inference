import numpy as np
import pytest

from dynsample.core.observation import Observation


def test_observation_shape() -> None:
    values = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])

    mask = np.array([
        [True, False],
        [False, True],
        [True, True],
    ])

    observation = Observation(
        time=1.0,
        values=values,
        mask=mask,
    )

    assert observation.values.shape == (3, 2)
    assert observation.mask.shape == (3, 2)
    assert observation.n_nodes == 3
    assert observation.n_features == 2


def test_observation_counts_observed_entries() -> None:
    values = np.zeros((3, 2))

    mask = np.array([
        [True, False],
        [False, True],
        [True, True],
    ])

    observation = Observation(
        time=1.0,
        values=values,
        mask=mask,
    )

    assert observation.n_observed == 4


def test_observation_converts_dtypes() -> None:
    observation = Observation(
        time=1.0,
        values=[
            [1, 2],
            [3, 4],
        ],
        mask=[
            [1, 0],
            [0, 1],
        ],
    )

    assert observation.values.dtype == np.float64
    assert observation.mask.dtype == np.bool_


def test_observation_requires_matching_shapes() -> None:
    values = np.zeros((3, 2))
    mask = np.ones((3, 3), dtype=bool)

    with pytest.raises(ValueError):
        Observation(
            time=1.0,
            values=values,
            mask=mask,
        )


def test_observation_requires_finite_time() -> None:
    values = np.zeros((3, 2))
    mask = np.ones((3, 2), dtype=bool)

    with pytest.raises(ValueError):
        Observation(
            time=np.nan,
            values=values,
            mask=mask,
        )