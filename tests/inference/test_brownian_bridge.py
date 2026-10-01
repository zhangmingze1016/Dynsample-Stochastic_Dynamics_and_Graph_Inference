import numpy as np
import pytest
from dynsample.core.trajectory import Trajectory
from dynsample.core.state import State
from dynsample.inference.reconstruction.brownian_bridge import (
    brownian_bridge_step,
    brownian_bridge,
)


def test_brownian_bridge_step_returns_state() -> None:
    left_state = State(
        time=0.0,
        values=np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ),
    )

    right_state = State(
        time=10.0,
        values=np.array(
            [
                [5.0, 6.0],
                [7.0, 8.0],
            ]
        ),
    )

    rng = np.random.default_rng(42)

    result = brownian_bridge_step(
        left_state=left_state,
        right_state=right_state,
        time=4.0,
        volatility=1.0,
        rng=rng,
    )

    assert isinstance(result, State)
    assert result.time == 4.0
    assert result.values.shape == left_state.values.shape


def test_brownian_bridge_step_zero_volatility() -> None:
    left_state = State(
        time=0.0,
        values=np.array(
            [
                [10.0, 20.0],
                [30.0, 40.0],
            ]
        ),
    )

    right_state = State(
        time=10.0,
        values=np.array(
            [
                [30.0, 40.0],
                [50.0, 60.0],
            ]
        ),
    )

    rng = np.random.default_rng(42)

    result = brownian_bridge_step(
        left_state=left_state,
        right_state=right_state,
        time=2.0,
        volatility=0.0,
        rng=rng,
    )

    alpha = 2.0 / 10.0

    expected = (
        left_state.values
        + alpha
        * (right_state.values - left_state.values)
    )

    assert np.allclose(result.values, expected)


def test_brownian_bridge_step_reproducible() -> None:
    left_state = State(
        time=0.0,
        values=np.array([[1.0, 2.0]]),
    )

    right_state = State(
        time=10.0,
        values=np.array([[5.0, 8.0]]),
    )

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    result1 = brownian_bridge_step(
        left_state=left_state,
        right_state=right_state,
        time=4.0,
        volatility=1.5,
        rng=rng1,
    )

    result2 = brownian_bridge_step(
        left_state=left_state,
        right_state=right_state,
        time=4.0,
        volatility=1.5,
        rng=rng2,
    )

    assert np.allclose(result1.values, result2.values)


def test_brownian_bridge_step_rejects_left_endpoint() -> None:
    left_state = State(
        time=0.0,
        values=np.array([[1.0]]),
    )

    right_state = State(
        time=10.0,
        values=np.array([[2.0]]),
    )

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_bridge_step(
            left_state=left_state,
            right_state=right_state,
            time=0.0,
            volatility=1.0,
            rng=rng,
        )


def test_brownian_bridge_step_rejects_right_endpoint() -> None:
    left_state = State(
        time=0.0,
        values=np.array([[1.0]]),
    )

    right_state = State(
        time=10.0,
        values=np.array([[2.0]]),
    )

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_bridge_step(
            left_state=left_state,
            right_state=right_state,
            time=10.0,
            volatility=1.0,
            rng=rng,
        )


def test_brownian_bridge_step_rejects_time_before_interval() -> None:
    left_state = State(
        time=0.0,
        values=np.array([[1.0]]),
    )

    right_state = State(
        time=10.0,
        values=np.array([[2.0]]),
    )

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_bridge_step(
            left_state=left_state,
            right_state=right_state,
            time=-1.0,
            volatility=1.0,
            rng=rng,
        )


def test_brownian_bridge_step_rejects_time_after_interval() -> None:
    left_state = State(
        time=0.0,
        values=np.array([[1.0]]),
    )

    right_state = State(
        time=10.0,
        values=np.array([[2.0]]),
    )

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_bridge_step(
            left_state=left_state,
            right_state=right_state,
            time=11.0,
            volatility=1.0,
            rng=rng,
        )


def test_brownian_bridge_step_rejects_invalid_state_order() -> None:
    left_state = State(
        time=10.0,
        values=np.array([[1.0]]),
    )

    right_state = State(
        time=5.0,
        values=np.array([[2.0]]),
    )

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_bridge_step(
            left_state=left_state,
            right_state=right_state,
            time=7.0,
            volatility=1.0,
            rng=rng,
        )


def test_brownian_bridge_step_rejects_shape_mismatch() -> None:
    left_state = State(
        time=0.0,
        values=np.ones((2, 3)),
    )

    right_state = State(
        time=10.0,
        values=np.ones((3, 3)),
    )

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_bridge_step(
            left_state=left_state,
            right_state=right_state,
            time=5.0,
            volatility=1.0,
            rng=rng,
        )


def test_brownian_bridge_step_rejects_negative_volatility() -> None:
    left_state = State(
        time=0.0,
        values=np.array([[1.0]]),
    )

    right_state = State(
        time=10.0,
        values=np.array([[2.0]]),
    )

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_bridge_step(
            left_state=left_state,
            right_state=right_state,
            time=5.0,
            volatility=-0.5,
            rng=rng,
        )


@pytest.mark.parametrize(
    "time",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_brownian_bridge_step_rejects_nonfinite_time(
    time: float,
) -> None:
    left_state = State(
        time=0.0,
        values=np.array([[1.0]]),
    )

    right_state = State(
        time=10.0,
        values=np.array([[2.0]]),
    )

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_bridge_step(
            left_state=left_state,
            right_state=right_state,
            time=time,
            volatility=1.0,
            rng=rng,
        )


@pytest.mark.parametrize(
    "volatility",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_brownian_bridge_step_rejects_nonfinite_volatility(
    volatility: float,
) -> None:
    left_state = State(
        time=0.0,
        values=np.array([[1.0]]),
    )

    right_state = State(
        time=10.0,
        values=np.array([[2.0]]),
    )

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_bridge_step(
            left_state=left_state,
            right_state=right_state,
            time=5.0,
            volatility=volatility,
            rng=rng,
        )


def test_brownian_bridge_step_empirical_mean_and_variance() -> None:
    left_state = State(
        time=0.0,
        values=np.array([[10.0]]),
    )

    right_state = State(
        time=10.0,
        values=np.array([[30.0]]),
    )

    time = 2.0
    volatility = 2.0
    n_samples = 100_000

    rng = np.random.default_rng(42)

    samples = np.empty(n_samples)

    for i in range(n_samples):
        result = brownian_bridge_step(
            left_state=left_state,
            right_state=right_state,
            time=time,
            volatility=volatility,
            rng=rng,
        )

        samples[i] = result.values[0, 0]

    alpha = (
        (time - left_state.time)
        / (right_state.time - left_state.time)
    )

    expected_mean = (
        left_state.values[0, 0]
        + alpha
        * (
            right_state.values[0, 0]
            - left_state.values[0, 0]
        )
    )

    expected_variance = (
        volatility**2
        * (time - left_state.time)
        * (right_state.time - time)
        / (right_state.time - left_state.time)
    )

    assert np.isclose(
        samples.mean(),
        expected_mean,
        atol=0.03,
    )

    assert np.isclose(
        samples.var(),
        expected_variance,
        atol=0.08,
    )

# ---------------------------------------------------------------------------
# Multi-point Brownian bridge
# ---------------------------------------------------------------------------


def test_brownian_bridge_returns_trajectory() -> None:
    left_state = State(
        time=0.0,
        values=np.array([
            [0.0, 1.0],
            [2.0, 3.0],
        ]),
    )

    right_state = State(
        time=10.0,
        values=np.array([
            [10.0, 11.0],
            [12.0, 13.0],
        ]),
    )

    times = np.array([
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
    ])

    rng = np.random.default_rng(42)

    result = brownian_bridge(
        left_state=left_state,
        right_state=right_state,
        times=times,
        volatility=1.0,
        rng=rng,
    )

    assert isinstance(result, Trajectory)
    assert np.array_equal(result.times, times)
    assert result.values.shape == (9, 2, 2)


def test_brownian_bridge_single_time() -> None:
    left_state = State(
        time=0.0,
        values=np.array([
            [1.0],
        ]),
    )

    right_state = State(
        time=10.0,
        values=np.array([
            [5.0],
        ]),
    )

    times = np.array([
        4.0,
    ])

    rng = np.random.default_rng(42)

    result = brownian_bridge(
        left_state=left_state,
        right_state=right_state,
        times=times,
        volatility=1.0,
        rng=rng,
    )

    assert isinstance(result, Trajectory)
    assert result.times.shape == (1,)
    assert result.values.shape == (1, 1, 1)
    assert result.times[0] == 4.0


def test_brownian_bridge_zero_volatility() -> None:
    left_state = State(
        time=0.0,
        values=np.array([
            [0.0],
        ]),
    )

    right_state = State(
        time=10.0,
        values=np.array([
            [10.0],
        ]),
    )

    times = np.array([
        0.5,
        1.7,
        4.0,
        7.3,
        9.8,
    ])

    rng = np.random.default_rng(42)

    result = brownian_bridge(
        left_state=left_state,
        right_state=right_state,
        times=times,
        volatility=0.0,
        rng=rng,
    )

    expected = times.reshape(-1, 1, 1)

    assert np.allclose(
        result.values,
        expected,
    )


def test_brownian_bridge_zero_volatility_general_endpoints() -> None:
    left_state = State(
        time=2.0,
        values=np.array([
            [10.0, 20.0],
        ]),
    )

    right_state = State(
        time=12.0,
        values=np.array([
            [30.0, 40.0],
        ]),
    )

    times = np.array([
        3.0,
        5.0,
        8.0,
        11.0,
    ])

    rng = np.random.default_rng(42)

    result = brownian_bridge(
        left_state=left_state,
        right_state=right_state,
        times=times,
        volatility=0.0,
        rng=rng,
    )

    alpha = (
        (times - left_state.time)
        / (right_state.time - left_state.time)
    )

    expected = (
        left_state.values[None, :, :]
        + alpha[:, None, None]
        * (
            right_state.values
            - left_state.values
        )[None, :, :]
    )

    assert np.allclose(
        result.values,
        expected,
    )


def test_brownian_bridge_reproducible() -> None:
    left_state = State(
        time=0.0,
        values=np.array([
            [1.0],
            [2.0],
        ]),
    )

    right_state = State(
        time=10.0,
        values=np.array([
            [5.0],
            [8.0],
        ]),
    )

    times = np.array([
        0.5,
        1.7,
        4.2,
        6.1,
        8.8,
    ])

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    result1 = brownian_bridge(
        left_state=left_state,
        right_state=right_state,
        times=times,
        volatility=2.0,
        rng=rng1,
    )

    result2 = brownian_bridge(
        left_state=left_state,
        right_state=right_state,
        times=times,
        volatility=2.0,
        rng=rng2,
    )

    assert np.array_equal(
        result1.values,
        result2.values,
    )


def test_brownian_bridge_accepts_irregular_times() -> None:
    left_state = State(
        time=0.0,
        values=np.array([
            [0.0],
        ]),
    )

    right_state = State(
        time=10.0,
        values=np.array([
            [10.0],
        ]),
    )

    times = np.array([
        0.01,
        0.1,
        0.4,
        2.7,
        8.9,
        9.95,
    ])

    rng = np.random.default_rng(42)

    result = brownian_bridge(
        left_state=left_state,
        right_state=right_state,
        times=times,
        volatility=1.0,
        rng=rng,
    )

    assert np.array_equal(
        result.times,
        times,
    )

    assert result.values.shape == (
        len(times),
        1,
        1,
    )


def test_brownian_bridge_preserves_time_order() -> None:
    left_state = State(
        time=0.0,
        values=np.array([
            [0.0],
        ]),
    )

    right_state = State(
        time=8.0,
        values=np.array([
            [8.0],
        ]),
    )

    times = np.array([
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
    ])

    rng = np.random.default_rng(42)

    result = brownian_bridge(
        left_state=left_state,
        right_state=right_state,
        times=times,
        volatility=1.0,
        rng=rng,
    )

    # The bridge may be drawn in a different order internally,
    # yet the returned trajectory must retain chronological order.
    assert np.array_equal(
        result.times,
        times,
    )


def test_brownian_bridge_rejects_empty_times() -> None:
    left_state = State(
        time=0.0,
        values=np.array([
            [0.0],
        ]),
    )

    right_state = State(
        time=10.0,
        values=np.array([
            [10.0],
        ]),
    )

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_bridge(
            left_state=left_state,
            right_state=right_state,
            times=np.array([]),
            volatility=1.0,
            rng=rng,
        )


def test_brownian_bridge_rejects_two_dimensional_times() -> None:
    left_state = State(
        time=0.0,
        values=np.array([
            [0.0],
        ]),
    )

    right_state = State(
        time=10.0,
        values=np.array([
            [10.0],
        ]),
    )

    times = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_bridge(
            left_state=left_state,
            right_state=right_state,
            times=times,
            volatility=1.0,
            rng=rng,
        )


def test_brownian_bridge_rejects_unsorted_times() -> None:
    left_state = State(
        time=0.0,
        values=np.array([
            [0.0],
        ]),
    )

    right_state = State(
        time=10.0,
        values=np.array([
            [10.0],
        ]),
    )

    times = np.array([
        1.0,
        5.0,
        3.0,
    ])

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_bridge(
            left_state=left_state,
            right_state=right_state,
            times=times,
            volatility=1.0,
            rng=rng,
        )


def test_brownian_bridge_rejects_duplicate_times() -> None:
    left_state = State(
        time=0.0,
        values=np.array([
            [0.0],
        ]),
    )

    right_state = State(
        time=10.0,
        values=np.array([
            [10.0],
        ]),
    )

    times = np.array([
        1.0,
        3.0,
        3.0,
        7.0,
    ])

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_bridge(
            left_state=left_state,
            right_state=right_state,
            times=times,
            volatility=1.0,
            rng=rng,
        )


@pytest.mark.parametrize(
    "bad_time",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_brownian_bridge_rejects_nonfinite_times(
    bad_time: float,
) -> None:
    left_state = State(
        time=0.0,
        values=np.array([
            [0.0],
        ]),
    )

    right_state = State(
        time=10.0,
        values=np.array([
            [10.0],
        ]),
    )

    times = np.array([
        1.0,
        bad_time,
        7.0,
    ])

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_bridge(
            left_state=left_state,
            right_state=right_state,
            times=times,
            volatility=1.0,
            rng=rng,
        )


def test_brownian_bridge_rejects_left_endpoint() -> None:
    left_state = State(
        time=0.0,
        values=np.array([
            [0.0],
        ]),
    )

    right_state = State(
        time=10.0,
        values=np.array([
            [10.0],
        ]),
    )

    times = np.array([
        0.0,
        2.0,
        5.0,
    ])

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_bridge(
            left_state=left_state,
            right_state=right_state,
            times=times,
            volatility=1.0,
            rng=rng,
        )


def test_brownian_bridge_rejects_right_endpoint() -> None:
    left_state = State(
        time=0.0,
        values=np.array([
            [0.0],
        ]),
    )

    right_state = State(
        time=10.0,
        values=np.array([
            [10.0],
        ]),
    )

    times = np.array([
        2.0,
        5.0,
        10.0,
    ])

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_bridge(
            left_state=left_state,
            right_state=right_state,
            times=times,
            volatility=1.0,
            rng=rng,
        )


def test_brownian_bridge_rejects_time_before_left_boundary() -> None:
    left_state = State(
        time=0.0,
        values=np.array([
            [0.0],
        ]),
    )

    right_state = State(
        time=10.0,
        values=np.array([
            [10.0],
        ]),
    )

    times = np.array([
        -1.0,
        2.0,
        5.0,
    ])

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_bridge(
            left_state=left_state,
            right_state=right_state,
            times=times,
            volatility=1.0,
            rng=rng,
        )


def test_brownian_bridge_rejects_time_after_right_boundary() -> None:
    left_state = State(
        time=0.0,
        values=np.array([
            [0.0],
        ]),
    )

    right_state = State(
        time=10.0,
        values=np.array([
            [10.0],
        ]),
    )

    times = np.array([
        2.0,
        5.0,
        11.0,
    ])

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_bridge(
            left_state=left_state,
            right_state=right_state,
            times=times,
            volatility=1.0,
            rng=rng,
        )


def test_brownian_bridge_rejects_invalid_state_order() -> None:
    left_state = State(
        time=10.0,
        values=np.array([
            [0.0],
        ]),
    )

    right_state = State(
        time=0.0,
        values=np.array([
            [10.0],
        ]),
    )

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_bridge(
            left_state=left_state,
            right_state=right_state,
            times=np.array([
                2.0,
                5.0,
            ]),
            volatility=1.0,
            rng=rng,
        )


def test_brownian_bridge_rejects_shape_mismatch() -> None:
    left_state = State(
        time=0.0,
        values=np.array([
            [0.0, 1.0],
        ]),
    )

    right_state = State(
        time=10.0,
        values=np.array([
            [10.0],
        ]),
    )

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_bridge(
            left_state=left_state,
            right_state=right_state,
            times=np.array([
                2.0,
                5.0,
            ]),
            volatility=1.0,
            rng=rng,
        )


def test_brownian_bridge_rejects_negative_volatility() -> None:
    left_state = State(
        time=0.0,
        values=np.array([
            [0.0],
        ]),
    )

    right_state = State(
        time=10.0,
        values=np.array([
            [10.0],
        ]),
    )

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_bridge(
            left_state=left_state,
            right_state=right_state,
            times=np.array([
                2.0,
                5.0,
            ]),
            volatility=-1.0,
            rng=rng,
        )


@pytest.mark.parametrize(
    "bad_volatility",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_brownian_bridge_rejects_nonfinite_volatility(
    bad_volatility: float,
) -> None:
    left_state = State(
        time=0.0,
        values=np.array([
            [0.0],
        ]),
    )

    right_state = State(
        time=10.0,
        values=np.array([
            [10.0],
        ]),
    )

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_bridge(
            left_state=left_state,
            right_state=right_state,
            times=np.array([
                2.0,
                5.0,
            ]),
            volatility=bad_volatility,
            rng=rng,
        )


def test_brownian_bridge_empirical_mean_and_variance() -> None:
    left_state = State(
        time=0.0,
        values=np.array([
            [10.0],
        ]),
    )

    right_state = State(
        time=10.0,
        values=np.array([
            [30.0],
        ]),
    )

    times = np.array([
        2.0,
        5.0,
        8.0,
    ])

    volatility = 2.0
    n_samples = 20_000

    rng = np.random.default_rng(42)

    samples = np.empty(
        (
            n_samples,
            len(times),
        ),
        dtype=np.float64,
    )

    for i in range(n_samples):
        result = brownian_bridge(
            left_state=left_state,
            right_state=right_state,
            times=times,
            volatility=volatility,
            rng=rng,
        )

        samples[i] = result.values[:, 0, 0]

    alpha = (
        (times - left_state.time)
        / (right_state.time - left_state.time)
    )

    expected_mean = (
        left_state.values[0, 0]
        + alpha
        * (
            right_state.values[0, 0]
            - left_state.values[0, 0]
        )
    )

    expected_variance = (
        volatility**2
        * (times - left_state.time)
        * (right_state.time - times)
        / (right_state.time - left_state.time)
    )

    empirical_mean = np.mean(
        samples,
        axis=0,
    )

    empirical_variance = np.var(
        samples,
        axis=0,
    )

    assert np.allclose(
        empirical_mean,
        expected_mean,
        atol=0.1,
    )

    assert np.allclose(
        empirical_variance,
        expected_variance,
        atol=0.2,
    )


def test_brownian_bridge_empirical_covariance() -> None:
    left_state = State(
        time=0.0,
        values=np.array([
            [0.0],
        ]),
    )

    right_state = State(
        time=10.0,
        values=np.array([
            [0.0],
        ]),
    )

    times = np.array([
        1.0,
        2.5,
        6.0,
        9.0,
    ])

    volatility = 2.0
    n_samples = 30_000

    rng = np.random.default_rng(42)

    samples = np.empty(
        (
            n_samples,
            len(times),
        ),
        dtype=np.float64,
    )

    for i in range(n_samples):
        result = brownian_bridge(
            left_state=left_state,
            right_state=right_state,
            times=times,
            volatility=volatility,
            rng=rng,
        )

        samples[i] = result.values[:, 0, 0]

    shifted_times = (
        times
        - left_state.time
    )

    total_time = (
        right_state.time
        - left_state.time
    )

    expected_covariance = np.empty(
        (
            len(times),
            len(times),
        ),
        dtype=np.float64,
    )

    for i in range(len(times)):
        for j in range(len(times)):
            expected_covariance[i, j] = (
                volatility**2
                * (
                    min(
                        shifted_times[i],
                        shifted_times[j],
                    )
                    - (
                        shifted_times[i]
                        * shifted_times[j]
                        / total_time
                    )
                )
            )

    empirical_covariance = np.cov(
        samples,
        rowvar=False,
        ddof=0,
    )

    assert np.allclose(
        empirical_covariance,
        expected_covariance,
        atol=0.2,
    )
