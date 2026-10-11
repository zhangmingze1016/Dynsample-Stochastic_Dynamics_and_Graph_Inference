import numpy as np
import pytest

from dynsample.core.trajectory import Trajectory
from dynsample.metrics.prediction import prediction_metrics


# Check overall and per-node scores against hand-calculated values.
def test_prediction_metrics_matches_hand_calculation():
    times = np.array([0.0, 0.2])

    observed = Trajectory(
        times=times,
        values=np.zeros((2, 2, 2)),
    )
    predicted = Trajectory(
        times=times,
        values=np.array([
            [[1.0, -1.0], [2.0, -2.0]],
            [[3.0, -3.0], [4.0, -4.0]],
        ]),
    )

    result = prediction_metrics(observed, predicted)

    assert result["rmse"] == pytest.approx(np.sqrt(7.5))
    assert result["mae"] == pytest.approx(2.5)

    np.testing.assert_allclose(
        result["rmse_per_node"],
        [np.sqrt(5.0), np.sqrt(10.0)],
    )
    np.testing.assert_allclose(
        result["mae_per_node"],
        [2.0, 3.0],
    )


# Identical trajectories must have zero prediction error.
def test_prediction_metrics_identical_trajectories():
    trajectory = Trajectory(
        times=np.array([0.0, 1.0]),
        values=np.array([
            [[1.0], [2.0]],
            [[3.0], [4.0]],
        ]),
    )

    result = prediction_metrics(trajectory, trajectory)

    assert result["rmse"] == 0.0
    assert result["mae"] == 0.0

    np.testing.assert_array_equal(
        result["rmse_per_node"], np.zeros(2)
    )
    np.testing.assert_array_equal(
        result["mae_per_node"], np.zeros(2)
    )


# A single observation is sufficient to compute a prediction error.
def test_prediction_metrics_accepts_one_observation():
    observed = Trajectory(
        times=np.array([2.0]),
        values=np.array([[[5.0]]]),
    )
    predicted = Trajectory(
        times=np.array([2.0]),
        values=np.array([[[2.0]]]),
    )

    result = prediction_metrics(observed, predicted)

    assert result["rmse"] == pytest.approx(3.0)
    assert result["mae"] == pytest.approx(3.0)
    np.testing.assert_allclose(result["rmse_per_node"], [3.0])
    np.testing.assert_allclose(result["mae_per_node"], [3.0])


# Irregular time intervals must not introduce duration weighting.
def test_prediction_metrics_weights_observations_equally():
    times = np.array([0.0, 0.1, 10.0])

    observed = Trajectory(
        times=times,
        values=np.zeros((3, 1, 1)),
    )
    predicted = Trajectory(
        times=times,
        values=np.array([[[0.0]], [[0.0]], [[3.0]]]),
    )

    result = prediction_metrics(observed, predicted)

    assert result["rmse"] == pytest.approx(np.sqrt(3.0))
    assert result["mae"] == pytest.approx(1.0)


# Different node or feature counts must not be broadcast together.
@pytest.mark.parametrize("shape", [(2, 2, 1), (2, 1, 2)])
def test_prediction_metrics_rejects_shape_mismatch(shape):
    times = np.array([0.0, 1.0])

    observed = Trajectory(
        times=times,
        values=np.zeros((2, 1, 1)),
    )
    predicted = Trajectory(
        times=times,
        values=np.zeros(shape),
    )

    with pytest.raises(ValueError, match="matching shapes"):
        prediction_metrics(observed, predicted)


# Equal array shapes do not imply matching observation times.
def test_prediction_metrics_rejects_time_mismatch():
    observed = Trajectory(
        times=np.array([0.0, 1.0]),
        values=np.zeros((2, 1, 1)),
    )
    predicted = Trajectory(
        times=np.array([0.0, 1.1]),
        values=np.zeros((2, 1, 1)),
    )

    with pytest.raises(ValueError, match="times must match exactly"):
        prediction_metrics(observed, predicted)


# Empty node or feature axes cannot produce meaningful averages.
@pytest.mark.parametrize("shape", [(2, 0, 1), (2, 1, 0)])
def test_prediction_metrics_rejects_empty_values(shape):
    trajectory = Trajectory(
        times=np.array([0.0, 1.0]),
        values=np.zeros(shape),
    )

    with pytest.raises(ValueError, match="non-empty"):
        prediction_metrics(trajectory, trajectory)


# Recheck finiteness because stored arrays can be mutated after construction.
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("target", ["observed", "predicted"])
def test_prediction_metrics_rejects_nonfinite_values(
    bad_value, target
):
    observed = Trajectory(
        times=np.array([0.0]),
        values=np.zeros((1, 1, 1)),
    )
    predicted = Trajectory(
        times=np.array([0.0]),
        values=np.zeros((1, 1, 1)),
    )

    if target == "observed":
        observed.values[0, 0, 0] = bad_value
    else:
        predicted.values[0, 0, 0] = bad_value

    with pytest.raises(ValueError, match="values must be finite"):
        prediction_metrics(observed, predicted)


# Finite inputs can still overflow during subtraction or squaring.
@pytest.mark.parametrize(
    "observed_value, predicted_value",
    [
        (-1e308, 1e308),
        (0.0, 1e200),
    ],
)
def test_prediction_metrics_rejects_error_overflow(
    observed_value, predicted_value
):
    observed = Trajectory(
        times=np.array([0.0]),
        values=np.array([[[observed_value]]]),
    )
    predicted = Trajectory(
        times=np.array([0.0]),
        values=np.array([[[predicted_value]]]),
    )

    with pytest.raises(
        FloatingPointError,
        match="errors or squared errors are non-finite",
    ):
        prediction_metrics(observed, predicted)