import numpy as np
import pytest

from dynsample.core.linear_model import LinearSDE


# Check that a two-node model accepts one shared noise source.
def test_linear_sde_accepts_two_nodes() -> None:
    model = LinearSDE(
        drift=[
            [-0.7, 0.0],
            [0.4, -0.5],
        ],
        offset=[7.0, 1.0],
        diffusion=[
            [1.0],
            [0.5],
        ],
    )

    assert model.n_nodes == 2
    assert model.drift.shape == (2, 2)
    assert model.offset.shape == (2,)
    assert model.diffusion.shape == (2, 1)

    np.testing.assert_array_equal(
        model.drift,
        [[-0.7, 0.0], [0.4, -0.5]],
    )
    np.testing.assert_array_equal(model.offset, [7.0, 1.0])
    np.testing.assert_array_equal(model.diffusion, [[1.0], [0.5]])

    for values in (model.drift, model.offset, model.diffusion):
        assert values.dtype == np.dtype(np.float64)


# Zero coefficients are valid for both one-node and two-node models.
@pytest.mark.parametrize("n_nodes", [1, 2])
def test_linear_sde_accepts_zero_coefficients(n_nodes: int) -> None:
    model = LinearSDE(
        drift=np.zeros((n_nodes, n_nodes)),
        offset=np.zeros(n_nodes),
        diffusion=np.zeros((n_nodes, 1)),
    )

    assert model.n_nodes == n_nodes
    assert np.all(model.drift == 0.0)
    assert np.all(model.offset == 0.0)
    assert np.all(model.diffusion == 0.0)


# Reject malformed coefficient dimensions.
@pytest.mark.parametrize(
    "field, value, message",
    [
        ("drift", 0.0, "non-empty square matrix"),
        ("drift", [0.0, 0.0], "non-empty square matrix"),
        ("drift", np.zeros((2, 3)), "non-empty square matrix"),
        ("drift", np.zeros((0, 0)), "non-empty square matrix"),
        ("offset", np.zeros(3), "offset must have shape"),
        ("offset", np.zeros((2, 1)), "offset must have shape"),
        ("diffusion", np.zeros(2), "diffusion must have shape"),
        ("diffusion", np.zeros((3, 1)), "diffusion must have shape"),
        ("diffusion", np.zeros((2, 0)), "diffusion must have shape"),
    ],
)
def test_linear_sde_rejects_invalid_shapes(
    field,
    value,
    message,
) -> None:
    coefficients = {
        "drift": np.zeros((2, 2)),
        "offset": np.zeros(2),
        "diffusion": np.eye(2),
    }
    coefficients[field] = value

    with pytest.raises(ValueError, match=message):
        LinearSDE(**coefficients)


# Reject NaN and infinity in every coefficient array.
@pytest.mark.parametrize("field", ["drift", "offset", "diffusion"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_linear_sde_rejects_nonfinite_values(
    field,
    bad_value,
) -> None:
    coefficients = {
        "drift": np.zeros((2, 2)),
        "offset": np.zeros(2),
        "diffusion": np.eye(2),
    }
    coefficients[field].flat[0] = bad_value

    with pytest.raises(
        ValueError,
        match=f"{field} must contain only finite values",
    ):
        LinearSDE(**coefficients)


# Changing the original inputs must not change the stored model.
def test_linear_sde_copies_input_arrays() -> None:
    drift = np.array([[-0.7, 0.0], [0.4, -0.5]])
    offset = np.array([7.0, 1.0])
    diffusion = np.array([[1.0], [0.5]])

    model = LinearSDE(
        drift=drift,
        offset=offset,
        diffusion=diffusion,
    )

    drift[:] = 99.0
    offset[:] = 99.0
    diffusion[:] = 99.0

    np.testing.assert_array_equal(
        model.drift,
        [[-0.7, 0.0], [0.4, -0.5]],
    )
    np.testing.assert_array_equal(model.offset, [7.0, 1.0])
    np.testing.assert_array_equal(model.diffusion, [[1.0], [0.5]])