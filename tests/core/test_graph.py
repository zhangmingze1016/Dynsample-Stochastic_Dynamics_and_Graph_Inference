import numpy as np
import pytest
from dynsample.core.graph import Graph, graph_from_drift

def test_graph_shape() -> None:
    adjacency = np.array([
        [0.0, 0.8, 0.0],
        [0.8, 0.0, 0.3],
        [0.0, 0.3, 0.0],
    ])

    graph = Graph(adjacency=adjacency)

    assert graph.adjacency.shape == (3, 3)
    assert graph.n_nodes == 3


def test_graph_converts_dtype() -> None:
    adjacency = [
        [0, 1],
        [1, 0],
    ]

    graph = Graph(adjacency=adjacency)

    assert graph.adjacency.dtype == np.float64


def test_graph_requires_square_matrix() -> None:
    adjacency = np.zeros((3, 2))

    with pytest.raises(ValueError):
        Graph(adjacency=adjacency)


def test_graph_requires_finite_values() -> None:
    adjacency = np.array([
        [0.0, np.nan],
        [1.0, 0.0],
    ])

    with pytest.raises(ValueError):
        Graph(adjacency=adjacency)


def test_graph_degree_and_laplacian() -> None:
    adjacency = np.array([
        [0.0, 0.8, 0.0],
        [0.8, 0.0, 0.3],
        [0.0, 0.3, 0.0],
    ])

    graph = Graph(adjacency=adjacency)

    expected_degree = np.array([
        0.8,
        1.1,
        0.3,
    ])

    expected_laplacian = np.array([
        [0.8, -0.8, 0.0],
        [-0.8, 1.1, -0.3],
        [0.0, -0.3, 0.3],
    ])

    assert np.allclose(
        graph.degree,
        expected_degree,
    )

    assert np.allclose(
        graph.laplacian,
        expected_laplacian,
    )


def test_graph_detects_direction() -> None:
    undirected = Graph(
        adjacency=np.array([
            [0.0, 0.8],
            [0.8, 0.0],
        ])
    )

    directed = Graph(
        adjacency=np.array([
            [0.0, 0.8],
            [0.2, 0.0],
        ])
    )

    assert undirected.is_directed is False
    assert directed.is_directed is True

# Drift K[i, j] acts from j to i; graph adjacency uses source-to-target.
def test_graph_from_drift_preserves_direction():
    drift = np.array([
        [-1.0, 0.0, 0.0],
        [0.8, -1.0, 0.0],
        [0.0, 0.6, -1.0],
    ])

    graph = graph_from_drift(drift)

    expected = np.array([
        [0.0, 0.8, 0.0],
        [0.0, 0.0, 0.6],
        [0.0, 0.0, 0.0],
    ])

    assert isinstance(graph, Graph)
    assert graph.n_nodes == 3
    np.testing.assert_array_equal(graph.adjacency, expected)


# A zero default threshold must retain negative and tiny nonzero weights.
def test_graph_from_drift_preserves_signed_weights():
    drift = np.array([
        [-1.0, 1e-12],
        [-0.4, -2.0],
    ])

    graph = graph_from_drift(drift)

    np.testing.assert_array_equal(
        graph.adjacency,
        [
            [0.0, -0.4],
            [1e-12, 0.0],
        ],
    )


# Remove both positive and negative weights at or below the threshold.
def test_graph_from_drift_applies_threshold():
    drift = np.array([
        [-1.0, 0.01, -0.01],
        [0.005, -1.0, 0.2],
        [-0.3, -0.009, -1.0],
    ])

    graph = graph_from_drift(drift, threshold=0.01)

    expected = np.array([
        [0.0, 0.0, -0.3],
        [0.0, 0.0, 0.0],
        [0.0, 0.2, 0.0],
    ])

    np.testing.assert_array_equal(graph.adjacency, expected)


# Self-dynamics alone must not create connections between nodes.
@pytest.mark.parametrize("n_nodes", [1, 3])
def test_graph_from_drift_removes_self_dynamics(n_nodes):
    drift = -np.eye(n_nodes)

    graph = graph_from_drift(drift)

    np.testing.assert_array_equal(
        graph.adjacency,
        np.zeros((n_nodes, n_nodes)),
    )


# Conversion and later graph edits must leave the supplied drift unchanged.
def test_graph_from_drift_does_not_modify_input():
    drift = np.array([
        [-1.0, 0.005],
        [-0.4, -2.0],
    ])
    original = drift.copy()

    graph = graph_from_drift(drift, threshold=0.01)

    np.testing.assert_array_equal(drift, original)

    graph.adjacency[0, 1] = 7.0

    np.testing.assert_array_equal(drift, original)


# Reject arrays that cannot describe a nonempty square drift matrix.
@pytest.mark.parametrize(
    "drift",
    [
        np.array(1.0),
        np.zeros(2),
        np.zeros((2, 3)),
        np.zeros((0, 0)),
        np.zeros((2, 2, 1)),
    ],
)
def test_graph_from_drift_rejects_invalid_shape(drift):
    with pytest.raises(ValueError, match="non-empty square matrix"):
        graph_from_drift(drift)


# Reject nonfinite coefficients before creating the graph.
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_graph_from_drift_rejects_nonfinite_values(value):
    drift = np.array([
        [-1.0, value],
        [0.2, -1.0],
    ])

    with pytest.raises(ValueError, match="finite values"):
        graph_from_drift(drift)


# Thresholds must be finite and non-negative.
@pytest.mark.parametrize("threshold", [-0.01, np.nan, np.inf, -np.inf])
def test_graph_from_drift_rejects_invalid_threshold(threshold):
    drift = -np.eye(2)

    with pytest.raises(ValueError, match="threshold must"):
        graph_from_drift(drift, threshold=threshold)