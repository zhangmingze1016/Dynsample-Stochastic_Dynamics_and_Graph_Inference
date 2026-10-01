import numpy as np
import pytest

from dynsample.core.graph import Graph


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