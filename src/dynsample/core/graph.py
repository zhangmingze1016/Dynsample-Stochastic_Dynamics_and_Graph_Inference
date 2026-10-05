from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

@dataclass
class Graph:
    """
    The graph by which the several nodes of a dynamical system
    are joined one to another.

    Parameters
    ----------
    adjacency:
        The matrix declaring the connection from every node unto
        every other node.

        Shape: (N, N)
        Type: NDArray[np.float64]

        Axis 0:
            Source node.

        Axis 1:
            Target node.

        adjacency[i, j] is the weight of the connection proceeding
        from node i unto node j.

        No symmetry is required; the graph may therefore be either
        directed or undirected.
    """

    # Shape: (N, N)
    # Axis 0: source node
    # Axis 1: target node
    # adjacency[i, j] = weight of the connection from node i to node j.

    adjacency: NDArray[np.float64]

    def __post_init__(self) -> None:
        # Let the adjacency matrix be reduced to a uniform numerical form,
        # that subsequent computations may proceed without ambiguity.
        self.adjacency = np.asarray(
            self.adjacency,
            dtype = np.float64,
        )

        # An adjacency matrix ought to possess two dimensions.
        if self.adjacency.ndim != 2:
            raise ValueError(
                "adjacency must have shape (n_nodes, n_nodes)"
            )

        # The number of source nodes must equal the number of target nodes;
        # hence every proper adjacency matrix must be square.
        if self.adjacency.shape[0] != self.adjacency.shape[1]:
            raise ValueError(
                "adjacency must be a square matrix"
            )        

        # Every connection weight ought to be a finite quantity.
        if not np.all(np.isfinite(self.adjacency)):
            raise ValueError(
                "adjacency must contain only finite values"
            )

    @property
    def n_nodes(self) -> int:
        """return the number of nodes belonging to the graph"""
        return self.adjacency.shape[0]

    @property
    def degree(self) -> NDArray[np.float64]:
        """
        Return the degree belonging to each node.

        Shape: (N,)

        degree[i] is the sum of the weights proceeding from node i.
        """
        return np.sum(
            self.adjacency,
            axis = 1
        )

    @property
    def degree_matrix(self) -> NDArray[np.float64]:
        """
        Return the diagonal degree matrix of the graph.

        Shape: (N, N)
        """
        return np.diag(self.degree)

    @property
    def laplacian(self) -> NDArray[np.float64]:
        """
        Return the graph Laplacian.

        Shape: (N, N)

        L = D - A
        """
        return self.degree_matrix - self.adjacency

    @property
    def is_directed(self) -> bool:
        """
        Return whether the graph possesses unequal connections
        in opposite directions.
        """
        return not np.allclose(
            self.adjacency,
            self.adjacency.T,
        )

# Convert direct drift effects into source-to-target graph weights.
def graph_from_drift(
        drift: NDArray[np.float64],
        threshold: float = 0.0,
) -> Graph:
    """Convert drift to a signed graph, excluding self-dynamics."""

    drift = np.asarray(drift, dtype = np.float64)

    if (
        drift.ndim != 2
        or drift.shape[0] != drift.shape[1]
        or drift.shape[0] == 0
    ):
        raise ValueError("drift must be a non-empty square matrix")

    if not np.all(np.isfinite(drift)):
        raise ValueError("drift must contain only finite values")

    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")

    adjacency = drift.T.copy()

    np.fill_diagonal(adjacency, 0.0)

    adjacency[np.abs(adjacency) <= threshold] = 0.0

    return Graph(adjacency = adjacency)

