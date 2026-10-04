from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

# Store constant coefficients for dX = (K X + b) dt + B dW.
@dataclass
class LinearSDE:
    drift: NDArray[np.float64]
    offset: NDArray[np.float64]
    diffusion: NDArray[np.float64]

    def __post_init__(self) -> None:

        self.drift = np.array(
            self.drift,
            dtype = np.float64,
            copy = True
        )

        self.offset = np.array(
            self.offset,
            dtype = np.float64,
            copy = True
        )

        self.diffusion = np.array(
            self.diffusion,
            dtype = np.float64,
            copy = True
        )

        if (
            self.drift.ndim != 2
            or self.drift.shape[0] != self.drift.shape[1]
            or self.drift.shape[0] == 0
        ):
            raise ValueError(
                "drift must be a non-empty square matrix"
            )

        n_nodes = self.drift.shape[0]

        if self.offset.shape != (n_nodes,):
            raise ValueError(
                "offset must have shape (n_nodes,)"
            )

        if (
            self.diffusion.ndim != 2
            or self.diffusion.shape[0] != n_nodes
            or self.diffusion.shape[1] == 0
        ):
            raise ValueError(
                "diffusion must have shape (n_nodes, n_noise_sources) "
                "with at least one noise source"
            )

        for name, values in  (
            ("drift", self.drift),
            ("offset", self.offset),
            ("diffusion", self.diffusion),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"{name} must contain only finite values"
                )

    @property
    def n_nodes(self) -> int:
        return self.drift.shape[0]