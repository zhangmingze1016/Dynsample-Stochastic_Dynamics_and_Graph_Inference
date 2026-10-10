from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from dynsample.core.state import State

@dataclass
class Trajectory:
    """
    A record of the several states through which a dynamical system
    hath passed in the course of time.

    Parameters
    ----------
    times: NDArray
        A one-dimensional array containing the successive times at
        which the several states are reckoned.

        Its shape is (n_steps,).

    values: np.ndarray
        An array containing the state of the whole system at every
        recorded time.

        Its shape is (n_steps, n_nodes, n_features).

        The first dimension distinguishes one time from each another;
        the second, one node from another; and the third, the several
        features belonging to each node.
    """

    times: NDArray[np.float64]
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        #Let both quantities be reduced to one common numerical form,
        #that all subsequent computations may proceed without needless
        #distinction between Python sequences and NumPy arrays.

        # Own the arrays so changes to caller inputs cannot invalidate them.
        self.times = np.array(self.times, dtype=np.float64, copy=True)
        self.values = np.array(self.values, dtype=np.float64, copy=True)

        #The times ought to form one simple succession:
        #
        #       [t_0, t_1, ..., t_(T - 1)]
        #
        #No second dimension is required, for each state hath but one
        #time assigned to it.

        if self.times.ndim != 1:
            raise ValueError(
                "time must have shape (n_steps,)"
            )

        # At every time there stands a matrix of state values.
        # The whole record must therefore possess three dimensions:
        #
        #     time x node x feature
        #
        # or, in symbols,
        #
        #     (n_steps, n_nodes, n_features).

        if self.values.ndim != 3:
            raise ValueError(
                "values must have shape"
                "(n_steps, n_nodes, n_features)"
            )

        # There ought to be one, and only one, state for every time
        # entered in the record.  A disagreement between these numbers
        # would leave either a time without a state, or a state without
        # its proper time.

        if self.times.shape[0] != self.values.shape[0]:
            raise ValueError(
                "times and values must contain the same "
                "number of steps"
            )

        #An empty record can furnish us with no trajectory at all.

        if self.times.size == 0:
            raise ValueError(
                "trajectory must contain at least one time step"
            )
        
        # Every time must be a finite number.  Neither an infinite time
        # nor one wholly undefined can hold a proper place in the
        # chronological order of the trajectory.
        if not np.all(np.isfinite(self.times)):
            raise ValueError(
                "times must contain only finite values"
            )
        # Time must ever advance and never stand still or retreat:
        #
        #     t_0 < t_1 < ... < t_(T-1).
        #
        # np.diff gives each successive interval,
        #
        #     t_(k+1) - t_k,
        #
        # every one of which must consequently be greater than zero.

        with np.errstate(over="ignore", invalid="ignore"):
            intervals = np.diff(self.times)

        if not np.all(np.isfinite(intervals)):
            raise ValueError("time intervals must be finite")

        if np.any(intervals <= 0):
            raise ValueError(
                "times must be strictly increasing"
            )

        if not np.all(np.isfinite(self.values)):
            raise ValueError(
                "values must contain only finite values"
            )

    def __len__(self) -> int:
        """
        Return the number of recorded times contained in the trajectory.
        """

        return self.times.shape[0]

    @property
    def n_steps(self) -> int:
        """
        Return the number of time steps contained in the record.
        """
        return self.values.shape[0]

    @property
    def n_nodes(self) -> int:
        """
        Return the number of nodes in the system.
        """
        return self.values.shape[1]

    @property
    def n_features(self) -> int:
        """
        Return the number of features assigned to every node.
        """
        return self.values.shape[2]

    def state_at(self, index: int) -> State:
        """
        Return an independent state snapshot at the given index.

        Mutating the returned values does not change this trajectory,
        and later trajectory changes do not change the snapshot.

        The selected portion of ``values`` hath shape
        (n_nodes, n_features), and is therefore fit to constitute
        a single State.
        """
        return State(
            time = self.times[index],
            values = self.values[index].copy(),
        )
