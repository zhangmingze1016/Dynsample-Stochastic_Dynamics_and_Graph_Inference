import numpy as np
from dynsample.core.trajectory import Trajectory
from dynsample.estimation.ou import ou_negative_log_likelihood

def test_ou_likelihood_irregular_times() -> None:
    trajectory = Trajectory(
        times = np.array([0.0, 1.0, 3.0]),
        values = np.array([[[14.0]], [[13.0]],[[10.0]]]),
    )

    alpha = np.log(2.0)

    actual = ou_negative_log_likelihood(
        trajectory = trajectory,
        mean_reversion = alpha,
        long_run_mean = 10.0,
        volatility = 2.0,
    )

    q1 = 1.5 /alpha
    q2 = 1.875 / alpha

    expected = 0.5 * (
        np.log(2.0 * np.pi * q1)
        + 1.0 / q1+ np.log(2.0 * np.pi * q2)
        + 0.75**2 / q2
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12)