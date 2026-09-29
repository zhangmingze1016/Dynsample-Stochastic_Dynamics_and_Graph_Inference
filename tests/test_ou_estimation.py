import numpy as np
from dynsample.core.trajectory import Trajectory
from dynsample.estimation.ou import ou_negative_log_likelihood
import pytest
from dynsample.estimation.ou import _initial_ou_parameters
from dynsample.core.state import State
from dynsample.simulation.ou import simulate_ou
from dynsample.estimation.ou import fit_ou, _initial_ou_parameters
from dynsample.estimation.ou import _profile_ou_mu, _profile_ou_sigma
from dynsample.estimation.ou import _profile_ou_negative_log_likelihood
from scipy.optimize import minimize, minimize_scalar

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

@pytest.mark.parametrize(
    "alpha, mu, sigma",
    [
        (0.0, 10.0, 2.0),
        (-0.5, 10.0, 2.0),
        (0.7, 10.0, 0.0),
        (0.7, 10.0, -1.0),
        (np.nan, 10.0, 2.0),
        (0.7, np.inf, 2.0),
        (0.7, 10.0, np.nan),
    ],
)
def test_ou_likelihood_rejects_invalid_parameters(
    alpha: float,
    mu: float,
    sigma: float,
) -> None:
    trajectory = Trajectory(
        times=np.array([0.0, 1.0]),
        values=np.array([[[14.0]], [[13.0]]]),
    )

    with pytest.raises(ValueError):
        ou_negative_log_likelihood(
            trajectory=trajectory,
            mean_reversion=alpha,
            long_run_mean=mu,
            volatility=sigma,
        )

# The optimizer works in (log(alpha), mu, log(sigma)) coordinates.
from dynsample.estimation.ou import _ou_objective, fit_ou


@pytest.mark.parametrize("mu", [-10.0, 0.0, 10.0])
def test_objective_matches_direct_likelihood(mu):
    trajectory = Trajectory(
        times=np.array([0.0, 0.2, 1.3]),
        values=np.array([[[1.0]], [[0.7]], [[-0.2]]]),
    )
    search_parameters = np.array([np.log(0.7), mu, np.log(1.5)])
    actual = _ou_objective(search_parameters, trajectory)
    expected = ou_negative_log_likelihood(trajectory, 0.7, mu, 1.5)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


@pytest.mark.parametrize(
    "initial, bounds, message",
    [
        ((0.7, 10.0), ((0.01, 5.0), (-20.0, 20.0), (0.01, 10.0)), "three initial"),
        ((0.7, 10.0, 1.5), (0.01, 5.0), "three initial"),
        ((np.nan, 10.0, 1.5), ((0.01, 5.0), (-20.0, 20.0), (0.01, 10.0)), "finite"),
        ((0.7, 10.0, 1.5), ((0.01, np.inf), (-20.0, 20.0), (0.01, 10.0)), "finite"),
        ((0.7, 10.0, 1.5), ((5.0, 0.01), (-20.0, 20.0), (0.01, 10.0)), "lower bound"),
        ((0.7, 10.0, 1.5), ((0.0, 5.0), (-20.0, 20.0), (0.01, 10.0)), "positive"),
        ((0.7, 10.0, 1.5), ((0.01, 5.0), (-20.0, 20.0), (-1.0, 10.0)), "positive"),
        ((6.0, 10.0, 1.5), ((0.01, 5.0), (-20.0, 20.0), (0.01, 10.0)), "within"),
    ],
)
def test_fit_rejects_invalid_configuration(initial, bounds, message):
    trajectory = Trajectory(
        times=np.array([0.0, 1.0]),
        values=np.array([[[14.0]], [[13.0]]]),
    )
    with pytest.raises(ValueError, match=message):
        fit_ou(trajectory, initial, bounds)


@pytest.mark.parametrize("shape", [(1, 1, 1), (3, 2, 1), (3, 1, 2)])
def test_fit_rejects_unsupported_trajectory(shape):
    trajectory = Trajectory(
        times=np.arange(shape[0], dtype=float),
        values=np.zeros(shape),
    )
    with pytest.raises(ValueError):
        fit_ou(trajectory, (0.7, 10.0, 1.5),
               ((0.01, 5.0), (-20.0, 20.0), (0.01, 10.0)))


def test_fit_matches_independent_regular_grid_mle():
    # Generate an AR(1) path independently of the OU simulation implementation.
    rng = np.random.default_rng(42)
    values = np.empty(401)
    values[0] = -3.0
    for i in range(1, values.size):
        values[i] = 0.8 * values[i - 1] - 0.6 + 0.5 * rng.standard_normal()
    dt = 0.25
    trajectory = Trajectory(
        times=np.arange(values.size) * dt,
        values=values[:, None, None],
    )

    # On a regular grid, conditional Gaussian MLE is ordinary least squares.
    design = np.column_stack((values[:-1], np.ones(values.size - 1)))
    phi, intercept = np.linalg.lstsq(design, values[1:], rcond=None)[0]
    residual = values[1:] - design @ np.array([phi, intercept])
    q = np.mean(residual**2)
    assert 0.0 < phi < 1.0  # Ensures a finite positive OU reversion estimate.
    alpha = -np.log(phi) / dt
    expected_parameters = np.array([
        alpha, intercept / (1.0 - phi),
        np.sqrt(2.0 * alpha * q / (1.0 - phi**2)),
    ])
    expected_score = 0.5 * residual.size * (np.log(2.0 * np.pi * q) + 1.0)
    bounds = np.array([(0.01, 5.0), (-10.0, 10.0), (0.01, 5.0)])
    assert np.all(expected_parameters > bounds[:, 0])
    assert np.all(expected_parameters < bounds[:, 1])

    original = trajectory.values.copy()
    for start in [(0.3, -1.0, 0.5), (2.0, -5.0, 2.0)]:
        initial = np.array(start)
        initial_copy = initial.copy()
        bounds_copy = bounds.copy()
        result = fit_ou(trajectory, initial, bounds)
        assert result.success, result.message
        fitted = np.array([np.exp(result.x[0]), result.x[1], np.exp(result.x[2])])
        np.testing.assert_allclose(fitted, expected_parameters, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(result.fun, expected_score, rtol=0.0, atol=1e-5)
        assert result.fun < ou_negative_log_likelihood(trajectory, *start)
        np.testing.assert_allclose(result.fun, _ou_objective(result.x, trajectory))
        np.testing.assert_array_equal(initial, initial_copy)
        np.testing.assert_array_equal(bounds, bounds_copy)
    np.testing.assert_array_equal(trajectory.values, original)

def test_initial_ou_parameters_irregular_times() -> None:
    trajectory = Trajectory(
        times=np.array([2.0, 3.0, 5.0]),
        values=np.array([[[14.0]], [[13.0]], [[10.0]]]),
    )

    actual = _initial_ou_parameters(trajectory)

    expected = (
        1.0 / 3.0,
        37.0 / 3.0,
        np.sqrt(2.75),
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12)

def test_initial_ou_parameters_rejects_constant_path() -> None:
    trajectory = Trajectory(
        times=np.array([0.0, 0.5, 2.0]),
        values=np.array([[[10.0]], [[10.0]], [[10.0]]]),
    )

    with pytest.raises(ValueError, match="constant trajectory"):
        _initial_ou_parameters(trajectory)

def test_fit_ou_with_automatic_initial_parameters() -> None:
    trajectory = simulate_ou(
        initial_state=State(time=0.0, values=np.array([[10.0]])),
        times=np.linspace(0.0, 100.0, 1001),
        mean_reversion=0.7,
        long_run_mean=10.0,
        volatility=1.5,
        rng=np.random.default_rng(42),
    )

    initial = _initial_ou_parameters(trajectory)
    bounds = ((0.001, 5.0), (-20.0, 20.0), (0.01, 10.0))

    automatic = fit_ou(
        trajectory=trajectory,
        initial_parameters=None,
        parameter_bounds=bounds,
    )

    explicit = fit_ou(
        trajectory=trajectory,
        initial_parameters=initial,
        parameter_bounds=bounds,
    )

    assert automatic.success, automatic.message
    assert explicit.success, explicit.message
    assert np.isfinite(automatic.fun)
    np.testing.assert_allclose(automatic.x, explicit.x)
    np.testing.assert_allclose(automatic.fun, explicit.fun)


@pytest.mark.parametrize(
    "times, expected",
    [
        ([0.0, 1.0, 2.0], 9.5),
        ([0.0, 1.0, 3.0], 141.0 / 14.0),
    ],
)
def test_profile_ou_mu_matches_hand_calculation(times, expected) -> None:
    trajectory = Trajectory(
        times=np.array(times),
        values=np.array([14.0, 13.0, 10.0])[:, None, None],
    )
    original_times = trajectory.times.copy()
    original_values = trajectory.values.copy()

    actual = _profile_ou_mu(trajectory, np.log(2.0))

    # Regular grid: B/A = 9.5. Irregular grid: B/A = 18.8 / (28/15).
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    np.testing.assert_array_equal(trajectory.times, original_times)
    np.testing.assert_array_equal(trajectory.values, original_values)


@pytest.mark.parametrize("sigma", [0.4, 2.0])
def test_profile_ou_mu_minimizes_existing_likelihood(sigma) -> None:
    trajectory = Trajectory(
        times=np.array([0.0, 0.07, 0.4, 1.8, 2.0, 4.5]),
        values=np.array([-3.0, -2.7, -3.4, -1.0, -1.3, -2.0])[:, None, None],
    )
    alpha = 0.8
    actual = _profile_ou_mu(trajectory, alpha)

    # Independently minimize the existing likelihood over mu alone.
    def objective(mu):
        return ou_negative_log_likelihood(trajectory, alpha, mu, sigma)

    reference = minimize_scalar(
        objective, bounds=(-20.0, 20.0), method="bounded",
        options={"xatol": 1e-10},
    )

    assert reference.success
    np.testing.assert_allclose(actual, reference.x, rtol=0.0, atol=1e-6)
    assert objective(actual) < objective(actual - 0.5)
    assert objective(actual) < objective(actual + 0.5)


@pytest.mark.parametrize("alpha", [0.0, -0.5, np.nan, np.inf, -np.inf])
def test_profile_ou_mu_rejects_invalid_alpha(alpha) -> None:
    trajectory = Trajectory(
        times=np.array([0.0, 1.0]),
        values=np.array([14.0, 13.0])[:, None, None],
    )

    with pytest.raises(ValueError, match="finite and positive"):
        _profile_ou_mu(trajectory, alpha)


@pytest.mark.parametrize("shape", [(1, 1, 1), (3, 2, 1), (3, 1, 2)])
def test_profile_ou_mu_rejects_unsupported_trajectory(shape) -> None:
    trajectory = Trajectory(
        times=np.arange(shape[0], dtype=float),
        values=np.zeros(shape),
    )

    with pytest.raises(ValueError):
        _profile_ou_mu(trajectory, 0.7)


@pytest.mark.parametrize(
    "times, variance_factor",
    [
        ([0.0, 1.0, 2.0], 25.0 / 6.0),
        ([0.0, 1.0, 3.0], 27.0 / 14.0),
    ],
)
def test_profile_ou_sigma_matches_hand_calculation(times, variance_factor) -> None:
    trajectory = Trajectory(
        times=np.array(times),
        values=np.array([14.0, 13.0, 10.0])[:, None, None],
    )
    original_times = trajectory.times.copy()
    original_values = trajectory.values.copy()

    actual = _profile_ou_sigma(trajectory, np.log(2.0))

    # Hand-derived variance: factor * log(2); the function must return sigma.
    expected = np.sqrt(variance_factor * np.log(2.0))
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    np.testing.assert_array_equal(trajectory.times, original_times)
    np.testing.assert_array_equal(trajectory.values, original_values)


def test_profile_ou_sigma_minimizes_existing_likelihood() -> None:
    trajectory = Trajectory(
        times=np.array([0.0, 0.07, 0.4, 1.8, 2.0, 4.5]),
        values=np.array([-3.0, -2.7, -3.4, -1.0, -1.3, -2.0])[:, None, None],
    )
    alpha = 0.8
    mu = _profile_ou_mu(trajectory, alpha)
    actual = _profile_ou_sigma(trajectory, alpha)

    # Optimize the existing likelihood independently in log(sigma).
    def objective(log_sigma):
        return ou_negative_log_likelihood(
            trajectory, alpha, mu, float(np.exp(log_sigma))
        )

    reference = minimize_scalar(
        objective, bounds=(-10.0, 10.0), method="bounded",
        options={"xatol": 1e-10},
    )

    assert reference.success
    np.testing.assert_allclose(actual, np.exp(reference.x), rtol=1e-6)
    assert objective(np.log(actual)) < objective(np.log(actual * 0.5))
    assert objective(np.log(actual)) < objective(np.log(actual * 2.0))


def test_profile_ou_sigma_rejects_zero_residual_variance() -> None:
    trajectory = Trajectory(
        times=np.array([0.0, 0.3, 2.0]),
        values=np.zeros((3, 1, 1)),
    )

    with pytest.raises(ValueError, match="zero residual variance"):
        _profile_ou_sigma(trajectory, 0.7)


@pytest.mark.parametrize(
    "times, q1, q2",
    [
        ([0.0, 1.0, 2.0], 25.0 / 16.0, 25.0 / 16.0),
        ([0.0, 1.0, 3.0], 81.0 / 112.0, 405.0 / 448.0),
    ],
)
def test_profile_ou_likelihood_matches_hand_calculation(times, q1, q2) -> None:
    trajectory = Trajectory(
        times=np.array(times),
        values=np.array([14.0, 13.0, 10.0])[:, None, None],
    )

    actual = _profile_ou_negative_log_likelihood(trajectory, np.log(2.0))

    # Hand-derived fitted transition variances; standardized squares sum to n=2.
    expected = 0.5 * (
        np.log(2.0 * np.pi * q1) + np.log(2.0 * np.pi * q2) + 2.0
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


@pytest.mark.parametrize("alpha", [0.1, 0.8, 3.0])
def test_profile_ou_likelihood_matches_joint_nuisance_optimization(alpha) -> None:
    trajectory = Trajectory(
        times=np.array([0.0, 0.07, 0.4, 1.8, 2.0, 4.5]),
        values=np.array([-3.0, -2.7, -3.4, -1.0, -1.3, -2.0])[:, None, None],
    )

    # Independently search mu and log(sigma), keeping alpha fixed.
    def objective(parameters):
        mu, log_sigma = parameters
        return ou_negative_log_likelihood(
            trajectory, alpha, float(mu), float(np.exp(log_sigma))
        )

    reference = minimize(
        objective,
        x0=np.array([-2.0, 0.0]),
        method="Nelder-Mead",
        options={"xatol": 1e-9, "fatol": 1e-10, "maxiter": 2000},
    )
    actual = _profile_ou_negative_log_likelihood(trajectory, alpha)

    assert reference.success, reference.message
    np.testing.assert_allclose(actual, reference.fun, rtol=0.0, atol=1e-8)


def test_profile_ou_likelihood_rejects_zero_residual_variance() -> None:
    trajectory = Trajectory(
        times=np.array([0.0, 0.3, 2.0]),
        values=np.zeros((3, 1, 1)),
    )

    with pytest.raises(ValueError, match="zero residual variance"):
        _profile_ou_negative_log_likelihood(trajectory, 0.7)
