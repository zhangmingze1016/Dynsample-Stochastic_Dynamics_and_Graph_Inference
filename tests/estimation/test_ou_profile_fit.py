"""End-to-end checks against an independently derived regular-grid MLE."""

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from dynsample.core.trajectory import Trajectory
from dynsample.estimation import ou


@pytest.fixture
def auto_search_case(monkeypatch):
    """Control search outputs while retaining the real boundary diagnostics."""
    trajectory = Trajectory(
        times=np.array([0.0, 1.0, 2.0]),
        values=np.array([1.0, 4.0, 3.0])[:, None, None],
    )

    def configure(steps, boundary_nll=(20.0, 20.0), bounds=(0.1, 10.0)):
        calls = []
        boundaries = {
            "brownian": OptimizeResult(fun=boundary_nll[0]),
            "gaussian": OptimizeResult(fun=boundary_nll[1]),
        }
        monkeypatch.setattr(ou, "_initial_ou_alpha_bounds", lambda trajectory: bounds)
        monkeypatch.setattr(ou, "_ou_boundary_scores", lambda trajectory: boundaries)

        def search(trajectory, alpha_bounds, grid_size):
            index = len(calls)
            assert index < len(steps), "unexpected extra search"
            calls.append(alpha_bounds)
            location, score, success = steps[index]
            lower, upper = alpha_bounds
            if location == "lower":
                alpha = lower
            elif location == "upper":
                alpha = upper
            else:
                alpha = float(np.exp((np.log(lower) + np.log(upper)) / 2.0))
            return OptimizeResult(
                x=alpha, fun=score, success=success,
                scan=OptimizeResult(alphas=np.array([lower, upper])),
            )

        monkeypatch.setattr(ou, "_search_ou_profile", search)
        return trajectory, calls

    return configure


def test_auto_search_stops_at_interior_candidate(auto_search_case) -> None:
    trajectory, calls = auto_search_case([("middle", 10.0, True)])
    result = ou._search_ou_profile_auto(trajectory)
    assert result.success
    assert result.status == "interior_candidate"
    assert result.expansions == 0
    assert calls == [(0.1, 10.0)]


@pytest.mark.parametrize(
    "side, expected_bounds",
    [("lower", (0.01, 10.0)), ("upper", (0.1, 100.0))],
)
def test_auto_search_expands_correct_side(auto_search_case, side, expected_bounds) -> None:
    trajectory, calls = auto_search_case(
        [(side, 10.0, True), ("middle", 9.0, True)]
    )
    result = ou._search_ou_profile_auto(trajectory)
    assert result.success
    assert result.expansions == 1
    np.testing.assert_allclose(calls, [(0.1, 10.0), expected_bounds])
    assert result.fun == 9.0


def test_auto_search_expands_when_boundary_score_is_better(auto_search_case) -> None:
    trajectory, calls = auto_search_case(
        [("middle", 10.0, True), ("middle", 8.0, True)],
        boundary_nll=(9.0, 20.0),
    )
    result = ou._search_ou_profile_auto(trajectory)
    assert result.success
    np.testing.assert_allclose(calls[-1], (0.01, 10.0))
    assert result.fun == 8.0


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_auto_search_reports_boundary_limit(auto_search_case, side) -> None:
    trajectory, calls = auto_search_case(
        [(side, 10.0, True)], boundary_nll=(10.0, 10.0)
    )
    result = ou._search_ou_profile_auto(trajectory)
    assert not result.success
    assert result.status == "boundary_limit"
    assert len(calls) == 1


def test_auto_search_does_not_stop_if_other_boundary_is_better(auto_search_case) -> None:
    trajectory, calls = auto_search_case(
        [("lower", 10.0, True), ("middle", 8.0, True)],
        boundary_nll=(10.0, 9.0),
    )
    result = ou._search_ou_profile_auto(trajectory)
    assert result.success
    np.testing.assert_allclose(calls[-1], (0.01, 100.0))


def test_auto_search_keeps_best_candidate_at_expansion_limit(auto_search_case) -> None:
    trajectory, calls = auto_search_case(
        [("upper", 10.0, True), ("upper", 11.0, True)]
    )
    result = ou._search_ou_profile_auto(trajectory, max_expansions=1)
    assert not result.success
    assert result.status == "expansion_limit"
    assert len(calls) == 2
    assert result.expansions == 1
    assert result.fun == 10.0
    assert result.x == 10.0
    assert result.best_search is result.history[0].search


def test_auto_search_preserves_prior_result_on_incomplete_search(auto_search_case) -> None:
    trajectory, calls = auto_search_case(
        [("upper", 10.0, True), ("middle", 11.0, False)]
    )
    result = ou._search_ou_profile_auto(trajectory)
    assert not result.success
    assert result.status == "incomplete_search"
    assert len(calls) == 2
    assert result.fun == 10.0
    assert not result.history[-1].diagnostics.search_complete


def test_auto_search_stops_before_numeric_overflow(auto_search_case) -> None:
    trajectory, calls = auto_search_case(
        [("upper", 10.0, True)], bounds=(1.0, 1e308)
    )
    result = ou._search_ou_profile_auto(trajectory)
    assert not result.success
    assert result.status == "numerical_limit"
    assert len(calls) == 1


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"max_expansions": -1}, "max_expansions"),
        ({"max_expansions": True}, "max_expansions"),
        ({"max_expansions": 1.5}, "max_expansions"),
        ({"nll_tolerance": -1.0}, "nll_tolerance"),
        ({"nll_tolerance": np.nan}, "nll_tolerance"),
        ({"nll_tolerance": np.inf}, "nll_tolerance"),
    ],
)
def test_auto_search_rejects_invalid_controls(auto_search_case, kwargs, message) -> None:
    trajectory, calls = auto_search_case([])
    with pytest.raises(ValueError, match=message):
        ou._search_ou_profile_auto(trajectory, **kwargs)
    assert calls == []


def test_auto_search_matches_independent_regular_grid_mle() -> None:
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
    design = np.column_stack((values[:-1], np.ones(values.size - 1)))
    coefficients = np.linalg.lstsq(design, values[1:], rcond=None)[0]
    expected_alpha = -np.log(coefficients[0]) / dt
    residual = values[1:] - design @ coefficients
    q = np.mean(residual**2)
    expected_score = 0.5 * residual.size * (np.log(2.0 * np.pi * q) + 1.0)

    result = ou._search_ou_profile_auto(trajectory)

    assert result.success
    assert result.status == "interior_candidate"
    np.testing.assert_allclose(result.x, expected_alpha, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(result.fun, expected_score, rtol=0.0, atol=1e-7)


@pytest.mark.parametrize(
    "alpha, position, near_lower, near_upper",
    [
        (0.01, 0.0, True, False),
        (1.0, 0.5, False, False),
        (100.0, 1.0, False, True),
    ],
)
def test_search_diagnostics_use_logarithmic_position(
    alpha, position, near_lower, near_upper,
) -> None:
    search = OptimizeResult(
        x=alpha, fun=10.0, success=True,
        scan=OptimizeResult(alphas=np.array([0.01, 1.0, 100.0])),
    )
    boundaries = {
        "brownian": OptimizeResult(fun=13.0),
        "gaussian": OptimizeResult(fun=8.0),
    }

    result = ou._diagnose_ou_search(search, boundaries)

    np.testing.assert_allclose(result.alpha_search_position, position)
    assert result.near_lower_bound == near_lower
    assert result.near_upper_bound == near_upper
    assert result.alpha_bounds == (0.01, 100.0)
    assert result.brownian_nll_gap == 3.0
    assert result.gaussian_nll_gap == -2.0
    assert result.search_complete


def test_search_diagnostics_preserve_incomplete_status() -> None:
    search = OptimizeResult(
        x=1.0, fun=10.0, success=False,
        scan=OptimizeResult(alphas=np.array([0.01, 100.0])),
    )
    boundaries = {
        "brownian": OptimizeResult(fun=10.0),
        "gaussian": OptimizeResult(fun=10.0),
    }

    result = ou._diagnose_ou_search(search, boundaries)

    assert not result.search_complete
    assert result.brownian_nll_gap == 0.0
    assert result.gaussian_nll_gap == 0.0


@pytest.mark.parametrize(
    "alpha, score, brownian_score, message",
    [
        (200.0, 10.0, 13.0, "within valid alpha bounds"),
        (1.0, np.nan, 13.0, "within valid alpha bounds"),
        (1.0, 10.0, np.inf, "boundary scores must be finite"),
    ],
)
def test_search_diagnostics_reject_invalid_results(
    alpha, score, brownian_score, message,
) -> None:
    search = OptimizeResult(
        x=alpha, fun=score, success=True,
        scan=OptimizeResult(alphas=np.array([0.01, 100.0])),
    )
    boundaries = {
        "brownian": OptimizeResult(fun=brownian_score),
        "gaussian": OptimizeResult(fun=8.0),
    }

    with pytest.raises(ValueError, match=message):
        ou._diagnose_ou_search(search, boundaries)


def test_profile_fit_matches_regular_grid_mle() -> None:
    # Generate AR(1) observations without using the OU simulator.
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

    # Regular-grid Gaussian conditional MLE is an AR(1) least-squares fit.
    design = np.column_stack((values[:-1], np.ones(values.size - 1)))
    phi, intercept = np.linalg.lstsq(design, values[1:], rcond=None)[0]
    residual = values[1:] - design @ np.array([phi, intercept])
    q = np.mean(residual**2)
    assert 0.0 < phi < 1.0
    alpha = -np.log(phi) / dt
    expected = [alpha, intercept / (1.0 - phi), np.sqrt(2.0 * alpha * q / (1.0 - phi**2))]
    expected_score = 0.5 * residual.size * (np.log(2.0 * np.pi * q) + 1.0)
    original_values = trajectory.values.copy()
    original_times = trajectory.times.copy()

    result = ou.fit_ou_profile(trajectory, alpha_bounds=(0.01, 5.0))

    assert result.success
    np.testing.assert_allclose(
        [result.mean_reversion, result.long_run_mean, result.volatility],
        expected, rtol=1e-5, atol=1e-6,
    )
    np.testing.assert_allclose(result.fun, expected_score, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(result.x, result.mean_reversion)
    np.testing.assert_allclose(np.exp(result.log_alpha), result.mean_reversion)
    np.testing.assert_allclose(
        result.fun,
        ou.ou_negative_log_likelihood(
            trajectory, result.mean_reversion, result.long_run_mean, result.volatility
        ),
    )
    assert result.alpha_bounds == (0.01, 5.0)
    np.testing.assert_array_equal(trajectory.values, original_values)
    np.testing.assert_array_equal(trajectory.times, original_times)


@pytest.mark.parametrize(
    "bounds",
    [(0.0, 1.0), (-1.0, 1.0), (2.0, 1.0), (1.0, 1.0),
     (np.nan, 1.0), (0.1, np.inf), (0.1,), (0.1, 1.0, 2.0)],
)
def test_profile_fit_rejects_invalid_bounds(bounds) -> None:
    trajectory = Trajectory(
        times=np.array([0.0, 1.0, 2.0]),
        values=np.array([14.0, 13.0, 10.0])[:, None, None],
    )
    with pytest.raises(ValueError):
        ou.fit_ou_profile(trajectory, alpha_bounds=bounds)


def test_profile_fit_rejects_degenerate_data() -> None:
    trajectory = Trajectory(
        times=np.array([0.0, 0.3, 2.0]),
        values=np.zeros((3, 1, 1)),
    )
    with pytest.raises(ValueError, match="zero residual variance"):
        ou.fit_ou_profile(trajectory, alpha_bounds=(0.01, 5.0))

@pytest.mark.parametrize(
    "bounds, expected_lower, expected_upper",
    [
        ((0.01, 5.0), False, False),
        ((0.01, 0.3), False, True),
        ((2.0, 5.0), True, False),
    ],
)
def test_profile_fit_boundary_flags(
    bounds,
    expected_lower,
    expected_upper,
) -> None:
    rng = np.random.default_rng(42)

    values = np.empty(401)
    values[0] = -3.0

    for i in range(1, values.size):
        values[i] = (
            0.8 * values[i - 1]
            - 0.6
            + 0.5 * rng.standard_normal()
        )

    trajectory = Trajectory(
        times=np.arange(values.size) * 0.25,
        values=values[:, None, None],
    )

    result = ou.fit_ou_profile(
        trajectory=trajectory,
        alpha_bounds=bounds,
    )

    assert result.success
    assert result.near_lower_bound == expected_lower
    assert result.near_upper_bound == expected_upper
    assert 0.0 <= result.alpha_search_position <= 1.0
    assert result.boundary_fraction == 0.01

def test_initial_alpha_bounds_match_time_scales() -> None:
    trajectory = Trajectory(
        times=np.array([0.0, 0.5, 2.0, 5.0]),
        values=np.zeros((4, 1, 1)),
    )

    lower, upper = ou._initial_ou_alpha_bounds(trajectory)

    np.testing.assert_allclose(
        [lower, upper],
        [0.002, 9.210340371976184],
        rtol=1e-12,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "scale, shift",
    [
        (1.0, 100.0),
        (60.0, 0.0),
        (0.5, 10.0),
    ],
)
def test_initial_alpha_bounds_follow_time_units(
    scale: float,
    shift: float,
) -> None:
    times = np.array([0.0, 0.5, 2.0, 5.0])
    values = np.zeros((4, 1, 1))

    original = Trajectory(
        times=times,
        values=values,
    )
    transformed = Trajectory(
        times=scale * times + shift,
        values=values,
    )

    original_bounds = np.array(
        ou._initial_ou_alpha_bounds(original)
    )
    transformed_bounds = np.array(
        ou._initial_ou_alpha_bounds(transformed)
    )

    np.testing.assert_allclose(
        transformed_bounds,
        original_bounds / scale,
        rtol=1e-12,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "times, shape, message",
    [
        ([0.0], (1, 1, 1), "at least two time points"),
        ([0.0, 1.0], (2, 2, 1), "one node and one feature"),
        ([0.0, 1.0], (2, 1, 2), "one node and one feature"),
        (
            [0.0, 1e-308],
            (2, 1, 1),
            "initial alpha bounds must be finite",
        ),
    ],
)
def test_initial_alpha_bounds_reject_invalid_inputs(
    times,
    shape,
    message,
) -> None:
    trajectory = Trajectory(
        times=np.array(times),
        values=np.zeros(shape),
    )

    with pytest.raises(ValueError, match=message):
        ou._initial_ou_alpha_bounds(trajectory)


def test_profile_scan_rejects_all_failed_evaluations(
    monkeypatch,
) -> None:
    trajectory = Trajectory(
        times=np.array([0.0, 1.0, 2.0]),
        values=np.array([1.0, 4.0, 3.0])[:, None, None],
    )

    def objective(trajectory, alpha):
        raise ValueError("deliberate test failure")

    monkeypatch.setattr(
        ou,
        "_profile_ou_negative_log_likelihood",
        objective,
    )

    with pytest.raises(
        RuntimeError,
        match="all profile grid evaluations failed",
    ):
        ou._scan_ou_profile(
            trajectory,
            alpha_bounds=(0.1, 10.0),
            grid_size=5,
        )

def test_profile_scan_finds_multiple_candidate_intervals(
    monkeypatch,
) -> None:
    trajectory = Trajectory(
        times=np.array([0.0, 1.0, 2.0]),
        values=np.array([1.0, 4.0, 3.0])[:, None, None],
    )

    def objective(trajectory, alpha):
        u = np.log(alpha)
        return (u**2 - 1.0)**2

    monkeypatch.setattr(
        ou,
        "_profile_ou_negative_log_likelihood",
        objective,
    )

    result = ou._scan_ou_profile(
        trajectory,
        alpha_bounds=(np.exp(-2.0), np.exp(2.0)),
        grid_size=5,
    )

    np.testing.assert_allclose(
        np.log(result.alphas),
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        atol=1e-14,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.scores,
        [9.0, 0.0, 1.0, 0.0, 9.0],
        atol=1e-14,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.candidate_intervals,
        [
            [np.exp(-2.0), 1.0],
            [1.0, np.exp(2.0)],
        ],
        atol=1e-14,
        rtol=0.0,
    )

    assert result.best_index in (1, 3)
    assert result.failures == {}


def test_profile_scan_records_failure_without_crossing_it(
    monkeypatch,
) -> None:
    trajectory = Trajectory(
        times=np.array([0.0, 1.0, 2.0]),
        values=np.array([1.0, 4.0, 3.0])[:, None, None],
    )

    def objective(trajectory, alpha):
        u = np.log(alpha)

        if abs(u) < 0.1:
            raise ValueError("deliberate test failure")

        return u**2

    monkeypatch.setattr(
        ou,
        "_profile_ou_negative_log_likelihood",
        objective,
    )

    result = ou._scan_ou_profile(
        trajectory,
        alpha_bounds=(np.exp(-2.0), np.exp(2.0)),
        grid_size=5,
    )

    assert np.isnan(result.scores[2])
    assert result.failures == {2: "deliberate test failure"}
    assert result.best_index in (1, 3)
    assert result.candidate_intervals == []


def test_profile_search_refines_between_grid_points(monkeypatch) -> None:
    trajectory = Trajectory(
        times=np.array([0.0, 1.0, 2.0]),
        values=np.array([1.0, 4.0, 3.0])[:, None, None],
    )

    def objective(trajectory, alpha):
        return (np.log(alpha) - 0.3) ** 2

    monkeypatch.setattr(ou, "_profile_ou_negative_log_likelihood", objective)

    result = ou._search_ou_profile(
        trajectory, (np.exp(-2.0), np.exp(2.0)), grid_size=5
    )

    assert result.success
    assert len(result.local_results) == 1
    assert result.failures == {}
    assert result.fun < result.scan.best_score
    np.testing.assert_allclose(np.log(result.x), 0.3, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(result.fun, 0.0, rtol=0.0, atol=1e-12)


def test_profile_search_selects_best_of_multiple_minima(monkeypatch) -> None:
    trajectory = Trajectory(
        times=np.array([0.0, 1.0, 2.0]),
        values=np.array([1.0, 4.0, 3.0])[:, None, None],
    )

    def objective(trajectory, alpha):
        u = np.log(alpha)
        return min((u + 1.2) ** 2 + 0.5, (u - 1.3) ** 2)

    monkeypatch.setattr(ou, "_profile_ou_negative_log_likelihood", objective)

    result = ou._search_ou_profile(
        trajectory, (np.exp(-2.0), np.exp(2.0)), grid_size=9
    )

    assert result.success
    assert len(result.local_results) == 2
    assert all(local.success for local in result.local_results)
    np.testing.assert_allclose(np.log(result.x), 1.3, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(result.fun, 0.0, rtol=0.0, atol=1e-12)
    assert result.fun <= result.scan.best_score


@pytest.mark.parametrize("failure_mode", ["status", "exception"])
def test_profile_search_preserves_grid_result_on_local_failure(
    monkeypatch, failure_mode,
) -> None:
    from scipy.optimize import OptimizeResult

    trajectory = Trajectory(
        times=np.array([0.0, 1.0, 2.0]),
        values=np.array([1.0, 4.0, 3.0])[:, None, None],
    )

    def objective(trajectory, alpha):
        return (np.log(alpha) - 0.3) ** 2

    def failed_optimizer(**kwargs):
        if failure_mode == "exception":
            raise FloatingPointError("deliberate local failure")
        return OptimizeResult(
            x=0.3, fun=-100.0, success=False,
            message="deliberate local failure",
        )

    monkeypatch.setattr(ou, "_profile_ou_negative_log_likelihood", objective)
    monkeypatch.setattr(ou, "minimize_scalar", failed_optimizer)

    result = ou._search_ou_profile(
        trajectory, (np.exp(-2.0), np.exp(2.0)), grid_size=5
    )

    assert not result.success
    assert result.failures == {0: "deliberate local failure"}
    assert result.x == result.scan.best_alpha
    assert result.fun == result.scan.best_score
    np.testing.assert_allclose(result.x, 1.0)
    np.testing.assert_allclose(result.fun, 0.09)


@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_profile_search_preserves_endpoint_minimum(monkeypatch, direction) -> None:
    trajectory = Trajectory(
        times=np.array([0.0, 1.0, 2.0]),
        values=np.array([1.0, 4.0, 3.0])[:, None, None],
    )

    def objective(trajectory, alpha):
        return direction * np.log(alpha)

    def unexpected_optimizer(**kwargs):
        pytest.fail("monotone profile should not trigger local refinement")

    monkeypatch.setattr(ou, "_profile_ou_negative_log_likelihood", objective)
    monkeypatch.setattr(ou, "minimize_scalar", unexpected_optimizer)

    result = ou._search_ou_profile(
        trajectory, (np.exp(-2.0), np.exp(2.0)), grid_size=5
    )

    assert result.success
    assert result.local_results == []
    assert result.failures == {}
    np.testing.assert_allclose(np.log(result.x), -2.0 * direction)
    np.testing.assert_allclose(result.fun, -2.0)


def test_profile_search_matches_regular_grid_mle() -> None:
    # Independent AR(1) least-squares reference for the real OU objective.
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
    design = np.column_stack((values[:-1], np.ones(values.size - 1)))
    coefficients = np.linalg.lstsq(design, values[1:], rcond=None)[0]
    expected_alpha = -np.log(coefficients[0]) / dt
    residual = values[1:] - design @ coefficients
    q = np.mean(residual**2)
    expected_score = 0.5 * residual.size * (np.log(2.0 * np.pi * q) + 1.0)

    result = ou._search_ou_profile(trajectory, (0.01, 5.0))

    assert result.success
    assert result.fun <= result.scan.best_score
    np.testing.assert_allclose(result.x, expected_alpha, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(result.fun, expected_score, rtol=0.0, atol=1e-7)
