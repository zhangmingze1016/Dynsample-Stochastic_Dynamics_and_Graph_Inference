import numpy as np
from dynsample.core.trajectory import Trajectory
from dynsample.simulation.ou import ou_transition
from scipy.optimize import OptimizeResult, minimize, minimize_scalar
from dynsample.estimation.brownian import fit_brownian_drift

def ou_negative_log_likelihood(
        trajectory: Trajectory,
        mean_reversion: float,
        long_run_mean: float,
        volatility: float,
) -> float:
    if trajectory.values.shape[1:] != (1, 1):
        raise ValueError(
            "trajectory must contain one node and one feature"
        )

    if trajectory.times.size < 2:
        raise ValueError(
            "trajectory must contain at least two time points"
        )

    parameters = (mean_reversion, long_run_mean, volatility)
    if not np.all(np.isfinite(parameters)):
        raise ValueError("all parameters must be finite")

    if mean_reversion <= 0.0:
        raise ValueError("mean_reversion must be positive")

    if volatility <= 0.0:
        raise ValueError("volatility must be positive")

    times = trajectory.times
    values = trajectory.values[:, 0, 0]
    total = 0.0

    for i in range(1, times.size):
        dt = times[i] - times[i - 1]

        transition, offset, variance = ou_transition(
            dt = dt,
            mean_reversion = mean_reversion,
            long_run_mean = long_run_mean,
            volatility = volatility,
        )

        if not np.isfinite(variance) or variance <= 0.0:
            raise ValueError("transition variance must be finite and positive")

        mean = transition * values[i - 1] + offset
        residual = values[i] - mean

        total += 0.5 * (
            np.log(2.0 * np.pi)
            + np.log(variance)
            + residual**2 / variance
        )

    return float(total)

def _ou_objective(
        parameters: np.ndarray,
        trajectory: Trajectory,
) -> float:
    log_alpha, mu, log_sigma = parameters

    alpha = float(np.exp(log_alpha))
    sigma = float(np.exp(log_sigma))


    return ou_negative_log_likelihood(
        trajectory = trajectory,
        mean_reversion = alpha,
        long_run_mean = mu,
        volatility = sigma,
    )

def _initial_ou_parameters(
        trajectory: Trajectory,
) -> tuple[float, float, float]:
    if trajectory.values.shape[1:] != (1,1):
        raise ValueError(
            "trajectory must contain one node and one feature"
        )

    if trajectory.times.size < 2:
        raise ValueError(
            "trajectory must contain at least two time points"
        )

    times = trajectory.times
    values = trajectory.values[:, 0, 0]

    dt = np.diff(times)
    increments = np.diff(values)
    duration = times[-1] - times[0]

    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("time span must be finite and positive")

    if np.all(increments == 0.0):
        raise ValueError(
            "cannot initialize positive noise from a constant trajectory"
        )

    alpha = 1.0 / duration
    mu = float(np.mean(values))

    scaled_increments = increments / np.sqrt(dt)
    sigma = float(np.sqrt(np.mean(scaled_increments**2)))

    parameters = (float(alpha), mu, sigma)

    if not np.all(np.isfinite(parameters)):
        raise ValueError("initial parameter calculation produced non-finite values")

    if alpha <= 0.0 or sigma <= 0.0:
        raise ValueError("initial alpha and sigma must be positive")

    return parameters

def _profile_ou_mu(
        trajectory: Trajectory,
        mean_reversion: float,
) -> float:
    if trajectory.values.shape[1:] != (1, 1):
        raise ValueError(
            "trajectory must contain one node and one feature"
        )

    if trajectory.times.size < 2:
        raise ValueError(
            "trajectory must contain at least two time points"
        )

    if not np.isfinite(mean_reversion) or mean_reversion <= 0.0:
        raise ValueError(
            "mean_reversion must be finite and positive"
        )

    times = trajectory.times
    values = trajectory.values[:, 0, 0]
    dt = np.diff(times)

    transition = np.exp(-mean_reversion * dt)
    coefficient = -np.expm1(-mean_reversion * dt)
    variance_scale = (
        -np.expm1(-2.0 * mean_reversion * dt)
        / (2.0 * mean_reversion)
    )

    if (
        not np.all(np.isfinite(variance_scale))
        or np.any(variance_scale <= 0.0)
    ):
        raise ValueError(
            "variance scales must be finite and positive"
        )

    adjusted_values = values[1:] - transition * values[:-1]

    numerator = np.sum(
        coefficient * adjusted_values / variance_scale
    )

    denominator = np.sum(
        coefficient**2 / variance_scale
    )

    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError(
            "cannot determine the conditional optimum for mu"
        )

    mu = float(numerator / denominator)

    if not np.isfinite(mu):
        raise ValueError(
            "mu calculation produced a non-finite value"
        )

    return mu

def _profile_ou_sigma(
        trajectory: Trajectory,
        mean_reversion: float,
) -> float:
    mu = _profile_ou_mu(
        trajectory = trajectory,
        mean_reversion = mean_reversion
    )

    times = trajectory.times
    values = trajectory.values[:, 0, 0]
    scaled_squared_residuals = np.empty(times.size - 1)

    for i in range(1, times.size):
        dt = times[i] - times[i - 1]

        transition, offset, variance_scale = ou_transition(
            dt = dt,
            mean_reversion = mean_reversion,
            long_run_mean = mu,
            volatility = 1.0,
        )

        if (
            not np.isfinite(variance_scale)
            or variance_scale <= 0.0
        ):
            raise ValueError(
                "variance scales must be finite and positive"
            )

        conditional_mean = transition * values[i - 1] + offset
        residual = values[i] - conditional_mean

        scaled_squared_residuals[i - 1] = (
            residual / np.sqrt(variance_scale)
        ) ** 2

    sigma_squared = float(np.mean(scaled_squared_residuals))

    if not np.isfinite(sigma_squared):
        raise ValueError(
            "sigma squared calculation produced a non-finite value"
        )

    if sigma_squared <= 0.0:
        raise ValueError(
            "cannot estimate positive volatility from zero residual variance"
        )

    return float(np.sqrt(sigma_squared))  

def _profile_ou_negative_log_likelihood(
        trajectory: Trajectory,
        mean_reversion: float,
) -> float:
    mu = _profile_ou_mu(
        trajectory = trajectory,
        mean_reversion = mean_reversion,
    )

    sigma = _profile_ou_sigma(
        trajectory = trajectory,
        mean_reversion = mean_reversion,
    )

    score = ou_negative_log_likelihood(
        trajectory = trajectory,
        mean_reversion = mean_reversion,
        long_run_mean = mu,
        volatility = sigma,
    )

    if not np.isfinite(score):
        raise ValueError(
            "profile negative log-likelihood must be finite"
        )

    return float(score)  

def _initial_ou_alpha_bounds(
    trajectory: Trajectory,
) -> tuple[float, float]:
    """Choose an initial alpha search range from observation times."""
    if trajectory.values.shape[1:] != (1, 1):
        raise ValueError(
            "trajectory must contain one node and one feature"
        )

    times = trajectory.times

    if times.size < 2:
        raise ValueError(
            "trajectory must contain at least two time points"
        )

    with np.errstate(over = "ignore", invalid = "ignore"):
        dt = np.diff(times)
        duration = times[-1] - times[0]

    if not np.all(np.isfinite(dt)) or np.any(dt <= 0.0):
        raise ValueError(
            "time intervals must be finite and positive"
        )

    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError(
            "time span must be finite and positive"
        )

    min_dt = np.min(dt)

    slow_reversion_scale = 0.01
    fast_memory_fraction = 0.01

    with np.errstate(over = "ignore", under = "ignore", invalid = "ignore"):
        lower = slow_reversion_scale / duration
        upper = -np.log(fast_memory_fraction) / min_dt

    if not np.all(np.isfinite([lower, upper])):
        raise ValueError(
            "initial alpha bounds must be finite"
        )

    if lower <= 0.0 or upper <= lower:
        raise ValueError(
            "initial alpha bounds must satisfy 0 < lower < upper"
        )

    return float(lower), float(upper)

def fit_ou_profile(
    trajectory: Trajectory,
    alpha_bounds: tuple[float, float]
) -> OptimizeResult:
    bounds = np.asanyarray(alpha_bounds, dtype = np.float64)

    if bounds.shape != (2,):
        raise ValueError(
            "alpha_bounds must contain a lower and an upper bound"
        )

    if not np.all(np.isfinite(bounds)):
        raise ValueError(
            "alpha bounds must be finite"
        )

    lower, upper = bounds

    if lower <= 0.0 or upper <= lower:
        raise ValueError(
            "alpha bounds must satisfy 0 < lower < upper"
        )

    log_lower = float(np.log(lower))
    log_upper = float(np.log(upper))

    def objective(log_alpha: float) -> float:
        alpha = float(np.exp(log_alpha))

        return _profile_ou_negative_log_likelihood(
            trajectory =trajectory,
            mean_reversion = alpha,
        )

    result = minimize_scalar(
        fun = objective,
        bounds = (log_lower, log_upper),
        method = "bounded",
        options = {"xatol": 1e-8},
    )

    if not result.success:
        raise RuntimeError(
            f"profile optimization failed: {result.message}"
        )

    if not np.isfinite(result.x) or not np.isfinite(result.fun):
        raise RuntimeError(
            "profile optimization produced a non-finite result"
        )

    alpha = float(np.exp(result.x))

    mu = _profile_ou_mu(
        trajectory = trajectory,
        mean_reversion = alpha,
    )

    sigma = _profile_ou_sigma(
        trajectory = trajectory,
        mean_reversion = alpha,
    )

    result.log_alpha = float(result.x)
    result.x = alpha
    result.mean_reversion = alpha
    result.long_run_mean = mu
    result.volatility =sigma
    result.alpha_bounds = (float(lower), float(upper))

    log_width = log_upper - log_lower

    relative_position = (
        result.log_alpha - log_lower
    ) / log_width

    boundary_fraction = 0.01

    result.alpha_search_position = float(relative_position)
    result.boundary_fraction =boundary_fraction

    result.near_lower_bound = bool(
        relative_position <= boundary_fraction
    )

    result.near_upper_bound = bool(
        relative_position >= 1.0 -boundary_fraction
    )

    return result

def _ou_boundary_scores(
    trajectory: Trajectory,
) -> dict[str, OptimizeResult]:

    brownian = fit_brownian_drift(trajectory)

    targets = trajectory.values[1:, 0, 0]
    n = targets.size

    with np.errstate(over = "ignore", invalid = "ignore"):
        mean =float(np.mean(targets))
        variance = float(np.mean((targets - mean) ** 2))

    if not np.isfinite(mean):
        raise ValueError(
            "Gaussian boundary mean must be finite"
        )

    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError(
            "Gaussian boundary variance must be finite and positive"
        )

    score = float(
        0.5 * n * (
            np.log(2.0 * np.pi)
            + np.log(variance)
            + 1.0
        )
    )

    if not np.isfinite(score):
        raise ValueError(
            "Gaussian boundary negative log-likelihood must be finite"
        )

    gaussian = OptimizeResult(
        mean = mean,
        variance = variance,
        fun = score,
        success = True,
        message = "Closed-form independent Gaussian boundary estimate.",
    )

    return {
        "brownian": brownian,
        "gaussian": gaussian
    }

def _scan_ou_profile(
    trajectory: Trajectory,
    alpha_bounds = tuple[float, float],
    grid_size: int = 65,
) -> OptimizeResult:
    """Scan the OU profile on a logarithmic alpha grid."""

    if trajectory.values.shape[1:] != (1, 1):
        raise ValueError(
            "trajectory must contain one node and one feature"
        )

    if trajectory.times.size < 3:
        raise ValueError(
            "trajectory must contain at least three time points"
        )

    bounds = np.asanyarray(alpha_bounds, dtype = np.float64)

    lower, upper = bounds

    if lower <= 0.0 or upper <= lower:
        raise ValueError(
            "alpha bounds must satisfy 0 < lower < upper"
        )

    if (
        isinstance(grid_size, bool)
        or not isinstance(grid_size, (int, np.integer))
        or grid_size < 3
    ):
        raise ValueError(
            "grid_size must be an integer of at least three"
        )

    log_alphas = np.linspace(
        np.log(lower),
        np.log(upper),
        grid_size,
    )
    alphas = np.exp(log_alphas)
    alphas[0], alphas[-1] = lower, upper

    scores = np.full(grid_size, np.nan)
    failures = {}

    for i, alpha in enumerate(alphas):
        try:
            with np.errstate(
                over = "raise",
                divide = "raise",
                invalid= "raise",
                under = "ignore"
            ):
                score = ou_score = _profile_ou_negative_log_likelihood(
                    trajectory,
                    float(alpha)
                )

            if not np.isfinite(score):
                raise ValueError(
                    "profile score must be finite"
                )

            scores[i] = score

        except (ValueError, FloatingPointError) as error:
            failures[i] = str(error)

    valid = np.isfinite(scores)
    valid_indices = np.flatnonzero(valid)

    if valid_indices.size == 0:
        raise RuntimeError(
            "all profile grid evaluations failed; "
            f"first failure: {failures[0]}"
        )

    best_index = int(
        valid_indices[np.argmin(scores[valid_indices])]
    )

    candidate_intervals = []

    for i in range(1, grid_size - 1):
        if not np.all(valid[i - 1: i + 1]):
            continue
        
        left_score, center_score, right_score = scores[i - 1:i + 2]

        if (
            center_score <= left_score
            and center_score <= right_score
            and (
                center_score < left_score
                or center_score < right_score
            )
        ):
            candidate_intervals.append(
                (
                    float(alphas[i - 1]),
                    float(alphas[i + 1]),
                )
            )

    return OptimizeResult(
        alphas = alphas,
        scores = scores,
        valid = valid,
        failures = failures,
        best_index = best_index,
        best_alpha = float(alphas[best_index]),
        best_score = float(scores[best_index]),
        candidate_intervals = candidate_intervals,
        complete = bool(np.all(valid)),
        nfev = int(grid_size),
    )

def fit_ou(
        
        trajectory: Trajectory,
        initial_parameters: tuple[float, float, float] | None,
        parameter_bounds: tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ],
) -> OptimizeResult:
    if initial_parameters is None:
        initial_parameters = _initial_ou_parameters(trajectory)
    initial = np.asarray(
        initial_parameters,
        dtype = np.float64,
    )
    
    bounds = np.asarray(parameter_bounds, dtype = np.float64)

    if initial.shape != (3,) or bounds.shape != (3, 2):
            raise ValueError(
                "provide three initial values and three bounds pairs"
            )

    if not np.all(np.isfinite(initial)) or not np.all(np.isfinite(bounds)):
        raise ValueError("initial values and bounds must be finite")

    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("each lower bound must be smaller than its upper bound")

    if bounds[0, 0] <= 0.0 or bounds[2, 0] <= 0.0:
        raise ValueError("alpha and sigma bounds must be positive")

    if np.any(initial < bounds[:, 0]) or np.any(initial > bounds[:, 1]):
        raise ValueError("initial values must lie within the bounds")

    
    initial_score = ou_negative_log_likelihood(
        trajectory = trajectory,
        mean_reversion = float(initial[0]),
        long_run_mean = float(initial[1]),
        volatility = float(initial[2]),
    )

    if not np.isfinite(initial_score):
        raise ValueError("initial negative log-likelihood must be finite")

    search_initial = initial.copy()
    search_initial[[0, 2]] = np.log(initial[[0, 2]])

    search_bounds = bounds.copy()
    search_bounds[[0, 2], :] = np.log(bounds[[0, 2], :])

    result = minimize(
        fun = _ou_objective,
        x0 = search_initial,
        args = (trajectory,),
        method = "L-BFGS-B",
        bounds = search_bounds,
    )

    return result

