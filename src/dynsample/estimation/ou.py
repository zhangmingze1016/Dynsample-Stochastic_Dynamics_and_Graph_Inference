import numpy as np
from dynsample.core.trajectory import Trajectory
from dynsample.simulation.ou import ou_transition
from scipy.optimize import OptimizeResult, minimize, minimize_scalar
from dynsample.estimation.brownian import fit_brownian_drift

# Compute the conditional OU negative log-likelihood from observed transitions.
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

# Convert log-scale optimization parameters into physical OU parameters before scoring.
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

# Estimate starting values for alpha, mu, and sigma from the observed trajectory.
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

# Compute the conditional maximum-likelihood estimate of mu for a fixed alpha.
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

# Compute the conditional maximum-likelihood estimate of sigma after profiling out mu.
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

# Evaluate the OU negative log-likelihood with mu and sigma fitted at a fixed alpha.
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

# Choose an initial alpha search range from the observation duration and shortest interval.
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

# Fit OU parameters by optimizing log(alpha) within supplied bounds and profiling out mu and sigma.
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

# Compute reference fits and NLL scores for the Brownian and independent Gaussian limits.
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

# Scan a logarithmic alpha grid for candidate minima and record failed evaluations.
def _scan_ou_profile(
    trajectory: Trajectory,
    alpha_bounds: tuple[float, float],
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

    bounds = np.asarray(alpha_bounds, dtype=np.float64)

    if bounds.shape != (2,):
        raise ValueError(
            "alpha_bounds must contain a lower and an upper bound"
        )

    if not np.all(np.isfinite(bounds)):
        raise ValueError("alpha bounds must be finite")

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
                score  = _profile_ou_negative_log_likelihood(
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
        if not np.all(valid[i - 1: i + 2]):
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

# Refine candidate intervals and retain the best finite result from the grid and local searches.
def _search_ou_profile(
    trajectory: Trajectory,
    alpha_bounds: tuple[float, float],
    grid_size: int = 65,
) -> OptimizeResult:
    """Scan and refine the OU profile within supplied bounds."""

    scan = _scan_ou_profile(
        trajectory = trajectory,
        alpha_bounds = alpha_bounds,
        grid_size = grid_size,
    )

    best_alpha = float(scan.best_alpha)
    best_score = float(scan.best_score)

    local_results = []
    failures = {}

    def objective(log_alpha: float) -> float:
        with np.errstate(           
            over="raise",
            divide="raise",
            invalid="raise",
            under="ignore",
        ):
            alpha = float(np.exp(log_alpha))

            score = _profile_ou_negative_log_likelihood(
                trajectory,
                alpha
            )
        if not np.isfinite(score):
            raise ValueError("profile score must be finite")

        return float(score)

    for index, interval in enumerate(scan.candidate_intervals):
        lower, upper = interval

        try:
            result = minimize_scalar(
                fun = objective,
                bounds = (
                    float(np.log(lower)),
                    float(np.log(upper)),
                ),
                method = "bounded",
                options = {"xatol": 1e-8},
            )

            local_results.append(result)

            if not result.success:
                failures[index] = str(result.message)
                continue

            if (
                not np.isfinite(result.x)
                or not np.isfinite(result.fun)
            ):
                failures[index] = (
                    "local optimization produced a non-finite result"
                )
                continue

            alpha = float(np.exp(result.x))

            if not lower <= alpha <= upper:
                failures[index] =(
                    "local optimization returned alpha outside its interval"
                )
                continue

            if result.fun < best_score:
                best_alpha = alpha
                best_score =float(result.fun)

        except (ValueError, FloatingPointError) as error:
            failures[index] = str(error)

    complete = not scan.failures and not failures

    return OptimizeResult(
        x = best_alpha,
        fun = best_score,
        success = complete,
        message=(
            "Grid scan and local refinements completed."
            if complete
            else "Search incomplete; returning the best finite candidate."
        ),
        scan=scan,
        local_results=local_results,
        failures=failures,
    )

# Report proximity to search boundaries, boundary-limit NLL differences, and search completeness.
def _diagnose_ou_search(
    search: OptimizeResult,
    boundary_scores: dict[str, OptimizeResult]
) -> OptimizeResult:
    """Report search position and NLL differences from boundary limits."""

    lower = float(search.scan.alphas[0])
    upper = float(search.scan.alphas[-1])

    alpha = float(search.x)
    score = float(search.fun)

    if (
        not np.all(np.isfinite([lower, upper, alpha, score]))
        or lower <= 0.0
        or upper <= lower
        or not lower <= alpha <= upper
    ):
        raise ValueError(
            "search must contain finite results within valid alpha bounds"
        )

    log_lower = float(np.log(lower))
    log_upper = float(np.log(upper))
    log_alpha = float(np.log(alpha))

    position = (
        (log_alpha - log_lower)
        / (log_upper - log_lower)
    )

    boundary_fraction = 0.01

    near_lower = bool(position <= boundary_fraction)
    near_upper = bool(position >= 1.0 - boundary_fraction)

    brownian_score = float(boundary_scores["brownian"].fun)
    gaussian_score = float(boundary_scores["gaussian"].fun)

    if not np.all(
        np.isfinite([brownian_score, gaussian_score])
    ):
        raise ValueError("boundary scores must be finite")

    with np.errstate(over = "raise", invalid = "raise"):
        brownian_gap = float(
            np.float64(brownian_score) - np.float64(score)
        )
        gaussian_gap = float(
            np.float64(gaussian_score) - np.float64(score)
        )

    return OptimizeResult(
        alpha_bounds=(lower, upper),
        alpha_search_position=float(position),
        boundary_fraction=boundary_fraction,
        near_lower_bound=near_lower,
        near_upper_bound=near_upper,
        brownian_nll_gap=brownian_gap,
        gaussian_nll_gap=gaussian_gap,
        search_complete=bool(search.success),
    )

# Choose and expand alpha bounds automatically, retaining the best candidate and reporting the stopping reason.
def _search_ou_profile_auto(
    trajectory: Trajectory,
    grid_size = 65,
    max_expansions: int = 6,
    nll_tolerance: float = 1e-6,
) -> OptimizeResult:
    """Search automatically and report why the search stopped."""

    if (
        isinstance(max_expansions, bool)
        or not isinstance(max_expansions, (int, np.integer))
        or max_expansions < 0
    ):
        raise ValueError(
            "max_expansions must be a non-negative integer"
        )

    if not np.isfinite(nll_tolerance) or nll_tolerance < 0.0:
        raise ValueError(
            "nll_tolerance must be finite and non-negative"
        )

    lower, upper = _initial_ou_alpha_bounds(trajectory)
    boundary_scores = _ou_boundary_scores(trajectory)

    history = []
    best_search = None

    for expansion in range(max_expansions + 1):
        search = _search_ou_profile(
            trajectory= trajectory,
            alpha_bounds = (lower, upper),
            grid_size = grid_size,
        )

        diagnostics = _diagnose_ou_search(
            search = search,
            boundary_scores = boundary_scores,
        )

        history.append(
            OptimizeResult(
                search = search,
                diagnostics = diagnostics
            )
        )

        if best_search is None or search.fun < best_search.fun:
            best_search = search

        if not search.success:
            status = "incomplete_search"
            break

        lower_better = (
            diagnostics.brownian_nll_gap < -nll_tolerance
        )
        upper_better = (
            diagnostics.gaussian_nll_gap < -nll_tolerance
        )

        lower_close = (
            diagnostics.near_lower_bound
            and abs(diagnostics.brownian_nll_gap) <= nll_tolerance
        )
        upper_close = (
            diagnostics.near_upper_bound
            and abs(diagnostics.gaussian_nll_gap) <= nll_tolerance
        )

        if (
            (lower_close or upper_close)
            and not lower_better
            and not upper_better
        ):
            status = "boundary_limit"
            break

        expand_lower = (
            diagnostics.near_lower_bound or lower_better
        )
        expand_upper = (
            diagnostics.near_upper_bound or upper_better
        )

        if not expand_lower and not expand_upper:
            status = "interior_candidate"
            break

        if expansion == max_expansions:
            status = "expansion_limit"
            break

        new_lower = lower / 10.0 if expand_lower else lower
        new_upper = upper * 10.0 if expand_upper else upper

        if (
            not np.all(np.isfinite([new_lower, new_upper]))
            or new_lower <= 0.0
        ):
            status = "numerical_limit"
            break

        lower, upper = new_lower, new_upper

    return OptimizeResult(
        x=float(best_search.x),
        fun=float(best_search.fun),
        success=(status == "interior_candidate"),
        status=status,
        best_search=best_search,
        boundary_scores=boundary_scores,
        history=history,
        expansions=len(history) - 1,
    )

# Fit alpha, mu, and sigma jointly using bounded optimization with log-scale alpha and sigma.
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

