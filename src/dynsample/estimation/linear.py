import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm, solve_triangular
from scipy.optimize import OptimizeResult, minimize

from dynsample.core.linear_model import LinearSDE
from dynsample.core.trajectory import Trajectory
from dynsample.simulation.linear import linear_transition

# Score complete observations using the condition Gaussian likelihood

def linear_negative_log_likelihood(
        trajectory: Trajectory,
        model: LinearSDE,
) -> float:
    """Return the negative log-likelihood conditional on the first state."""

    if trajectory.n_steps < 2:
        raise ValueError("trajectory must contain at least two time steps")

    if trajectory.values.shape[1:] != (model.n_nodes, 1):
        raise ValueError(
            "trajectory must contain matching nodes and one feature per node"
        )

    with np.errstate(over = "ignore", invalid = "ignore"):
        intervals = np.diff(trajectory.times)

    if not np.all(np.isfinite(intervals)):
        raise ValueError("time intervals must be finite")

    if np.any(intervals <= 0):
        raise ValueError("time must be strictly increasing")

    values = trajectory.values[:, :, 0]
    normalizing_constant = model.n_nodes * np.log(2.0 * np.pi)
    total = 0.0

    for i, dt in enumerate(intervals, start = 1):
        transition, offset, covariance = linear_transition(
            model = model,
            dt = float(dt)
        )

        with np.errstate(over = "ignore", invalid = "ignore"):
            mean = transition @ values[i - 1] + offset
            residual = values[i] - mean

        if not np.all(np.isfinite(residual)):
            raise FloatingPointError(
                f"non-finite residual at transition {i}")

        try:
            lower = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"transition {i} requires a positive-definite covariance"
            ) from exc

        standardized_residual = solve_triangular(
            lower,
            residual,
            lower = True
        )

        with np.errstate(over="ignore", invalid="ignore"):
            log_determinant = 2.0 * np.sum(
                np.log(np.diag(lower))
            )

            quadratic = float(
                standardized_residual @ standardized_residual
            )

            contribution = 0.5 * (
                normalizing_constant
                + log_determinant
                + quadratic
            )

            total = total + float(contribution)

            if not np.isfinite(total):
                raise FloatingPointError(
                    f"non-finite likelihood at transition {i}"
            )

    return float(total)

# Estimate the constant offset by weighted least squares with fixed K and B.
def fit_linear_offset(
    trajectory: Trajectory,
    drift: NDArray[np.float64],
    diffusion: NDArray[np.float64],
) -> OptimizeResult:
    """Fit b conditional on the initial state, with K and B fixed."""

    n = trajectory.n_nodes

    # Validate the supplied coefficients using the existing model class.
    model = LinearSDE(
        drift=drift,
        offset=np.zeros(n, dtype=np.float64),
        diffusion=diffusion,
    )

    if trajectory.n_steps < 2:
        raise ValueError(
            "trajectory must contain at least two time steps"
        )

    if trajectory.values.shape[1:] != (model.n_nodes, 1):
        raise ValueError(
            "trajectory must contain matching nodes "
            "and one feature per node"
        )

    with np.errstate(over="ignore", invalid="ignore"):
        intervals = np.diff(trajectory.times)

    if not np.all(np.isfinite(intervals)):
        raise ValueError("time intervals must be finite")

    if np.any(intervals <= 0):
        raise ValueError("time must be strictly increasing")

    values = trajectory.values[:, :, 0]

    # The upper-right block of its exponential gives the integral of exp(Ku).
    integral_block = np.zeros((2 * n, 2 * n), dtype=np.float64)
    integral_block[:n, :n] = model.drift
    integral_block[:n, n:] = np.eye(n, dtype=np.float64)

    design_blocks = []
    response_blocks = []

    for i, dt in enumerate(intervals, start=1):
        transition, _, covariance = linear_transition(
            model=model,
            dt=float(dt),
        )

        with np.errstate(over="ignore", invalid="ignore"):
            scaled_block = integral_block * dt

        if not np.all(np.isfinite(scaled_block)):
            raise FloatingPointError(
                f"non-finite offset integration block at transition {i}"
            )

        with np.errstate(over="ignore", invalid="ignore"):
            block_exponential = expm(scaled_block)

        if not np.all(np.isfinite(block_exponential)):
            raise FloatingPointError(
                f"non-finite offset integral at transition {i}"
            )

        integrated_transition = block_exponential[:n, n:]

        with np.errstate(over="ignore", invalid="ignore"):
            response = values[i] - transition @ values[i - 1]

        if not np.all(np.isfinite(response)):
            raise FloatingPointError(
                f"non-finite response at transition {i}"
            )

        try:
            lower = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"transition {i} requires a positive-definite covariance"
            ) from exc

        # Whiten the equation so ordinary least squares has the correct weights.
        whitened_design = solve_triangular(
            lower,
            integrated_transition,
            lower=True,
        )
        whitened_response = solve_triangular(
            lower,
            response,
            lower=True,
        )

        if (
            not np.all(np.isfinite(whitened_design))
            or not np.all(np.isfinite(whitened_response))
        ):
            raise FloatingPointError(
                f"non-finite whitened equation at transition {i}"
            )

        design_blocks.append(whitened_design)
        response_blocks.append(whitened_response)

    design = np.vstack(design_blocks)
    response = np.concatenate(response_blocks)

    offset, _, rank, singular_values = np.linalg.lstsq(
        design,
        response,
        rcond=None,
    )

    if rank < n:
        raise ValueError(
            "offset is not uniquely identifiable: "
            "the weighted design matrix is rank deficient"
        )

    if not np.all(np.isfinite(offset)):
        raise FloatingPointError("estimated offset must be finite")

    fitted_model = LinearSDE(
        drift=model.drift,
        offset=offset,
        diffusion=model.diffusion,
    )

    score = linear_negative_log_likelihood(
        trajectory=trajectory,
        model=fitted_model,
    )

    return OptimizeResult(
        x=offset.copy(),
        offset=offset.copy(),
        model=fitted_model,
        fun=score,
        success=True,
        message="Unique weighted least-squares offset solution found.",
        rank=int(rank),
        singular_values=singular_values.copy(),
    )

# Fit permitted drift entries while profiling out the constant offset.
def fit_linear_drift(
        trajectory: Trajectory,
        diffusion: NDArray[np.float64],
        drift_mask: NDArray[np.bool_] | None = None,
        initial_drift: NDArray[np.float64] | None = None,
        maxiter: int = 300,
) -> OptimizeResult:
    """Fit K and b with fixed B and optional drift restrictions."""
    n = trajectory.n_nodes
    # Validate the mask without silently converting numbers to booleans.
    if drift_mask is None:
        mask = np.ones((n, n), dtype=bool)
    else:
        mask = np.asarray(drift_mask)

    
    if mask.shape != (n, n):
        raise ValueError("drift_mask must have shape (n_nodes, n_nodes)")

    if mask.dtype != np.dtype(bool):
        raise ValueError("drift_mask must contain boolean values")

    mask = mask.copy()

    if (
        isinstance(maxiter, (bool, np.bool_))
        or not isinstance(maxiter, (int, np.integer))
        or maxiter <= 0
    ):
        raise ValueError("maxiter must be a positive integer")
 
    if trajectory.n_steps < 2:
        raise ValueError("trajectory must contain at least two time steps")

    if trajectory.n_features != 1:
        raise ValueError("trajectory must contain one feature per node")

    # Construct a default starting point, or validate the supplied one.
    if initial_drift is None:
        with np.errstate(over = "ignore", invalid = "ignore"):
            duration = trajectory.times[-1] - trajectory.times[0]

        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("time span must be finite and positive")

        with np.errstate(over = "ignore", divide = "ignore", invalid = "ignore"):
            initial = -np.eye(n, dtype = np.float64) / duration

        initial[~mask] = 0.0

    else:
        initial = np.array(
            initial_drift,
            dtype = np.float64,
            copy = True,
        )


    if initial.shape != (n, n):
        raise ValueError("initial_drift must have shape (n_nodes, n_nodes)")

    if not np.all(np.isfinite(initial)):
        raise ValueError("initial_drift must contain only finite values")

    if np.any(initial[~mask] != 0.0):
        raise ValueError("initial_drift must be zero outside drift_mask")

    # Validate the initial model and obtain its profiled likelihood.
    initial_fit = fit_linear_offset(
        trajectory = trajectory,
        drift = initial,
        diffusion = diffusion,
    )

    fixed_diffusion = initial_fit.model.diffusion.copy()
    initial_parameters = initial[mask].copy()
    invalid_evaluations = 0
    last_invalid_reason = None

    # Rebuild K from the free parameters and solve for its optimal b.
    def evaluate(parameters):
        drift = np.zeros((n, n), dtype = np.float64)
        drift[mask] = parameters

        return fit_linear_offset(
            trajectory = trajectory,
            drift = drift,
            diffusion = fixed_diffusion
        )
    
    # Reject unusable trial models while keeping diagnostics.
    def objective(parameters):
        nonlocal invalid_evaluations, last_invalid_reason

        try:
            fit = evaluate(parameters)
            return float(fit.fun)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            invalid_evaluations += 1
            last_invalid_reason = str(exc)
            return np.inf

        # An all-False mask fixes K at zero; only b needs fitting.
    if initial_parameters.size == 0:
        fitted = initial_fit
        success = True
        message = (
            "No free drift entries; "
            "fitted the offset with K fixed at zero."
        )
        optimizer_result = None
    else:
        optimizer_result = minimize(
            objective,
            x0=initial_parameters,
            method="Powell",
            options={
                "maxiter": int(maxiter),
                "xtol": 1e-6,
                "ftol": 1e-8,
            },
        )

        success = bool(optimizer_result.success)
        message = str(optimizer_result.message)

        # Verify the returned candidate rather than trusting success alone.
        try:
            fitted = evaluate(optimizer_result.x)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            fitted = initial_fit
            success = False
            message = (
                "Final candidate was unusable; returned the initial fit. "
                + str(exc)
            )

        if fitted.fun > initial_fit.fun:
            fitted = initial_fit
            success = False
            message = (
                "Final candidate was worse than the initial fit; "
                "returned the initial fit."
            )

    # Both branches above must reach these diagnostics and the return.
    eigenvalues = np.linalg.eigvals(fitted.model.drift)
    spectral_abscissa = float(np.max(eigenvalues.real))

    return OptimizeResult(
        x=fitted.model.drift[mask].copy(),
        drift=fitted.model.drift.copy(),
        offset=fitted.offset.copy(),
        diffusion=fixed_diffusion.copy(),
        model=fitted.model,
        fun=float(fitted.fun),
        initial_fun=float(initial_fit.fun),
        success=success,
        message=message,
        drift_mask=mask.copy(),
        spectral_abscissa=spectral_abscissa,
        is_stable=bool(spectral_abscissa < 0.0),
        invalid_evaluations=invalid_evaluations,
        last_invalid_reason=last_invalid_reason,
        optimizer_result=optimizer_result,
    )