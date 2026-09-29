# Dynsample: Stochastic Dynamics and Graph Inference
A Python research project for learning dynamic relationships between nodes from finite time-series observations, with explicit assumptions and reliability evaluation.

The intended workflow is:

```text
Node time series + timestamps
    -> Estimate dynamical relationships
    -> Track changes in the graph
    -> Assess uncertainty, stability, and predictive value
```

**Current status:** the repository implements core data structures, independent Brownian and Ornstein–Uhlenbeck (OU) simulation, Brownian bridge sampling, and scalar OU parameter fitting. Unknown graph estimation, dynamic graph inference, and the reliability workflow below are planned, not implemented.

## Objective and Scope

The central question is:

> Which dynamical relationships between nodes are supported by the observations, how do they change over time, and how reliable are those conclusions?

The objective is a reusable tool for scientific research and engineering analysis. Complete observations are the starting point; support for noisy, asynchronous, and partially missing observations extends the same workflow. Missing-value reconstruction is a supporting capability, not the primary product.

Initial scope:

- fixed, known node identities;
- one or more state variables per node;
- small systems, initially targeting tens of nodes, subject to measured performance;
- linear stochastic dynamics with sparse, initially static and later piecewise-changing relationships;
- externally supplied timestamps, with native continuous-time transitions;
- explicit graph semantics, simple baselines, and reproducible evaluation.

No particular industry application is required. Synthetic systems provide known graph ground truth; suitable public-data examples will test usability and model limitations. An application example does not establish validity across an entire field.

The project does not currently promise causal discovery, inference of human intentions, automatic team or community discovery, arbitrary nonlinear dynamics, or a general-purpose graph-learning framework.

## Current Implementation

| Component | Implemented behavior |
| --- | --- |
| `State` | A time and a state array of shape `(N, d)`. |
| `Trajectory` | Strictly increasing finite times and state arrays of shape `(T, N, d)`. |
| `Observation` | A time, values, and a Boolean observation mask of shape `(N, d)`. |
| `Graph` | Weighted adjacency, node count, row-sum degree, degree matrix, and `D - adjacency`. |
| Brownian simulation | Independent node-feature increments with one shared scalar volatility; arbitrary strictly increasing requested times. |
| OU simulation | Exact scalar transitions, single-step and trajectory simulation on irregular times; shared scalar parameters and independent node-feature noise. |
| OU estimation | Conditional Gaussian likelihood, bounded joint fitting with automatic initialization, and scalar profile fitting with supplied alpha bounds. |
| Brownian bridge | Single-point and joint multi-point conditional sampling between two supplied endpoints. |
| Validation | Unit tests and empirical checks of Brownian bridge means, variances, and multi-point covariance. |

`Graph` stores supplied relationships; it does not learn them. An observation mask can represent missing entries, but the current bridge functions do not perform general masked-data inference. The `estimation` package currently supports scalar OU fitting; `metrics` and `sampling` remain placeholders.

Remaining correctness and packaging verification work is listed under [Immediate Development Work](#immediate-development-work). Passing the existing tests does not resolve those gaps.

## Local Setup and Working Example

Use Python 3.10 or newer (the development environment has been verified with Python 3.12). Run these commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` installs the project in editable mode with the `dev` and `plot` extras. `pyproject.toml` declares NumPy and SciPy as core dependencies, pytest for development, and Matplotlib for plotting. Clean-environment installation verification remains a release check.

The following example uses implemented functions only:

```python
import numpy as np

from dynsample.core.state import State
from dynsample.inference.reconstruction.brownian_bridge import brownian_bridge
from dynsample.simulation.brownian import simulate_brownian

trajectory = simulate_brownian(
    initial_state=State(time=0.0, values=np.zeros((3, 1))),
    times=np.array([0.0, 0.2, 0.7, 1.0]),
    volatility=0.5,
    rng=np.random.default_rng(42),
)

sample = brownian_bridge(
    left_state=trajectory.state_at(0),
    right_state=trajectory.state_at(-1),
    times=np.array([0.25, 0.5, 0.75]),
    volatility=0.5,
    rng=np.random.default_rng(43),
)

print(sample.values.shape)  # (3, 3, 1): time, node, feature
```

This draws a conditional path using only the two supplied endpoints. It does not estimate a graph, fit volatility, or recover the actual hidden path.

After installation, run:

```bash
python -m pytest -q
python experiments/experiment_brownian_bridge.py
python experiments/experiment_ou.py
python experiments/experiment_ou_estimation.py
```

The Brownian bridge experiment prints the bridge construction and displays a plot. The OU experiment uses irregular sampling times and displays a sampled trajectory, its conditional mean, and pointwise 95% conditional intervals given the initial state and known parameters. These intervals describe process noise, not parameter-estimation uncertainty or simultaneous path coverage. Lines between sampled states are display connections, not reconstructed intermediate paths. Experiment entry points and optional verbose output remain cleanup tasks.

## Scalar OU Fitting and Experiment

The current estimator assumes a complete trajectory of shape `(T, 1, 1)`, exact observations without measurement noise, and constant scalar OU parameters. It conditions on the first observation and supports unequal time intervals. It does not infer inter-node relationships or reconstruct missing states.

| Function in `dynsample.estimation.ou` | Purpose |
| --- | --- |
| `ou_negative_log_likelihood` | Score supplied positive `mean_reversion` and `volatility`, with a finite `long_run_mean`, against the observed transitions. |
| `fit_ou` | Minimize that score using L-BFGS-B in `(log(alpha), mu, log(sigma))` coordinates. |
| `fit_ou_profile` | Search over `log(alpha)` within supplied alpha bounds, analytically optimizing mu and sigma for each candidate alpha. |
| `_initial_ou_parameters` | Internal heuristic starting values; not a parameter estimate or a public API guarantee. |
| `_ou_objective` | Internal conversion from optimizer coordinates to model parameters. |

`fit_ou` currently requires `trajectory`, `initial_parameters`, and `parameter_bounds`. Pass `initial_parameters=None` to generate a starting guess automatically. Bounds are three finite `(lower, upper)` pairs in `(alpha, mu, sigma)` order; alpha and sigma bounds must be positive. Automatic starts must lie within the supplied bounds. A constant trajectory is rejected by the initializer. Optional bounds, adaptive search, and automatic multiple starts are not implemented yet.

The SciPy `OptimizeResult` returned by `fit_ou` contains:

- `x`: optimizer coordinates, **not** physical OU parameters; recover alpha with `exp(x[0])`, mu with `x[1]`, and sigma with `exp(x[2])`;
- `fun`: final conditional negative log-likelihood;
- `success` and `message`: numerical stopping status, not guarantees of a global optimum or accurate parameter recovery.

Run `python experiments/experiment_ou_estimation.py` after installation. The experiment generates 1,001 observations over 100 time units with alternating intervals of 0.05 and 0.15. It reports true, initial, and fitted parameters; initial and final scores; optimizer status; and proximity to its explicitly supplied search bounds.

The figure has three panels:

1. Observations and fitted one-step conditional means, each using the previous observed state.
2. In-sample standardized residuals: observation minus its fitted one-step mean, divided by its transition standard deviation.
3. Five new trajectories under fixed fitted parameters, plus the conditional mean given only the initial state.

Close agreement of one-step means with densely sampled observations is not a long-horizon forecast validation. Residual variance near one is partly enforced by fitting the noise scale and is not independent evidence of model adequacy. New simulated paths are not reconstructions of the original path and do not include parameter uncertainty.

Tests include a hand-calculated irregular-time likelihood, invalid inputs, coordinate conversion, automatic initialization, and comparison with an independent regular-grid conditional MLE obtained through AR(1) least squares. Repeated-seed recovery experiments, held-out prediction checks, and sensitivity to search settings remain future validation work. R1 graph estimation is not yet complete.

### Profile likelihood fitting

For fixed positive alpha, the internal helpers `_profile_ou_mu` and `_profile_ou_sigma` calculate the conditional maximum-likelihood mu and sigma. `_profile_ou_negative_log_likelihood` scores those parameters. This eliminates two numerical search dimensions; it does not change the conditional likelihood or add Bayesian inference.

```python
from dynsample.estimation.ou import fit_ou_profile

# trajectory is an existing complete scalar Trajectory.
result = fit_ou_profile(trajectory, alpha_bounds=(0.001, 5.0))
print(result.mean_reversion, result.long_run_mean, result.volatility)
print(result.fun)
```

The bounds above are an example, not universal defaults. `alpha_bounds` is required and must contain two finite values satisfying `0 < lower < upper`. No initial parameters or mu/sigma bounds are required. Zero residual variance is rejected because there is no positive-volatility interior maximum in that case.

Unlike `fit_ou`, this result's `x` is the **physical scalar alpha**. `log_alpha` retains its search coordinate; `mean_reversion`, `long_run_mean`, and `volatility` expose all three physical parameters. `fun` is the conditional negative log-likelihood and `alpha_bounds` records the supplied range. Failed or non-finite optimization results raise an error.

```bash
python -m pytest tests/test_ou_profile_fit.py tests/test_ou_estimation.py -q
python -m experiments.experiment_ou_profile
# Save the figure without opening a window:
python -m experiments.experiment_ou_profile --no-show --save /tmp/ou_profile.png
```

The profile experiment compares joint and profile fitting on the same irregularly sampled trajectory. It displays the observations and a log-alpha profile score curve, with the true and fitted alpha marked. In the seed-42 example, both methods return approximately `(0.8646, 9.7985, 1.4937)` with NLL `554.53474842`. Agreement verifies this example, not universal parameter accuracy. The joint method additionally constrains mu and sigma; agreement is not expected when those constraints exclude the profile optimum.

Validation includes hand-derived regular and irregular profile scores, independent numerical optimization of the nuisance parameters, and an independent AR(1) conditional-MLE reference for the complete fit. A plotted finite grid and successful bounded optimization do not prove global optimality. Automatic range selection, boundary and flat-profile diagnostics, and calibrated parameter intervals remain unimplemented. The experiment does not perform missing-value reconstruction or held-out forecasting.

## State, Graph, and Dynamical Conventions

States have shape `(N, d)` and trajectories `(T, N, d)`. Observations distinguish measured entries from unobserved entries through a mask. Complete states and incomplete observations must remain separate concepts.

The current graph convention is:

```text
adjacency[i, j] = weight of the edge from node i to node j
```

For the planned dynamics, stack each node's features into a column vector `x` of length `N * d`, keeping each node's features together. A proposed linear model is:

```math
dx_t = \bigl(K(t)x_t + C u_t + b\bigr)\,dt + B\,dW_t.
```

Here `u` is an optional observed external input, `K` is the drift matrix, and `B` determines process noise. The block `K[i, j]` maps node **j into node i**. Thus drift blocks and the existing adjacency convention have opposite source/target indexing.

For scalar nodes, an off-diagonal drift coefficient `K[i, j]` corresponds to `adjacency[j, i]`. For multiple features, an edge corresponds to a block of coefficients; any scalar summary must declare its aggregation rule. This conversion is a design requirement, not an existing helper.

Diagonal drift blocks describe self-dynamics and are reported separately from inter-node edges. Signed drift coefficients are not automatically diffusion weights. Graph Laplacian models require their own sign, orientation, and stability assumptions; `Graph.laplacian` alone does not establish them.

A selected edge describes dependence within the specified dynamical model. It is not automatically a causal effect, a correlation edge, or a physical connection. Unobserved common drivers can change its interpretation.

## Mathematical Foundation

### Stochastic Analysis and Exact Transitions

For constant coefficients over an interval, the linear SDE has a Gaussian transition:

```math
x_{t+\Delta}=F_\Delta x_t+c_\Delta+\eta_\Delta,
\qquad \eta_\Delta\sim\mathcal N(0,Q_\Delta),
```

```math
F_\Delta=e^{K\Delta},\qquad
Q_\Delta=\int_0^\Delta e^{Ks}BB^\top e^{K^\top s}\,ds.
```

For a constant affine drift `b`,

```math
c_\Delta=\int_0^\Delta e^{Ks}b\,ds.
```

External inputs require a declared interpolation or integration rule. Exact transitions depend on the actual elapsed time, so unequal observation intervals need not be replaced by an artificial uniform grid. Intervals crossing changes in `K` require composition of the appropriate transitions and covariances.

The main mathematical tools are Brownian motion, Itô integration, linear SDE solutions, the Markov property, conditional Gaussian distributions, sparse statistical estimation, and numerical linear algebra. More advanced path-measure methods should be introduced only when a specific inference problem requires them.

Even when `K` is sparse, `exp(K * delta)` may be dense. A discrete-time transition graph is therefore not interchangeable with the direct continuous-time drift graph.

### Brownian Reference Model

The implemented independent Brownian model is:

```math
dX_t=\sigma\,dW_t,\qquad
X_{t+\Delta}=X_t+\sigma\sqrt{\Delta}\,Z,
\qquad Z\sim\mathcal N(0,I).
```

For each component of a Brownian bridge with positive volatility and endpoints at `t_L < t_R`,

```math
\mathbb E[X_t\mid X_L,X_R]
= X_L+\frac{t-t_L}{t_R-t_L}(X_R-X_L),
```

```math
\mathrm{Var}(X_t\mid X_L,X_R)
=\sigma^2\frac{(t-t_L)(t_R-t)}{t_R-t_L}.
```

The current multi-point sampler selects a requested time near the temporal midpoint, samples it conditionally, and subdivides the remaining intervals. Reusing sampled boundaries preserves the joint bridge distribution; independently drawing each point from its endpoint marginal would not.

Brownian simulation and bridges remain analytical reference tools. They are not MCMC, and a full Brownian reconstruction product is not a prerequisite for graph estimation. Zero-volatility behavior needs an explicit contract: the current code interpolates even unequal endpoints, whereas a strictly zero-noise Brownian process cannot produce such endpoints.

### OU Reference Model

The implemented OU model applies independently to every node-feature component, with shared scalar parameters:

```math
dX_t=\alpha(\mu-X_t)\,dt+\sigma\,dW_t.
```

For positive `alpha`, `ou_transition` returns the exact transition coefficient, affine offset, and noise variance:

```math
F_\Delta=e^{-\alpha\Delta},\qquad
c_\Delta=\mu(1-F_\Delta),\qquad
q_\Delta=\frac{\sigma^2}{2\alpha}(1-e^{-2\alpha\Delta}).
```

`ou_step` samples the next state and `simulate_ou` returns a trajectory including the initial state. Each step uses its actual elapsed time. Zero mean reversion reduces to Brownian motion; zero volatility gives deterministic evolution. This is forward simulation with known parameters, not OU bridge reconstruction or graph inference. Scalar parameter fitting is implemented separately in `estimation/ou.py`.

### Probability, Bayesian Inference, and MCMC

Probability modeling is foundational; general-purpose MCMC is not an early development milestone.

- Conditional Gaussian state inference with known parameters can be computed analytically. Kalman filtering is a Bayesian state update under its assumptions, without MCMC.
- Early parameter and graph estimation will prioritize likelihoods, regularization, identifiability experiments, and optimization baselines.
- A conjugate Brownian variance calculation may serve as an optional correctness reference. It is not a required release gate.
- General parameter or graph-structure MCMC is deferred until the dynamical likelihood, graph estimator, and basic validation work reliably and a concrete uncertainty question warrants it.
- Later Bayesian work should first exploit model structure, such as integrating out linear Gaussian latent states with a filtering likelihood, before sampling large collections of latent variables.

State uncertainty conditional on fitted parameters does not include all parameter or structural uncertainty. Bayesian posteriors can be prior-sensitive or overconfident under model misspecification. Continuous shrinkage priors do not, by themselves, assign posterior probability to an exactly absent edge.

## Release Roadmap: R1–R4

These releases replace the earlier V1–V7 planning labels. They are development milestones, not completed capabilities or promised package major versions. Basic validation and reliability checks begin in R1; R4 integrates and strengthens them.

| Release | Goal | Question answered | Main output |
| --- | --- | --- | --- |
| R1 | Static relationship estimation | Which fixed dynamical relationships are supported by the data? | Drift coefficients, selected edges, predictions, preliminary stability checks. |
| R2 | Dynamic graph estimation | When do relationships appear, disappear, or change strength? | Piecewise graph estimates and change intervals. |
| R3 | Inference under imperfect observations | What can be inferred despite noise, asynchrony, and missing entries? | Latent-state estimates, graphs, and insufficient-information diagnostics. |
| R4 | Validated reliability workflow | Which edges and changes remain credible under repeated analysis and model checks? | Intervals, selection stability, calibration, and sensitivity reports. |

### R1 — Static Relationship Estimation

**Scope:** complete observations, fixed node identities, a static sparse linear drift, and known or negligible measurement noise. Begin with scalar nodes and regular sampling, then validate actual-time transitions and node-feature blocks.

**Methods:** exact linear-SDE transitions, maximum likelihood, L1 and group sparsity, optional known-edge or forbidden-edge constraints, and observed common inputs. Use least-squares and sparse VAR as discrete-time prediction baselines. Fit known graph structures first to validate parameter estimation, then learn unknown support.

**Difficulties and directions:**

- Too many candidate parameters: use sparse blocks, shared parameters, and small validated systems.
- Common-driver confounding: support observed inputs and include common-driver counterexamples; do not claim this removes hidden confounding.
- Finite-sampling ambiguity: investigate identifiable model classes, sampling intervals, and sensitivity instead of forcing one graph interpretation.
- Nonconvex continuous-time likelihood: use suitable initialization, multiple starts, gradients, and optimization diagnostics. Do not assume a matrix-exponential likelihood is a convex regression problem.
- Shrinkage bias: distinguish penalized edge selection from fixed-structure refitting; refitting alone does not correct post-selection uncertainty.

**Acceptance:** known-truth graph and parameter experiments across seeds, sample sizes, and sampling intervals; no-interaction and simple prediction baselines; documented failures and preliminary edge-stability checks.

### R2 — Dynamic Graph Estimation

**Scope:** piecewise-constant relationships before unrestricted continuously changing graphs. Provide a rolling-window baseline and a joint temporally regularized estimator. Keep retrospective analysis separate from online estimation.

A candidate objective is:

```math
\min_{\{K_k\}}
-\log p(Y\mid K_1,\ldots,K_m,\theta)
+\lambda\sum_{k=1}^{m}\Omega(K_k)
+\gamma\sum_{k=2}^{m}\|K_k-K_{k-1}\|_F.
```

`Omega` penalizes selected inter-node blocks; self-dynamics and other parameters have explicitly declared treatment. Temporal regularization encourages neighboring segments to agree.

**Methods:** rolling estimation, fused or total-variation penalties, warm starts, and suitable proximal, alternating, or ADMM-based optimization. The choice depends on the actual objective; convergence guarantees must not be borrowed from a different convex problem.

**Difficulties and directions:** noisy graph flicker versus oversmoothing; limited evidence near change points; confusion between changed noise and changed drift; and observation intervals crossing segment boundaries. Use no-change controls, noise-only-change controls, minimum segment information requirements, transition composition, and sensitivity to temporal penalties.

**Acceptance:** evaluate no-change, abrupt-change, and gradual-change systems. Report false changes, missed changes, localization error, graph recovery, and prediction performance. An animated graph is not sufficient evidence.

### R3 — Noisy, Asynchronous, and Missing Observations

**Scope:** keep complete data as the default and extend inference through an explicit observation model:

```math
Y_k=H_kx_{t_k}+\varepsilon_k,
\qquad \varepsilon_k\sim\mathcal N(0,R_k).
```

**Methods:** Kalman filtering, RTS smoothing, observed-component updates, interval-specific transitions, filtering marginal likelihood optimization, and EM or generalized EM where appropriate. Use Cholesky factorizations and linear solves instead of explicit dense inverses. Foundational state-space components may be implemented earlier when needed.

**Difficulties and directions:** process and measurement noise can be hard to distinguish; long gaps weaken graph identification; joint optimization can be slow or non-identifiable. Start with known or structured measurement noise, release parameters gradually, and report insufficient evidence. Declare assumptions about the observation mechanism; informative missingness requires a separate model.

Integrate latent states in the inference procedure where possible. Do not fill gaps once and treat the imputed values as new independent measurements.

**Acceptance:** degrade complete known-truth trajectories with controlled measurement noise, asynchronous schedules, and missing blocks. Check prediction and graph degradation, state uncertainty, convergence, and failures. State intervals conditional on estimated parameters must be labeled accordingly.

### R4 — Reliability and Model Checking

**Scope:** provide separate, interpretable assessments rather than an undefined overall confidence score.

| Reliability question | Candidate method | Interpretation |
| --- | --- | --- |
| Does an edge survive resampling? | Dependence-preserving block, local, or model-based resampling with the estimation pipeline rerun. | Selection stability, not probability that the edge is true. |
| How uncertain is an estimated strength? | Suitable likelihood or resampling intervals; explicit conditional versus selection-aware treatment. | Coverage requires validation under the declared procedure. |
| How uncertain is a change location? | Repeated estimation and change-location summaries. | An interval or distribution, not unjustified exact timing. |
| Is the conclusion sensitive to choices? | Sampling, window, regularization, noise-model, and later prior sensitivity checks. | Robustness across declared alternatives. |
| Does the model help on unseen observations? | Walk-forward predictions, predictive scores, and no-edge or perturbed-graph comparisons. | Predictive evidence, not proof of causal structure. |

**Difficulties and directions:** selection bias, nonstationary resampling, multiple candidate edges, model misspecification, and repeated-fit cost. Preserve relevant temporal structure, test full-pipeline coverage in simulations, use independent evaluation data, and parallelize justified repeated fits. Claim error-rate control only when its assumptions and implementation support it.

Optional later Bayesian extensions may address fixed-structure parameter uncertainty and a limited set of candidate graphs. They require a validated likelihood, identifiable scope, and prior/computational diagnostics. General graph-space MCMC is neither required to complete R4 nor a substitute for calibration.

**Acceptance:** measured interval coverage where claimed, informative edge-stability behavior, misspecification tests, runtime and memory reports, reproducible configuration, and an external user able to run the workflow independently.

## Immediate Development Work

Completed groundwork includes repaired trajectory test collection, exact initial-time matching in Brownian simulation, finite-value validation for states and observed entries, package configuration, and independent scalar OU simulation. Remaining work is listed below.

| Priority | Change | Acceptance criterion |
| --- | --- | --- |
| P0 | Fix graph/drift semantics before estimation | Document source/target conversion, self-dynamics, feature blocks, signed weights, and model-specific graph constraints. |
| P0 | Verify packaging | Verify editable installation and examples in a clean environment; package configuration and dependency extras are implemented. |
| P0 | Extend OU validation and fitting usability | Joint and profile fitting have analytical-reference tests. Add boundary and flat-profile diagnostics before automatic range selection; add repeated-seed recovery and held-out diagnostics. Preserve transition moment/composition checks as further validation work. |
| P1 | Define remaining data contracts | Specify zero-volatility bridge behavior, empty inputs, and array copying/sharing rules. |
| P1 | Clean up the bridge experiment | Add an execution entry point, optional verbose tracing, and separate plotting from reusable computation. |
| P2 | Improve bridge queue handling | Replace front-removal from a list with a queue; profile other searches before optimizing and preserve joint covariance. |

Then build this small end-to-end workflow:

```text
Known sparse linear stochastic system
    -> Exact-transition simulation
    -> No-interaction and discrete-time prediction baselines
    -> Fixed-structure continuous-time parameter estimation
    -> Error, convergence, and failure report
    -> Unknown-edge estimation
```

Use no-edge, directed-chain, and sparse stable examples with recorded true parameters. Start with declared diffusion parameters to isolate drift-estimation correctness. Validate transition moments and covariance composition before expanding the estimator.

Suggested five work sessions, adjusted to actual progress:

1. Complete OU validation and clean-environment installation checks; establish graph/drift conventions.
2. Implement known-structure linear-SDE simulation and numerical checks.
3. Fit simple discrete-time prediction baselines with chronological splits.
4. Fit a small fixed-structure continuous-time model and inspect multiple initializations.
5. Run reproducible experiments over sampling intervals, noise levels, and seeds; record failures and the next blocker.

Do not advance merely to meet a date if the preceding numerical checks fail. Dynamic visualization, generic plugin architecture, and MCMC should not displace this initial closed loop.

## Evaluation and Release Gates

Every release needs both a usable workflow and evidence supporting its conclusions.

- **Implementation correctness:** analytical limits, transition moments, covariance composition, and meaningful deterministic or statistical tests.
- **Graph recovery:** distinguish direct drift edges from discrete-time propagation; evaluate false edges, missed edges, and strength error against synthetic truth.
- **Time variation:** include no-change, drift-change, and noise-only-change controls.
- **Predictions:** use chronological holdouts, predictive errors and scores, and suitable baselines. Fit scaling, graph selection, and tuning only on permitted training/validation data.
- **Reliability:** assess interval width and empirical coverage, stability, sensitivity, and failure under model misspecification.
- **Usability:** installation, a working example, documented result semantics, reproducible experiment records, and measured resource use at the supported scale.

Separate known-parameter/correct-model checks, estimated-parameter/correct-model experiments, and misspecified-model experiments. Synthetic ground truth supports graph-recovery evaluation; predictive success on real data does not establish a true or causal graph.

For point prediction, score a declared point estimator rather than an arbitrary posterior path draw. Path samples, conditional means, and uncertainty summaries are different outputs.

Adding latent query points does not add observations. Refining a bridge grid must preserve the posterior distribution at existing query times, not necessarily the same seeded sample. With fixed model parameters in a linear Gaussian system, adding observations cannot increase conditional covariance in the positive-semidefinite ordering; this does not imply the same monotonic behavior after refitting an uncertain model.

A future posterior path sampler with shared uncertain parameters must draw those parameters once per joint path, not independently at each time point. Resampling stability, confidence intervals, conditional state uncertainty, and Bayesian posterior probabilities must remain separately labeled.

## Development Sequence and Planning

```text
Correctness and packaging
    -> Known-structure simulation and parameter estimation
    -> R1: unknown static relationships
    -> R2: piecewise dynamic relationships
    -> R3: imperfect observations
    -> R4: validated reliability workflow
```

Research estimates, not deadlines: with one primary developer contributing roughly 25–30 focused hours per week, a constrained R1 may take about two months, R2 about four months cumulatively, R3 about six, and R4 roughly eight to nine. These estimates include iteration and validation, exclude unrestricted structural MCMC and domain products, and must be revised after the first estimation benchmark. R2 can already support research on complete data; R3 is the intended wider trial stage. Release by evidence, not calendar alone.

## Deferred Research

The following are possible extensions, not R1–R4 requirements:

- nonlinear, jump, event-driven, or unrestricted continuously changing dynamics;
- latent communities, changing node identities, and semantic behavior or intention inference;
- active observation and experimental design;
- general graph-structure Bayesian sampling and hierarchical model families;
- task-driven adaptive path resolution;
- GPU/distributed execution without a measured bottleneck;
- domain-specific products and a broad plugin ecosystem.

Adopt an extension when an established use case or benchmark demonstrates the limitation it addresses. Preserve a small exact or analytically checkable reference whenever possible.

## Repository Layout

```text
src/dynsample/
    core/                       # State, Trajectory, Observation, Graph
    simulation/brownian.py      # Implemented reference simulation
    simulation/ou.py            # Exact independent scalar OU simulation
    inference/reconstruction/
        brownian_bridge.py      # Implemented conditional sampling
    estimation/ou.py            # Scalar OU likelihood, initialization, and fitting
    metrics/                    # Placeholder
    sampling/                   # Placeholder
experiments/
    experiment_brownian_bridge.py
    experiment_ou.py
    experiment_ou_estimation.py
    experiment_ou_profile.py    # Joint/profile comparison and alpha score curve
tests/
requirements.txt
pyproject.toml                  # Package metadata and dependency extras
```

Extract shared transition and estimation interfaces when working implementations demonstrate the need. Keep computation, experiment evaluation, and visualization separable.

## Status

- [x] Core state, trajectory, observation, and graph representations.
- [x] Independent Brownian simulation on irregular requested times.
- [x] Single-point and joint multi-point Brownian bridge sampling.
- [x] Empirical checks of bridge mean, variance, and joint covariance.
- [x] Brownian bridge demonstration.
- [x] Independent scalar OU transitions and trajectory simulation on irregular times.
- [x] OU visualization with conditional mean and pointwise state intervals.
- [x] Scalar OU conditional likelihood, bounded fitting, and automatic starting values.
- [x] OU estimation experiment with in-sample diagnostics and fitted-model simulations.
- [x] Scalar OU profile fitting with supplied alpha bounds, analytical-reference tests, and a profile-curve experiment.
- [x] Core finite-value validation, repaired trajectory tests, and package configuration.
- [ ] Remaining correctness checks and clean-environment installation verification listed above.
- [ ] Known-structure linear-SDE simulation and parameter-estimation benchmark.
- [ ] R1: unknown static relationship estimation.
- [ ] R2: dynamic relationship estimation.
- [ ] R3: noisy, asynchronous, and incomplete observations.
- [ ] R4: validated reliability workflow.

The next deliverable is an installable, reproducible linear-system estimation experiment. The longer-term goal is a tool that helps researchers identify both supported relationships and the limits of what their observations can establish.
