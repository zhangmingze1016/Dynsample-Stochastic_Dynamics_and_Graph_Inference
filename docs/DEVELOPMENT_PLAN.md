# Dynsample Development Execution Plan

Updated: 2026-10-05. Baseline: `0b159cb` (v027), with v024 flexible OU interfaces, v025 time-rescaling validation, v026 LinearSDE, and v027 coupled transitions and simulation committed. This document governs development order, scope, and acceptance criteria; the README provides a summary. It is not a list of implemented features. Files and functions marked as planned do not exist yet.

## 1. Objective and Working Rules

Estimate dynamical dependencies from timestamped node observations, with interpretable diagnostics and reliability evaluation. Support both engineering use and scientific validation. Start with complete observations and static relationships, then extend to changing relationships and imperfect observations.

Development sequence:

```text
Scalar OU closeout (A)
  -> Graph semantics and matrix transitions (B)
  -> Parameter inference with known structure (C)
  -> Unknown sparse graph inference (D)
  -> R1 validation and release, with static hierarchy prototype (E, I1)
  -> R2 time-varying graphs and hierarchy tracking (F, I2–I3)
  -> R3 missing/noisy/asynchronous observations (G)
  -> R4 reliability evaluation (H)
```

- R denotes a release or milestone, not a model count or a Git commit label such as v019.
- Current task: finish B with `graph_from_drift` and direction/weight validation. Scalar OU work is frozen; LinearSDE, matrix transitions, single-step sampling, and trajectory simulation are implemented. Do not add overlapping scalar experiments.
- Before providing code, identify the task, objective, file, function, mathematical assumptions, return values, and acceptance checks. Explain code line by line or block by block.
- The user studies and enters implementation, test, and experiment code. Provide copyable code with explanations by default; do not write these files without an explicit request to edit them. Documentation edits and execution of saved code follow the current task authorization.
- Record the implementation, relevant tests, and unresolved limitations when completing a task. One successful experiment does not establish correctness for all inputs, global optimality, or statistical reliability.
- Explain and discuss mathematical obstacles involving identifiability, optimization guarantees, or stability constraints before implementing a solution. An optimizer's success field cannot replace an argument.
- Record new requests in the backlog before inserting them into the sequence. Document the reason, alternatives, and acceptance changes for a scope revision, and confirm the revision with the user.
- Default usage should not require manually supplied parameter search bounds. Retain overrides. Positivity, positive definiteness, stability, and other model constraints must remain explicit; automation cannot remove non-identifiability.
- Do not create empty future modules or build generic utilities, plugin frameworks, or a GUI in advance. Extract shared logic when actual reuse justifies it.

### Development Cadence

Technical dependencies, evidence from failures, and stage acceptance determine the sequence. Discussing an idea does not automatically change implementation order. Planning does not require freezing every distant implementation detail.

1. Work on one verifiable objective at a time. Explain the mathematics, assumptions, and hand calculation or independent reference before providing and explaining a core function.
2. Review the implementation after the user enters it, then run relevant tests. Add an experiment when tests alone do not adequately describe overall behavior. Reuse existing experiments rather than repeatedly adding similar plots.
3. Commit a complete feature after its tests pass; do not measure progress by partially implemented function counts. Review the diff before committing and run the full suite at stage exits.
4. Complete an end-to-end workflow at each stage, then review errors, failures, and complexity. Optimize measured bottlenecks and abstract actual duplication.
5. Plan A's three work packages in detail. Fix responsibilities and acceptance criteria for B–E, selecting research algorithms at their mathematical checkpoints. Retain objectives and candidate functions for F–H without prematurely promising signatures or solver guarantees.
6. Handle routine spelling and implementation choices directly. Discuss changes to mathematical models, research scope, and unproven guarantees explicitly. Avoid interrupting ordinary development with repeated clarification requests.

Schedule verifiable work units rather than rigid day counts: B drift-to-graph semantics and tests -> B completion audit -> C likelihood and known-structure fitting -> D sparse graph selection. Revise the sequence when new evidence warrants it, not whenever a new topic comes up.

### Current checkpoint and experiment evidence

- Implemented: `LinearSDE`, `linear_transition`, `linear_step`, and `simulate_linear`, with one feature per node. Sampling handles positive-semidefinite covariance through a checked eigendecomposition. Extreme block-exponential overflow is reported; it is not automatically resolved.
- The user reported all 40 tests in `tests/simulation/test_linear.py` passing. This documentation update did not independently rerun the full suite.
- All eight existing experiment scripts were rerun on 2026-10-05. Settings, environment, numbers, and figures are recorded in [EXPERIMENTS.md](EXPERIMENTS.md). Experiment completion is not a substitute for test assertions or graph-recovery evaluation.
- Next: inspect the existing Graph contract, decide signed-weight compatibility explicitly, implement `graph_from_drift`, remove diagonal self-dynamics, preserve j-to-i direction through transposition, and document any threshold as a conversion choice rather than statistical evidence.
- Remaining B acceptance work: audit the current tests against the stage requirements, include direction and signed-weight tests, and verify any missing no-edge control before closing B. The existing three-node experiment replaces the proposed two-node demo; extend it only when an uncovered question justifies the change.

### Product scope and deferred extensions

The primary output is model-supported relationships, their reliability, and their changes; fitted coefficients serve this purpose. Keep observed states, inferred states, fixed-graph propagation, and changing relationships distinct. Behavioral labels are optional interpretation, not mandatory inference outputs. Zero, positive, and negative coefficients describe scalar dynamic channels, not every possible kind of relationship.

Preserve A-D dependencies. Observed-input support, multi-feature block inference, task-dependent feature/node importance, and selected nonlinear interaction functions are extension work requiring separate assumptions and acceptance checks; they are not prerequisites for the first scalar-node graph estimator. Feature importance must account for units and redundant inputs, and node importance must specify a task and horizon. Future relationship forecasting requires its own evolution model and is not implied by change detection.

The eventual user workflow should be data -> validated default analysis -> a simple state-and-graph view, with a timeline, uncertainty, and optional details. Keep advanced controls separate. Preserve full matrices and channel information under visual summaries. Build visualization around validated outputs; no unified analysis API or interactive graph viewer exists yet.

## 2. Baseline: Existing Files and Responsibilities

Existing functionality was identified through code inspection. This is not a claim that the full test suite was rerun during this documentation task.

| File, relative to repository root | Functions or types | Current responsibility and limits |
| --- | --- | --- |
| `src/dynsample/core/state.py` | `State.__post_init__`, `n_nodes`, `n_features` | Validate a complete state `(N,d)` at one time and expose dimensions. |
| `src/dynsample/core/trajectory.py` | `Trajectory.__post_init__`, `__len__`, `n_steps`, `n_nodes`, `n_features`, `state_at` | Represent complete trajectories `(T,N,d)`, strictly increasing times, and state access. |
| `src/dynsample/core/observation.py` | `Observation.__post_init__`, `n_nodes`, `n_features`, `n_observed` | Represent a masked observation at one time; no general missing-data inference yet. |
| `src/dynsample/core/graph.py` | `Graph.__post_init__`, `n_nodes`, `degree`, `degree_matrix`, `laplacian`, `is_directed` | Store supplied adjacency relationships; does not learn graphs. |
| `src/dynsample/core/linear_model.py` | `LinearSDE` | Validate constant K, b, B; one feature per node in the current simulation contract. |
| `src/dynsample/simulation/linear.py` | `linear_transition`, `linear_step`, `simulate_linear` | Exact Gaussian transitions and coupled simulation at supplied times; no graph fitting. |
| `src/dynsample/simulation/brownian.py` | `brownian_step`, `simulate_brownian` | Simulate independent zero-drift increments with shared scalar volatility. |
| `src/dynsample/simulation/ou.py` | `ou_transition`, `ou_step`, `simulate_ou` | Exact scalar OU transitions `(F,offset,q)` and simulation; nodes remain independent. |
| `src/dynsample/inference/reconstruction/brownian_bridge.py` | `brownian_bridge_step`, `brownian_bridge` | Sample individual states and joint paths conditional on endpoints; no parameter or graph inference. |
| `src/dynsample/estimation/brownian.py` | `fit_brownian_drift` | Drift and volatility MLE for complete irregular scalar data; also the small-alpha OU boundary reference. |
| `src/dynsample/estimation/ou.py` | `ou_negative_log_likelihood` | Scalar OU conditional NLL. |
| Same file | `_ou_objective`, `_initial_ou_parameters`, `fit_ou`, `fit_ou_joint` | Public dispatch through `fit_ou`; `fit_ou_joint` supports optional starts and optional bounds, with `x` in optimizer coordinates. |
| Same file | `_profile_ou_mu`, `_profile_ou_sigma`, `_profile_ou_negative_log_likelihood` | Analytically eliminate mu and sigma at fixed alpha and evaluate the profile NLL. |
| Same file | `fit_ou_profile` | Automatic bounded expansion when bounds are omitted; grid scan and local refinement within explicit bounds; `x` is physical alpha. |
| Package `__init__.py` files | Package exports | Top-level exports include State, Trajectory, fit_ou, fit_ou_joint, and fit_ou_profile; metrics and sampling remain placeholders. |

Existing validation files:

| File | Validation responsibility |
| --- | --- |
| `tests/core/test_state.py`, `test_trajectory.py`, `test_observation.py`, `test_graph.py` | Data contracts; fill specific discovered gaps as needed. |
| `tests/core/test_linear_model.py`, `tests/simulation/test_linear.py` | Model validation, OU reduction, singular drift, shared noise, composition, sampling moments, and irregular trajectories. |
| `tests/simulation/test_brownian.py`, `test_ou.py` | Transitions, simulation, time and parameter validation, reproducibility. |
| `tests/inference/test_brownian_bridge.py` | Endpoint conditions, joint means and covariances; correct zero-noise semantics in E. |
| `tests/estimation/test_ou_estimation.py` | NLL, initialization, analytical profile parameters, independent optimization references. |
| `tests/estimation/test_ou_profile_fit.py` | Complete profile fitting, supplied ranges, and boundary flags. |
| `tests/estimation/test_brownian_estimation.py` | Analytical drifted Brownian MLE and independent density checks. |
| `tests/estimation/test_ou_limits.py` | Numerical checks near both limits; not a substitute for theoretical proofs. |

Existing experiment `main` functions or script entry points organize data, calls, reporting, and plots. Reusable estimation logic belongs in the package:

- `experiments/experiment_brownian_bridge.py`: conditional paths and bridge distributions; entry-point cleanup belongs to E.
- `experiments/experiment_ou.py`: independent OU simulation and conditional intervals with known parameters.
- `experiments/experiment_ou_estimation.py`: joint fitting, residuals, and new trajectories.
- `experiments/experiment_ou_profile.py`: joint/profile comparison and sensitivity to supplied ranges.
- `experiments/experiment_ou_interface.py`: eight optional-input combinations through `ds.fit_ou`, with parameter and NLL comparison plots.
- `experiments/experiment_ou_brownian_limit.py`: small-alpha limits of alpha*mu, sigma, and NLL.
- `experiments/experiment_ou_large_alpha_limit.py`: large-alpha limits of mu, sigma²/(2alpha), and NLL.

## 3. A: Scalar OU Closeout — Three Work Packages Only

Objective: a usable scalar estimation entry point that does not require bounds by default. A complete scalar statistics package is outside this stage's scope.

### A1. Automatic Search and Boundary Diagnostics — Implemented; Acceptance Audit Remains

Keep new computations in `src/dynsample/estimation/ou.py` initially. Do not build a general optimization framework.

| Function | Input/output objective | Limits of responsibility |
| --- | --- | --- |
| `_initial_ou_alpha_bounds` | Return an initial positive interval from trajectory times, using duration and interval scales; rescale correctly when time units change. | Does not claim to contain the global optimum. |
| `_ou_boundary_scores` | Return Brownian and independent Gaussian boundary NLLs and reference parameters; conditional on x0, use only x1…xn for the Gaussian reference. | Do not represent boundary models as finite OU parameters. |
| `_scan_ou_profile`, `_search_ou_profile`, `_search_ou_profile_auto` | Scan candidate minima, refine locally, and control bounded expansion in separate helpers; retain the best finite candidate and search records. | Do not assume unimodality or guarantee global optimality. |
| `_diagnose_ou_search` | Combine finite candidates, boundary scores, failures, and computational budget into diagnostics. | Similar scores are not confidence intervals or model probabilities. |
| `fit_ou_profile`, extending the existing function | Use automatic search for `alpha_bounds=None`; preserve constrained search for explicitly supplied bounds. | Do not silently exceed supplied bounds or change existing `x` semantics. |

Remaining acceptance review must check consistent conditional likelihoods, changes of time units, candidate minima, and expansion stopping rules. Treat degenerate variance, extreme interval ratios, and non-finite calculations separately.

Consider `0.01/T` and `-log(0.01)/min(dt)` as initial bounds to validate, not guaranteed limits. The value 0.01 is not a coverage guarantee; a tiny dt can create an enormous range. Centralize and record search settings instead of scattering unexplained constants.

Required result information: physical parameters, finite-candidate NLL, both boundary NLLs, initial/final ranges, evaluation and expansion counts, stopping reason, and diagnostics. Distinguish at least an interior candidate, boundary competition, numerical flatness, budget exhaustion, and numerical failure. Numerical flatness is a tolerance-based diagnostic, not a statistical identifiability test. If a boundary scores better, retain the finite candidate for inspection but state that a reliable finite OU solution has not been established; do not silently switch models.

Acceptance: automatic and manual searches agree on an interior solution; both boundary competition and degenerate data are covered; budget exhaustion is reported honestly; alpha, sigma, and NLL transform as theoretically expected under time rescaling; search logic handles multiple candidate minima in dedicated tests. If profile evaluation near either boundary is unstable, pause to examine the parameterization before proceeding.

### A2. Interface and Compatibility — Implemented in v024

Keep implementation in `estimation/ou.py`, updating `tests/estimation/test_ou_estimation.py` and `tests/estimation/test_ou_profile_fit.py`.

- Preserve `fit_ou_joint` as an explicit joint optimizer. Starts and bounds are independently optional; omitted bounds do not introduce arbitrary finite parameter limits.
- `fit_ou(trajectory)` dispatches to automatic profile estimation. Either initial parameters or three-parameter bounds selects the joint path; all eight combinations with optional alpha bounds are supported. Intersect overlapping alpha ranges and reject contradictory inputs, not mixed configurations themselves.
- Keep `_ou_objective` and `_initial_ou_parameters` for the joint path; neither is a bounds generator.
- Encourage named fields: `mean_reversion`, `long_run_mean`, `volatility`, `fun`, `method`, and diagnostics. Add named physical parameter fields to joint results.
- Do not silently change joint `x=[log(alpha),mu,log(sigma)]` into physical parameters; profile `x` remains physical alpha. Document the distinction in the README. Consider a new result type only through a separate versioned migration.

Acceptance: regression tests cover both existing calls and the new default; the default path requires no mu/sigma initialization or bounds; successful numerical stopping remains distinct from a credible interior candidate.

### A3. Validation, Example, and Scope Freeze — Historical Acceptance Scope

| Planned or updated file | Functions/entry points and objectives |
| --- | --- |
| `tests/estimation/test_ou_profile_fit.py`, existing | Existing tests cover automatic search, boundary states, budgets, input validation, and independent MLE references. Audit full-fit time rescaling and remaining numerical edge cases before claiming all A1 acceptance criteria; no separate auto-fit test file is required. |
| `tests/estimation/test_ou_profile_fit.py`, `test_ou_estimation.py`, `test_ou_limits.py` | Preserve analytical references, legacy calls, and both boundary regressions; avoid testing only the implementation against itself. |
| `experiments/experiment_ou_interface.py`, `experiment_ou_profile.py` | The interface experiment compares eight input combinations; retain the profile experiment for score curves and restrictive-range diagnostics. Do not add further overlapping scalar demos. |
| `README.md` | Provide default and manual examples, assumptions, and result-status semantics. |

Exit A when relevant tests and the full suite pass, automatic calls and diagnostics are reproducible, and documentation separates numerical from statistical conclusions. Proceed to B.

Excluded: MCMC, scalar parameter confidence intervals, OU bridges, a new GUI, C++, new scalar models, and open-ended global optimization research.

## 4. B: Graph Semantics and Matrix Transitions

Status: model and simulation functions are implemented in v026-v027; graph conversion and the final acceptance audit remain.

Scope: complete observations `(T,N,1)`, starting with two nodes and `dX=(KX+b)dt+B dW`. Model matrices support general finite values; stability constraints belong to the estimation model specification. Distinguish correlated noise through B from drift relationships through K.

| Planned file; existing files updated where applicable | Functions or types | Objective |
| --- | --- | --- |
| `src/dynsample/core/linear_model.py` | `LinearSDE`, `__post_init__`, `n_nodes` | Store K, b, and B; validate shapes and finite values. Initially one feature per node. B's column count represents the number of noise sources. |
| `src/dynsample/core/graph.py` | `graph_from_drift` | K[i,j] represents j→i, so adjacency[j,i]=K[i,j]. Keep diagonal self-dynamics separate from edges. Record any threshold explicitly. |
| `src/dynsample/simulation/linear.py` | `linear_transition` | Return F, c, and Q using matrix exponentials and block-matrix integration without requiring invertible K. |
| Same file | `linear_step`, `simulate_linear` | Generate State/Trajectory through exact Gaussian transitions, retain actual times, and handle degenerate positive-semidefinite simulation covariance. |
| `tests/core/test_linear_model.py`, `tests/simulation/test_linear.py`, `tests/core/test_graph.py` | Parameter, direction, scalar-reduction, zero-drift, composition, and covariance tests | Verify F(a+b), c/Q composition, positive semidefiniteness, and Monte Carlo moments. |
| `experiments/experiment_linear.py` | `main` | Implemented: a three-node directed chain, analytical mean curves, and transition-moment comparison. A no-edge control remains an acceptance audit item; do not claim inferred relationships. |

Acceptance: one node reduces to OU; uncoupled nodes reduce to independent models; nonzero affine drift is correct; j→i is never reversed. Handle rounding errors in Q with declared tolerances rather than arbitrary jitter that hides errors. Analyze subdivision/composition alternatives if stiffness causes unstable matrix exponentials or covariance calculations.

## 5. C: Parameter Inference with Known Graph Structure

Input: complete trajectories, a known edge mask, and initially a supplied diffusion B. Output: mask-constrained K, b, score, and numerical diagnostics. Start with fixed noise, then add estimation of positive diagonal diffusion as a separate step within this stage. Do not begin by estimating unrestricted full covariance.

| Planned file | Functions | Objective |
| --- | --- | --- |
| `src/dynsample/estimation/linear.py` | `linear_negative_log_likelihood` | Exact multivariate transition NLL using Cholesky factorizations and linear solves, not explicit inverses. Ordinary densities require positive-definite Q; explicitly reject unsupported singular cases. |
| Same file | `_pack_drift_parameters`, `_unpack_drift_parameters` | Map between masked parameters and optimization vectors; excluded edges remain zero. |
| Same file | `_initial_linear_parameters` | Construct starts compatible with model constraints; use short-step approximations only for initialization. |
| Same file | `fit_linear_known_graph` | Fit from multiple starts, record scores and convergence, and avoid requiring user-supplied parameter bounds by default. Use positive transforms when estimating diffusion. |
| `src/dynsample/inference/prediction.py` | `predict_linear` | Return fitted conditional means and process covariances; distinguish updated one-step predictions from forecasts starting at a fixed state. |
| `tests/test_linear_estimation.py`, `test_linear_prediction.py` | Density-reference, mask, recovery, prediction-moment, and failure tests | Cover sensitivity to initialization and irregular times. |
| `experiments/experiment_known_graph_fit.py` | `main` | Fit known no-edge, directed-chain, and sparse stable structures; record errors, failure rates, and runtime. |

Mathematical checkpoint before implementation: choose how to constrain stable drift. Do not assume arbitrary K is stable or describe a sufficient diagonal-dominance condition as the entire stable family. Compare a restricted stable model family against general K with stability constraints, and record the choice and scope. Pause to discuss unexplained sampling aliasing or conflicting solutions from different starts.

Acceptance: match independent density references; record estimation errors across experimental conditions rather than expecting one fit to equal the truth; validate fixed and estimated noise separately; test changes of time units. C performs parameter inference, not unknown-edge selection.

## 6. D: Unknown Static Sparse Graph Inference — R1 Core

Scope: N scalar nodes, complete exact observations, and fixed K, starting with small systems. Use exact likelihood with an off-diagonal sparsity penalty and separate treatment of self-dynamics. Select regularization strength automatically by default, allowing overrides; regularization is not a parameter search bound.

| Planned file | Functions | Objective |
| --- | --- | --- |
| `src/dynsample/estimation/sparse_graph.py` | `_off_diagonal_penalty` | Declare penalty scope and scale; do not penalize b or diagonal terms unless the model explicitly specifies it. |
| Same file | `_fit_graph_at_penalty` | Fit a candidate K at a supplied penalty; select an optimizer based on the actual objective. |
| Same file | `_select_graph_penalty` | Select penalties with chronological training/validation; fit preprocessing on training data only. |
| Same file | `fit_graph` | Organize candidate fitting, selection, results, and graph conversion; return drift, offset, diffusion, graph, NLL, penalty, and diagnostics. |
| `src/dynsample/estimation/baselines.py` | `fit_independent_ou`, `fit_discrete_var` | Supply a no-interaction baseline and a VAR baseline on suitable regular grids; do not treat irregular data as equally spaced. |
| `src/dynsample/metrics/graph.py` | `graph_recovery_metrics` | Exclude diagonals; report precision/recall, false/missed edges, and weight errors; define empty-graph denominator behavior. |
| `src/dynsample/metrics/prediction.py` | `prediction_metrics` | Score declared predictive means for point errors and predictive distributions for NLL; do not use random paths as point predictions. |
| `tests/test_sparse_graph.py`, `test_metrics.py`, `test_baselines.py` | Support/direction, empty-graph, leakage, and baseline-applicability tests | Do not establish recovery from one favorable seed. |
| `experiments/experiment_static_graph.py` | `main` | Vary true graphs, seeds, duration, sampling density, and noise; report recovery, prediction, failures, and runtime. |

Mathematical checkpoint: exp(K dt) is generally dense, so thresholding F does not recover the direct drift graph. The likelihood is generally nonconvex in K; ordinary Lasso convexity guarantees do not transfer. Document how sparsity, stability, noise scale, and penalties interact before implementation. Do not present an unresolved algorithm choice as settled. Edges describe dynamical dependence within the specified model.

Acceptance: record configurations and evaluation procedures in advance; report results and failures across conditions, including no-edge negative controls; compare held-out predictions with independent models. Sparse selection is not an edge confidence probability.

## 7. E: R1 Release and Documentation

| File | Objective |
| --- | --- |
| `src/dynsample/__init__.py` and subpackage exports | Provide a minimal public interface; internal helpers are not stable APIs. |
| `docs/USAGE.md`, planned | Explain input shapes, default fitting, overrides, result access, assumptions, errors, and diagnostics. |
| `docs/MATHEMATICS.md`, planned | Document data models, transitions, likelihoods, edge direction, limitations, and references; screenshots do not replace derivations. |
| `README.md` | Installation, a minimal working graph-inference example, supported scope, and a link to this plan. |
| `pyproject.toml`, `requirements.txt` | Keep dependencies, versions, and Python support consistent; verify installation in a clean environment. |
| Existing core/bridge files and corresponding tests | Define empty-input and array-sharing contracts; reject incompatible distinct endpoints for zero-noise Brownian bridges, updating documentation and legacy tests together. |
| `experiments/experiment_brownian_bridge.py` | Clean entry point and optional logging; address queue efficiency only as needed without delaying graph inference. |
| Release record in this document | Record the full suite, clean installation, documentation examples, experiment summaries, resource measurements, and limitations. |

Stage E also includes the bounded automatic static hierarchy prototype in I1. Dynamic group tracking and computational savings remain early-R2 tasks, not assumed R1 capabilities.

R1 is complete when an external user can supply a complete multivariate node time series, fit a static graph, read weights/directions/diagnostics, and reproduce experiments. Simulation and a plot window alone are insufficient. Determine supported node counts through measurements rather than promising scale or accuracy in advance.

## 8. F–H: R2–R4 File and Function Objectives

These are responsibility-based design targets, not frozen signatures. At each stage, complete the mathematical design using evidence from prior stages. Record changes to files/functions here before implementation, without implying unresolved research questions are already solved.

### F / R2: Time-Varying Graphs

| Planned file | Functions | Objective |
| --- | --- | --- |
| `simulation/piecewise.py` | `compose_transitions`, `simulate_piecewise_linear` | Compose F/c/Q across segment boundaries and generate data with known change points. |
| `estimation/dynamic_graph.py` | `fit_rolling_graph`, `_temporal_penalty`, `fit_dynamic_graph` | Establish a rolling baseline, then introduce temporal fusion. Use a specified default window/penalty selection procedure with overrides. |
| `metrics/change_points.py` | `change_point_metrics` | Report false/missed changes and localization errors with explicit matching tolerances. |
| `tests/test_piecewise.py`, `test_dynamic_graph.py` | Tests organized by these contracts | Cover boundary-crossing transitions, no change, noise-only change, and actual drift change. |
| `experiments/experiment_dynamic_graph.py` | `main` | Evaluate abrupt and gradual changes, recording graph recovery, detection delay, and prediction. |

Acceptance: do not automatically interpret noise changes as graph changes. Compare against the rolling baseline; animation is not validation. Analyze nonconvex fused objectives and insufficient window information before proceeding. Depends on R1. Pair the first rolling baseline with I2 dynamic group tracking and I3 scaling experiments; these precede more elaborate joint temporal optimization.

### G / R3: Missing Data, Measurement Noise, and Asynchronous Sampling

| Planned file | Functions or types | Objective |
| --- | --- | --- |
| `core/observation_series.py` | `ObservationSeries`, `__post_init__` | Define time events, observed components/observation matrices, and noise contracts without changing complete Trajectory semantics. |
| `inference/kalman.py` | `kalman_filter` | Predict over actual intervals, update using observed components only, and return state moments and innovation likelihood. |
| `inference/smoothing.py` | `rts_smoother` | Reconstruct intermediate states using all observations; return conditional moments and cross moments needed for EM. |
| `estimation/state_space.py` | `state_space_negative_log_likelihood`, `fit_state_space_graph` | Fit using the filtering likelihood; decide whether EM/generalized EM applies after deriving the actual objective. |
| `tests/test_kalman.py`, `test_smoothing.py`, `test_state_space.py` | Contract tests | Use small joint-Gaussian references; cover prediction-only missing events, complete-observation limits, positive semidefiniteness, and failures. |
| `experiments/experiment_missing_graph.py` | `main` | Vary random/block missingness, asynchronous schedules, and noise; evaluate state reconstruction separately from graph recovery. |

Acceptance: do not treat imputed values as new observations. Distinguish conditional intervals with fixed parameters from parameter uncertainty. Do not promise a universal maximum missing percentage. Initially fix or structure process and measurement noise, then relax parameters separately. Depends on B–D; validate static models before integrating R2.

### H / R4: Reliability and Model Checking

| Planned file | Functions | Objective |
| --- | --- | --- |
| `inference/reliability/resampling.py` | `resample_trajectory` | Declare block/local/model-based resampling and preserve temporal dependence; do not default to independent row resampling. |
| `inference/reliability/stability.py` | `edge_stability` | Rerun the full selection/fitting workflow; return edge selection frequencies and failures, not probabilities that edges exist. |
| `inference/reliability/intervals.py` | `parameter_intervals` | Distinguish fixed-structure from post-selection inference; validate coverage before making corresponding confidence claims. |
| `inference/reliability/sensitivity.py` | `sensitivity_analysis` | Assess sensitivity to time scales, windows, regularization, and noise assumptions. |
| `metrics/calibration.py` | `coverage_metrics` | Report coverage, interval width, valid repetition counts, and failure fractions. |
| `tests/test_reliability.py`, `experiments/experiment_reliability.py` | Tests and `main` | Evaluate calibration under correct specification, misspecification, and varying sampling conditions; record computational cost. |

Acceptance: name reliability measures separately rather than combining them into an unexplained overall score. Retain resampling fit failures in reports. Bayesian inference is an optional later method requiring explicit priors, identifiability, and computational diagnostics; general graph-space MCMC is not a required deliverable.

Package paths in the F–H tables are relative to `src/dynsample/`; `tests/` and `experiments/` paths are relative to the repository root.

### I. Automatic Hierarchy — Late R1 Prototype and Early R2 Extension

Scope revision requested on 2026-09-30: groups must be discovered from observed dynamics, not require user-supplied domain labels. Introduce a bounded automatic grouping and visualization prototype in late R1, followed by temporally consistent grouping and a computational approximation benchmark in early R2. A–D remain unchanged. Known groups are test references or optional overrides, not the final user workflow. All terminology and interfaces remain domain-independent.

Distinguish three deliverables: an aggregate view of a fitted graph, an estimated model of aggregate dynamics, and a computationally cheaper hierarchical estimator. Completing the first does not establish the second or third. Begin with two levels and hard disjoint membership, allowing singleton groups and an unresolved/ungrouped status. More levels and overlapping membership are deferred. Do not force every dataset to contain useful groups.

#### I1. Late R1: Data Contracts, Automatic Static Groups, and Inspection

Depends on D producing a validated single-layer graph. Add the following structures without replacing State, Trajectory, or Graph, or inserting aggregate nodes into the observation array as independent measurements.

| Planned file | Type/function | Contract and objective |
| --- | --- | --- |
| `src/dynsample/core/hierarchy.py` | `HierarchySnapshot`, `__post_init__`, `members` | Store the fitting window, fixed base-node IDs, aligned membership vector, group IDs, grouping method/settings, and diagnostics. Membership is separate from node identity. |
| Same file | `HierarchicalGraph` | Bundle the base graph, snapshot, aggregation map, group-level view, and provenance. Distinguish estimated drift edges from summarized edges, membership links, and loadings. |
| `src/dynsample/inference/grouping.py` | `discover_groups`, `_select_group_resolution` | Discover groups from declared dynamical similarity or graph structure, with default resolution selection and overrides. Record whether signed/directed information is used or transformed. No manual group labels required. |
| `src/dynsample/inference/aggregation.py` | `aggregate_trajectory`, `summarize_group_graph` | Produce group states with recorded weights and coverage, and separately summarize base edges. An edge summary is not an inferred aggregate drift matrix. |
| `src/dynsample/inference/hierarchical_graph.py` | `build_hierarchical_graph`, `extract_subgraph` | Assemble the view and expose within-group and cross-group node edges without fitting all levels jointly. |
| `src/dynsample/visualization/hierarchy.py` | `plot_hierarchy` | Plot aggregate nodes and a selected group's constituents, with consistent labels, direction, and distinct edge types. Return plot objects; no standalone GUI framework. |
| `tests/test_hierarchy.py`, `test_grouping.py` | Data-contract and grouping tests | Reject invalid memberships; preserve node IDs and edge direction; verify group-label permutation invariance and no-group controls. |
| `experiments/experiment_hierarchical_graph.py` | `main` | Complete synthetic data with known groups and cross-group exceptions; compare automatically discovered groups with truth and display both levels. |

Before selecting a clustering algorithm, define what a group means: dense internal interaction, similar dynamical response, and approximately closed aggregate dynamics are different criteria. Select and document one initial criterion, automatic resolution rule, scaling, and common-factor treatment; do not conflate these definitions. Group-selection tuning uses training data and must not inspect future validation outcomes.

Late-R1 acceptance: no supplied labels needed; interpretable aggregate/constituent views; original cross-group edges retained; reproducible settings; stable behavior on clear grouped systems and honest diagnostics on weak/no-group systems. Report grouping cost separately. This prototype groups an already fitted graph and makes no inference-speedup claim. It is a bounded late-R1 deliverable; joint hierarchy and dynamic membership are not R1 requirements. If no grouping criterion passes the agreed checks, record the blocker and agree a scope revision instead of silently advertising automatic hierarchy.

#### I2. Early R2: Dynamic Membership and Group Identity

Depends on I1 and the initial rolling-graph baseline in F. Develop these together before adding more elaborate joint temporal penalties.

| Planned file | Type/function | Contract and objective |
| --- | --- | --- |
| `src/dynsample/core/hierarchy.py` | `HierarchyTimeline`, `GroupEvent`, `at` | Store ordered snapshots and lineage events: continuation, membership transfer, birth/death, split, and merge. Snapshot membership remains immutable; changing group membership never changes base-node IDs. |
| `src/dynsample/inference/grouping.py` | `update_groups`, `_match_group_ids` | Match groups across windows, avoid treating label permutations as changes, and use temporal regularization/hysteresis to limit noise-driven reassignment. Allow split/merge rather than forcing one-to-one matches. |
| `src/dynsample/metrics/hierarchy.py` | `hierarchical_graph_metrics`, `group_tracking_metrics` | Evaluate within-/between-/cross-group recovery, partition agreement independent of labels, false group changes, and detection delay. |
| `tests/test_dynamic_hierarchy.py` | Temporal and lineage tests | Cover stable groups, genuine member transfers, split/merge, and noise-only changes without future leakage. |
| `experiments/experiment_dynamic_hierarchy.py` | `main` | Show group evolution and constituent changes with known truth; compare independent reclustering against temporal tracking. |

Update states frequently, relationships on a declared window schedule, and memberships less frequently or when accumulated evidence warrants it. Periodically audit cross-group relationships so an old partition cannot permanently hide new structure. Preserve full provenance for aggregation weights: a membership or weight change is not automatically a change in physical group dynamics. Retrospective smoothing and online tracking must be labeled separately.

Early-R2 acceptance: preserve group identities under arbitrary label permutations, limit spurious changes in controls, detect actual changes with reported delay, and retain cross-group node exceptions. One changed constituent must not automatically trigger an entire group split.

#### I3. Early R2: Benchmark Computational Savings Separately

| Planned file | Function | Objective |
| --- | --- | --- |
| `src/dynsample/estimation/hierarchical.py` | `fit_hierarchical_graph`, `_screen_cross_group_candidates` | Prototype local fits plus a group-level model with selected cross-group corrections. Use inexpensive dynamical summaries or cached structure for candidate grouping; do not require a full dense fit on every update. |
| `tests/test_hierarchical_estimation.py` | Reference and exception tests | Compare with small full-system fits, expose omitted cross-group effects, and reject unsupported aggregation assumptions. |
| `experiments/benchmark_hierarchical_scaling.py` | `main` | Compare wall time, peak memory, prediction quality, edge recovery, grouping overhead, and full-refresh cost across node/feature/sample counts. |

For G groups of n scalar nodes, separate dense within-group and group-level drift blocks have roughly G*n²+G² coefficients, before cross-level and exception parameters, compared with G²*n² for an unrestricted full drift. This is a parameter-count illustration, not runtime complexity. Dense matrix exponentials and covariance propagation may erase savings; avoiding the full-system computation requires a justified approximation or special structure. Maintain a small exact reference and report approximation error. Promote a fast mode only if measurements establish useful savings at a declared accuracy cost; otherwise retain the hierarchy as an inspection feature and report the performance blocker.

The intended relationship views are within-group node links, between-group relations, cross-group node exceptions, and explicit membership/aggregation links. Joint dynamical couplings across levels require additional identifiability work. Within-node feature dynamics belongs to the multi-feature extension and must not be confused with grouping nodes.

#### I4. R3/R4: Partial Observations and Validated Multilevel Uncertainty

In R3, extend coverage and grouping diagnostics to partially observed nodes. Missingness must not silently change aggregate weights or force group disappearance. In R4, evaluate uncertainty and selection stability through the complete grouping-and-fitting pipeline, including resolution choice and lineage uncertainty. Account for dependent alerts across levels rather than treating parent and constituent evidence as independent.

Mathematical checkpoint: aggregates are derived observations. A joint group-factor/node-residual model needs identifiable scale, rotation, and coupling conventions. Arbitrary aggregation need not preserve Markov dynamics. For Z=P X with constant diffusion, P K=A P is a sufficient drift-closure condition; otherwise quantify approximation error or add state. Changes in membership, weights, noise, and true dynamics must be distinguished. No precise causal attribution or complete cross-level identification is promised without supporting evidence.

## 9. Outside the Current Development Sequence

- Nonlinear, jump, or event-driven models; changing base-node identities; semantic intention inference. Automatic structural grouping is covered by I; discovered groups do not automatically carry domain meaning.
- Standalone domain products. Domain-specific terminology and workflows do not belong in the core hierarchy model.
- C++, GPU, distributed execution, and a generic plugin system without profiling evidence.
- A standalone GUI product; current visualization uses experiment scripts.
- Interpreting “no required bounds” as “no modeling assumptions.”

## 10. Execution and Change Log

Current checkpoint: v023 contains unified automatic/bounded profile fitting. The working tree adds flexible joint inputs, top-level exports, and the eight-combination experiment for the intended v024 commit. Scalar closeout is at A3; graph inference is not implemented. Follow dependencies rather than skipping mathematical checks to meet dates. Do not promise exact research-function counts or completion dates without supporting evidence.

| Date | Task | Status/evidence | Next step |
| --- | --- | --- | --- |
| 2026-09-30 | Planning baseline | Inventory checked against v019, actual functions, and README; documentation-only changes | A1: document search rules and result states mathematically and algorithmically, then implement `_initial_ou_alpha_bounds` |
| 2026-10-01 | A1–A3 implementation checkpoint, intended v024 (not yet committed) | Flexible `fit_ou`, explicit `fit_ou_joint`, top-level exports, and interface experiment present. `.venv/bin/python -m pytest -q`: 219 passed. Headless interface experiment reproduced all eight successful fits and maximum NLL difference about 2.12e-8. | Finish the remaining A acceptance audit, then begin B matrix derivation; no new scalar-model scope. |

Documentation maintenance on 2026-09-30: replaced the Chinese plan with this English version and moved detailed scalar API, mathematical, and evaluation references from the README into the appendices. No source-code changes.

Scope update on 2026-09-30: added extension I for general known-group hierarchical and multiscale relationships. Domain-specific examples are not development requirements. The original update is superseded by the schedule revision below; joint cross-level inference remains gated by mathematical validation.

Schedule revision on 2026-09-30: the user requested automatic, dynamically discovered groups and earlier hierarchy support. Added I1 to late R1 and I2–I3 to early R2, with explicit structures, file/function responsibilities, visualization checks, and measured computational tradeoffs. A–D remain unchanged. Known groups are references, not required user input. Missing-observation and uncertainty extensions remain in R3/R4. No source implementation was added.

After each task, append its commit or files, test commands and results, experiment conclusions, and unresolved issues. Assign additions to a stage and explain changes in stage order so the implementation plan does not drift through conversation.


## Appendix A. Current Scalar API and Experiment Reference

Moved from the README to keep implementation details in one place. This appendix describes the current working-tree API, including the interface changes intended for v024. Run commands from the repository root. The scalar `trajectory` used in the profile snippet must already exist.

### Scalar Brownian Drift Estimation

`fit_brownian_drift` in `dynsample.estimation.brownian` fits the model `dX = b dt + sigma dW` by closed-form conditional maximum likelihood. It assumes complete, exact scalar observations of shape `(T, 1, 1)`, constant drift and volatility, and at least three time points. Unequal time intervals are supported; the likelihood conditions on the initial observation.

```python
import numpy as np

from dynsample.core.trajectory import Trajectory
from dynsample.estimation.brownian import fit_brownian_drift

trajectory = Trajectory(
    times=np.array([2.0, 3.0, 5.0]),
    values=np.array([1.0, 4.0, 4.0])[:, None, None],
)
result = fit_brownian_drift(trajectory)
print(result.drift)       # 1.0
print(result.volatility)  # 1.7320508075688772
print(result.fun)         # Conditional negative log-likelihood
```

For `n = T - 1` increments, the estimates are:

```math
\widehat b = \frac{x_{T-1}-x_0}{t_{T-1}-t_0},\qquad
\widehat\sigma^2 = \frac{1}{n}\sum_{k=1}^{n}
\frac{(x_k-x_{k-1}-\widehat b\,\Delta t_k)^2}{\Delta t_k}.
```

The returned SciPy `OptimizeResult` exposes `drift`, `volatility`, `x = [drift, volatility]` in physical units, and `fun`. Its `success=True` indicates completion of the closed-form calculation, not an iterative optimization or a guarantee of parameter accuracy. The variance estimate uses the maximum-likelihood denominator `n`, without a degrees-of-freedom correction. Zero residual variance is rejected because it has no positive-volatility interior maximum.

This estimator does not add a drift argument to `simulate_brownian`, which still simulates zero-drift Brownian motion. It does not infer graphs or provide parameter intervals. Tests compare regular and irregular examples with hand calculations and an independent Gaussian-density optimization, and check rejected degenerate inputs.

### Scalar OU Fitting and Experiment

OU estimators use complete, exact scalar observations `(T,1,1)`, constant parameters, and a likelihood conditional on the initial observation. Unequal time intervals are supported. They do not estimate graph relationships, handle measurement noise, or reconstruct missing states. Profile scanning requires at least three observations; joint likelihood evaluation permits two but such a small sample does not generally support reliable estimation of three parameters.

```python
import dynsample as ds

# trajectory is an existing complete scalar Trajectory.
result = ds.fit_ou(trajectory)
print(result.mean_reversion, result.long_run_mean, result.volatility)
print(result.method, result.success, result.message)
```

#### Inputs and method selection

| initial_parameters | parameter_bounds | alpha_bounds | Method |
| --- | --- | --- | --- |
| omitted | omitted | omitted | Automatic profile search |
| supplied | omitted | omitted | Joint optimization without explicit bounds |
| omitted | supplied | omitted | Joint optimization with an automatic start |
| omitted | omitted | supplied | Profile search within supplied bounds |
| supplied | supplied | omitted | Joint optimization with supplied start and bounds |
| supplied | omitted | supplied | Joint optimization with only alpha constrained |
| omitted | supplied | supplied | Joint optimization with an automatic start and intersected alpha ranges |
| supplied | supplied | supplied | Joint optimization with supplied start and intersected alpha ranges |

`initial_parameters` is `None` or a finite `(alpha, mu, sigma)` tuple, with alpha and sigma strictly positive. It supplies a starting guess, not fixed parameters. Partial tuples containing `None` are not supported.

`parameter_bounds` is `None` or three `(lower, upper)` pairs in alpha, mu, sigma order. Bounds must be ordered and contain no NaN. Infinite endpoints are supported: mu may be unbounded, and zero lower bounds for alpha/sigma map to negative infinity in log coordinates while physical fitted parameters remain positive. Negative alpha/sigma lower bounds are rejected. Individual pairs or endpoints expressed as `None` are not supported.

`alpha_bounds` is keyword-only and, when supplied, contains two finite values with `0 < lower < upper`. If both bounds options are present, use their alpha-range intersection; no interval in common is an error. Automatic initial values are clipped into supplied joint bounds. Explicit initial values outside the effective bounds are rejected. `initial_was_adjusted` records automatic adjustment on joint results.

```python
result = ds.fit_ou(
    trajectory,
    initial_parameters=(0.7, 10.0, 1.5),
    parameter_bounds=((0.01, 5.0), (-20.0, 20.0), (0.1, 5.0)),
    alpha_bounds=(0.1, 2.0),
)
```

Direct `ds.fit_ou_profile(trajectory, alpha_bounds=None)` selects profile fitting. Direct `ds.fit_ou_joint(trajectory, initial_parameters=None, parameter_bounds=None)` selects joint fitting. Supplying initial values to the dispatcher changes the algorithm, so results need not be identical on every dataset. Joint optimization is local and may fail with extreme unbounded trial parameters; optional inputs do not imply a global-optimum guarantee or automatic multiple starts.

#### Return values

Both paths expose `mean_reversion`, `long_run_mean`, `volatility`, `fun`, `success`, and `message`. The dispatcher adds `method` (`profile` or `joint`). Use named physical parameters rather than assuming a common `x` shape:

- Profile: `x` is physical scalar alpha; `log_alpha` records its log coordinate.
- Joint: `x` retains `(log(alpha), mu, log(sigma))` for compatibility. `initial_parameters` records the actual start and `initial_was_adjusted` reports clipping of an automatic start.

Profile results include `alpha_bounds`, `searched_alpha_bounds`, `alpha_search_position`, `boundary_fraction`, `near_lower_bound`, and `near_upper_bound`. Position is normalized within the log-alpha interval; the proximity threshold is 1% of its width. For automatic searches, `alpha_bounds` belongs to the retained best candidate's search round, whereas `searched_alpha_bounds` is the final explored range. These ranges can differ.

#### Automatic search and stopping states

Automatic profile search starts from `0.01/T` and `-log(0.01)/min(dt)`. It scans 65 log-spaced points per round, refines candidate intervals, and expands eligible sides by a factor of ten, for at most six expansions. These are internal defaults, not statistical confidence bounds. An absolute NLL tolerance of `1e-6` is used for comparison with boundary limits; it is not a significance threshold or a proven error bound. Explicit profile bounds are never expanded.

| Profile status | Meaning |
| --- | --- |
| `interior_candidate` | Search completed with an interior candidate under the current stopping rules. |
| `bounded_candidate` | Explicit-range search completed near a supplied boundary. |
| `boundary_limit` | Automatic search stopped near a boundary with a numerically similar limit score. |
| `incomplete_search` | Some evaluations/refinements failed; a finite candidate is retained. |
| `expansion_limit` | Expansion budget exhausted. |
| `numerical_limit` | Further range expansion would exceed usable floating-point values. |

Automatic results set `success=True` only for `interior_candidate`. Manual profile results can report success for `bounded_candidate`; success means the search completed within the supplied constraints. Joint `status` and `message` retain SciPy optimizer semantics. None of these establishes parameter accuracy or calibrated uncertainty.

Automatic profile results retain `history`, `best_search`, `boundary_scores`, `expansions`, and diagnostics. A boundary NLL gap is `boundary NLL - candidate NLL`: negative means the boundary reference scores better. Boundary references are limiting models, not finite OU parameter estimates. Diagnostics attached to the best round do not override the overall stopping state. Degenerate boundary variances raise errors, and an all-failed scan raises `RuntimeError`; not every failure is returned as a result object.

#### Experiments and interpretation

```bash
python -m pytest tests/estimation/test_ou_profile_fit.py tests/estimation/test_ou_estimation.py -q
python -m experiments.experiment_ou_interface
python -m experiments.experiment_ou_interface --no-show --save /tmp/ou_interface.png
python -m experiments.experiment_ou_profile
python -m experiments.experiment_ou_estimation
```

The interface experiment uses one seed-42 trajectory with 1,001 observations, alternating intervals 0.05/0.15, and duration 100. It compares all eight input combinations, prints physical parameters, NLL differences and stopping messages, and plots three parameter comparisons plus NLL differences. The observed example returned approximately `(0.86458, 9.79848, 1.49371)` in every case, with maximum NLL difference about `2.12e-8`. This demonstrates numerical agreement on one dataset. It does not establish global optimality, repeated-sample accuracy, or agreement when bounds exclude the unconstrained optimum. Plot colors indicate reported optimizer success only.

The profile experiment compares restrictive and wider ranges and displays a log-alpha score curve. The estimation experiment plots fitted one-step means, in-sample standardized residuals, and new simulations under fixed fitted parameters. One-step agreement is not long-horizon forecast validation; residual variance near one is partly enforced by fitted noise scale. New simulations neither reconstruct the observed path nor include parameter uncertainty.

Independent regular-grid AR(1) MLE references, irregular-time likelihood checks, boundary regressions, input combinations, and failure handling are tested. Full-fit time-rescaling tests were added in v025. Repeated-seed recovery, held-out prediction, and search-setting sensitivity should not be claimed from a single passing experiment. Calibrated parameter intervals and graph estimation remain outside the current implementation.

## Appendix B. Mathematical and Modeling Reference

The following reference combines implemented scalar models with future graph models. Stage B starts with one feature per node and no external input; feature blocks and observed inputs below describe extensions, not additional prerequisites for scalar closeout or the first graph estimator.

### State, Graph, and Dynamical Conventions

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

### Mathematical Foundation

#### Stochastic Analysis and Exact Transitions

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

#### Brownian Reference Model

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

#### OU Reference Model

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

#### Probability, Bayesian Inference, and MCMC

Probability modeling is foundational; general-purpose MCMC is not an early development milestone.

- Conditional Gaussian state inference with known parameters can be computed analytically. Kalman filtering is a Bayesian state update under its assumptions, without MCMC.
- Early parameter and graph estimation will prioritize likelihoods, regularization, identifiability experiments, and optimization baselines.
- A conjugate Brownian variance calculation may serve as an optional correctness reference. It is not a required release gate.
- General parameter or graph-structure MCMC is deferred until the dynamical likelihood, graph estimator, and basic validation work reliably and a concrete uncertainty question warrants it.
- Later Bayesian work should first exploit model structure, such as integrating out linear Gaussian latent states with a filtering likelihood, before sampling large collections of latent variables.

State uncertainty conditional on fitted parameters does not include all parameter or structural uncertainty. Bayesian posteriors can be prior-sensitive or overconfident under model misspecification. Continuous shrinkage priors do not, by themselves, assign posterior probability to an exactly absent edge.

## Appendix C. Evaluation Principles Across Releases

### Evaluation and Release Gates

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

## Appendix D. Additional Research Design Notes

These notes preserve design considerations from the former README roadmap. The stage definitions above determine implementation order and required scope.

- Initial graph work assumes fixed node identities and small systems. Supported scale must be measured. Synthetic data supplies known ground truth; a successful domain example does not establish validity across a field.
- Later node-feature blocks may use group sparsity. Any scalar summary of a block must declare its aggregation rule. Observed external inputs require their own integration contract and common-driver controls; they do not eliminate hidden confounding.
- Distinguish penalized structure selection from fixed-structure refitting. Refitting may reduce shrinkage bias but does not resolve post-selection uncertainty.
- For R2, start with piecewise-constant graphs and keep retrospective inference distinct from online estimation. A candidate objective combines transition NLL, within-segment sparsity, and differences between neighboring drift matrices. Select proximal, alternating, or ADMM methods only after analyzing that actual objective; their names alone provide no convergence guarantee.
- R3 uses an observation model `Y_k = H_k x(t_k) + epsilon_k`, with Gaussian measurement covariance R_k. Declare assumptions about missingness; informative missingness needs a separate model.
- R4 should also assess uncertainty in change locations, prior sensitivity if Bayesian methods are added, and computational cost of repeated fitting. Claim error-rate control only under assumptions supported by the procedure and validation.
- Active observation design, unrestricted hierarchical models beyond extension I, adaptive path resolution, semantic community interpretation beyond structural grouping, and unrestricted continuously changing dynamics remain optional research beyond the required sequence.
- Earlier README calendar estimates were provisional and are not release commitments. Re-estimate effort after the first known-structure benchmark. Acceptance evidence, rather than a calendar target, determines release readiness.
