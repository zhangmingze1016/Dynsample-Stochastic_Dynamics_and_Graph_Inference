# Dynsample: Stochastic Dynamics and Graph Inference

A Python research project for estimating dynamical relationships between nodes from finite time-series observations, with explicit assumptions and reliability evaluation.

The intended workflow is:

```text
Node time series + timestamps
    -> Estimate node states and dynamical relationships
    -> Track state evolution and changes in relationships
    -> Assess reliability and visualize the results
```

**Current status:** scalar stochastic-model foundations, a validated `LinearSDE` coefficient container, and coupled linear transitions and simulation are implemented. Unknown graph estimation, dynamic graphs, general missing-data inference, and calibrated reliability evaluation are planned, not implemented.

## Current Capabilities

| Component | Implemented behavior |
| --- | --- |
| Data structures | `State` `(N,d)`, `Trajectory` `(T,N,d)`, masked `Observation`, and supplied weighted `Graph`. |
| Linear model specification | `LinearSDE` stores constant drift, offset, and diffusion arrays with shape and finite-value validation; fitting coupled dynamics remains planned. |
| Coupled linear simulation | Exact matrix transitions, single-step sampling, and irregular-time trajectories with one feature per node; singular positive-semidefinite simulation covariance is supported. |
| Brownian simulation | Independent increments with shared scalar volatility on irregular times. |
| OU simulation | Exact independent scalar transitions and trajectory simulation on irregular times. |
| Brownian bridge | Single-point and joint multi-point conditional sampling between supplied endpoints. |
| Brownian estimation | Closed-form scalar drift and volatility MLE. |
| OU estimation | Conditional NLL, automatic profile search, optional joint-fit starts/bounds, and boundary diagnostics through `ds.fit_ou`. |
| Validation examples | Analytical-reference tests, eight input-combination comparisons, and small-/large-alpha boundary experiments. |

`Graph` stores relationships; it does not learn them. Current estimators require complete, exact scalar observations of shape `(T,1,1)`. Brownian and scalar OU simulations use independent components; linear SDE simulation supports coupled nodes. An observation mask alone does not provide missing-data inference.

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

## Fit Scalar OU Parameters

For an existing complete scalar `trajectory` of shape `(T,1,1)`:

```python
import dynsample as ds

result = ds.fit_ou(trajectory)
print(result.mean_reversion, result.long_run_mean, result.volatility)
print(result.fun, result.success, result.message)
```

`ds.State`, `ds.Trajectory`, `ds.fit_ou`, `ds.fit_ou_profile`, and `ds.fit_ou_joint` are exported at package level. Simulation and Brownian bridge functions retain their module imports.

All three fitting options are independently optional and may be combined:

```python
result = ds.fit_ou(
    trajectory,
    initial_parameters=(0.7, 10.0, 1.5),
    parameter_bounds=((0.01, 5.0), (-20.0, 20.0), (0.1, 5.0)),
    alpha_bounds=(0.1, 2.0),
)
```

These numbers are examples, not universal defaults. Initial values are starting guesses, not fixed parameters. With neither initial parameters nor three-parameter bounds, the interface uses profile fitting; supplying either selects joint optimization. Two supplied alpha ranges are intersected. Conflicting ranges or explicit starting values outside the effective bounds are rejected. Individual `None` entries inside parameter tuples are not supported.

Prefer the named parameter fields above: profile `x` is physical alpha, whereas joint `x` contains `(log(alpha), mu, log(sigma))`. The dispatcher reports `method`. Inspect `success`, `message`, and profile boundary diagnostics before using a result; numerical convergence does not establish a global optimum, parameter accuracy, or confidence intervals.

See the [scalar API reference](docs/DEVELOPMENT_PLAN.md#scalar-ou-fitting-and-experiment) for all eight combinations, allowed bounds, stopping states, and limitations.

## Tests and Experiments

Run from the repository root after installation:

```bash
python -m pytest -q
python -m experiments.experiment_brownian_bridge
python -m experiments.experiment_ou
python -m experiments.experiment_ou_estimation
python -m experiments.experiment_ou_profile
python -m experiments.experiment_ou_interface
python -m experiments.experiment_ou_brownian_limit
python -m experiments.experiment_ou_large_alpha_limit
python -m experiments.experiment_linear
```

Save the interface-comparison figure without opening a window:

```bash
python -m experiments.experiment_ou_interface --no-show --save /tmp/ou_interface.png
```

The interface experiment fits one irregular trajectory with eight combinations of optional inputs. Similar scores demonstrate agreement on that dataset, not universal recovery accuracy. `argparse` and `pathlib` are Python standard-library modules and need no extra dependency.

Plots of conditional state intervals describe process uncertainty under supplied parameters, not parameter uncertainty or simultaneous path coverage. Lines connecting sampled states are display connections, not inferred intermediate paths.

See the [experiment record](docs/EXPERIMENTS.md) for reproducible settings, measured results, representative figures, and limits of each experiment.

## Development Direction

The goal is a reusable scientific and engineering tool for estimating relationships, evaluating their reliability, and tracking their changes. Start with complete observations and static sparse linear dynamics. Parameters are intermediate tools for these tasks; exact recovery of every coefficient or a unique underlying equation is not the product requirement. Missing-state reconstruction supports this goal.

### What a Connection Means

The initial model is `dX = (K X + b) dt + B dW`. For scalar node states, an off-diagonal entry `K[i,j] != 0` defines a direct model-based dynamical dependence from node j to node i. Diagonal entries describe self-dynamics. A positive or negative coefficient describes the direction of its contribution to drift while other states are held fixed; it is not a behavioral label or proof of causation.

An estimated nonzero coefficient alone is not sufficient evidence to display an edge. Planned graph estimation combines sparse selection, optimization diagnostics, held-out evaluation, and stability checks as those capabilities become available. An unselected edge means insufficient support under the chosen procedure, not proof of no relationship. Uncertain edges must remain distinguishable from absent ones; selection frequency is not automatically an edge-existence probability.

Direct dependence, common inputs, and shared random variation are distinct. Future extensions may include observed inputs through `C U(t)` and separate analysis of the noise covariance rate `Q = B B.T`. Correlation and lagged association can result from these structures or indirect paths; they should not all be interpreted as direct edges. Unknown common factors require additional identification assumptions.

### Features, States, and Propagation

Future multi-feature inference will use a node-to-feature mapping and matrix blocks: `K[i,j]` then represents all feature channels from node j to node i. Only some channels may be supported, and their signs can differ. Preserve the full blocks and feature units even when the visualization shows one summarized edge. This extends the current scalar-node inference target; it is not implemented by the coefficient container alone.

Keep three phenomena separate:

- **State evolution:** node values change, even when relationships stay fixed.
- **Propagation:** a disturbance travels through direct and indirect paths in a fixed graph. For a fixed linear model, `exp(K * tau)` describes the response to an initial-state perturbation after elapsed time `tau`; it is not the direct-edge matrix.
- **Relationship change:** the estimated interaction structure changes across time, beyond what estimation variability can explain.

Node states and relationships must therefore be displayed together. Feature and node importance are future, task-dependent evaluations, with an explicit prediction horizon and treatment of redundant information. Raw coefficients in different units are not comparable importance scores. Group membership, aggregate views, and group dynamics must also remain distinct; grouping alone does not establish a valid reduced dynamical model or a computational speedup.

### Simple User Workflow

The intended interface is **provide timestamped node data -> run an analysis -> inspect a dynamic view**. Users should not need to prescribe behaviors, supply the true graph, or choose an optimizer before starting. Model assumptions and the meaning of each relationship layer must still be explicit. Use validated defaults, expose advanced controls separately, and ask only for necessary data interpretation such as timestamps, identities, and feature units.

Planned visualization includes a timeline, node states, relationship strengths, uncertainty indicators, change summaries, and optional feature-level details. Use observed coordinates when available; otherwise maintain a stable layout so layout motion is not mistaken for propagation. Distinguish observations, reconstructed states, inferred edges, and simulated trajectories. No unified analysis interface or interactive graph viewer is implemented yet.

### Scope and Validation

Relationship estimation and change detection are the main objectives. Predicting node states conditional on a fitted model is distinct from predicting future relationships; the latter requires a separate evolution model and remains optional research. The project does not promise unrestricted equation discovery, causal identification, or automatic behavioral interpretation.

Validate graph recovery on synthetic systems with hidden ground-truth edges: independent nodes, one-way and reciprocal effects, common drivers, indirect chains, selected feature channels, propagation on fixed graphs, and changing connections. Vary seeds, observation duration, sampling intervals, noise, and effect sizes. Evaluate false and missed edges, stability, held-out predictive value, and, for changing graphs, false alarms and detection delay. A convincing animation or optimizer success flag is not sufficient evidence.

Extend model expressiveness incrementally: static linear dynamics first, then observed common inputs and piecewise-changing relationships, followed by selected nonlinear interaction functions when experiments justify them. Candidate functions must avoid redundant parameterizations. Reliability checks begin with the first estimator; calibrated uncertainty, imperfect observations, and more flexible models require their own validation.

| Milestone | Planned objective |
| --- | --- |
| R1 | Static graph estimation from complete observations, with validation and a late-stage automatic grouping/aggregate-view prototype. |
| R2 | Piecewise-changing relationships, dynamic group tracking, and hierarchical computation benchmarks. |
| R3 | Noisy, asynchronous, and missing observations through a state-space model. |
| R4 | Validated intervals, selection stability, calibration, and sensitivity evaluation. |

The next step is drift-to-graph conversion and direction checks, followed by known-structure estimation, unknown static graphs, and R1 validation. Coupled transitions and simulation are implemented. Basic diagnostics and validation apply throughout; MCMC is not an early milestone.

The [development execution plan](docs/DEVELOPMENT_PLAN.md) is the source of truth for stage dependencies, each file/function's responsibility, acceptance criteria, and deferred work. Its appendices contain the [mathematical reference](docs/DEVELOPMENT_PLAN.md#appendix-b-mathematical-and-modeling-reference) and [evaluation principles](docs/DEVELOPMENT_PLAN.md#appendix-c-evaluation-principles-across-releases). Planned APIs are explicitly distinguished from existing functionality.

## Current Limitations

- No graph fitting or calibrated parameter intervals yet. Automatic profile search is heuristic and budget-limited; unbounded joint optimization can still fail numerically.
- Adjacency uses source-to-target indexing: `adjacency[i,j]` means i→j. Planned drift K[i,j] acts from j→i; conversion must preserve this distinction.
- Zero-volatility Brownian bridges currently interpolate even incompatible distinct endpoints. This behavior needs correction before release; use positive volatility for stochastic bridge examples.
- Clean-environment installation verification and the remaining data-contract checks are release tasks recorded in the plan.

## Repository Layout

```text
src/dynsample/
    core/                       # State, Trajectory, Observation, Graph, LinearSDE
    simulation/                 # Brownian, scalar OU, and coupled linear SDE simulation
    estimation/                 # Scalar Brownian and OU fitting
    inference/reconstruction/   # Brownian bridge sampling
    metrics/                    # Placeholder
    sampling/                   # Placeholder
experiments/                    # Reproducible scripts and plots
tests/                          # Analytical and numerical-reference tests
docs/EXPERIMENTS.md             # Reproducible experiment results and figures
docs/images/                    # Representative experiment figures
docs/DEVELOPMENT_PLAN.md        # Execution plan and detailed references
pyproject.toml                  # Metadata and dependency extras
requirements.txt                # Editable installation with dev/plot extras
```
