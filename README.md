# Dynsample: Stochastic Dynamics and Graph Inference

A Python research project for estimating dynamical relationships between nodes from finite time-series observations, with explicit assumptions and reliability evaluation.

The intended workflow is:

```text
Node time series + timestamps
    -> Estimate dynamical relationships
    -> Track changes in the graph
    -> Assess uncertainty, stability, and predictive value
```

**Current status:** scalar stochastic-model foundations are implemented. Unknown graph estimation, dynamic graphs, general missing-data inference, and calibrated reliability evaluation are planned, not implemented.

## Current Capabilities

| Component | Implemented behavior |
| --- | --- |
| Data structures | `State` `(N,d)`, `Trajectory` `(T,N,d)`, masked `Observation`, and supplied weighted `Graph`. |
| Brownian simulation | Independent increments with shared scalar volatility on irregular times. |
| OU simulation | Exact independent scalar transitions and trajectory simulation on irregular times. |
| Brownian bridge | Single-point and joint multi-point conditional sampling between supplied endpoints. |
| Brownian estimation | Closed-form scalar drift and volatility MLE. |
| OU estimation | Conditional NLL, automatic profile search, optional joint-fit starts/bounds, and boundary diagnostics through `ds.fit_ou`. |
| Validation examples | Analytical-reference tests, eight input-combination comparisons, and small-/large-alpha boundary experiments. |

`Graph` stores relationships; it does not learn them. Current estimators require complete, exact scalar observations of shape `(T,1,1)`. Multi-node simulation currently uses independent components. An observation mask alone does not provide missing-data inference.

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
```

Save the interface-comparison figure without opening a window:

```bash
python -m experiments.experiment_ou_interface --no-show --save /tmp/ou_interface.png
```

The interface experiment fits one irregular trajectory with eight combinations of optional inputs. Similar scores demonstrate agreement on that dataset, not universal recovery accuracy. `argparse` and `pathlib` are Python standard-library modules and need no extra dependency.

Plots of conditional state intervals describe process uncertainty under supplied parameters, not parameter uncertainty or simultaneous path coverage. Lines connecting sampled states are display connections, not inferred intermediate paths.

## Development Direction

The goal is a reusable scientific and engineering tool, starting with complete observations and static sparse linear dynamics. Missing-state reconstruction supports this goal. The project does not promise causal discovery, human-intention inference, automatic team discovery, or unrestricted nonlinear graph learning.

| Milestone | Planned objective |
| --- | --- |
| R1 | Static graph estimation from complete observations, with validation and a late-stage automatic grouping/aggregate-view prototype. |
| R2 | Piecewise-changing relationships, dynamic group tracking, and hierarchical computation benchmarks. |
| R3 | Noisy, asynchronous, and missing observations through a state-space model. |
| R4 | Validated intervals, selection stability, calibration, and sensitivity evaluation. |

The immediate sequence is scalar OU closeout -> matrix transitions -> known-structure estimation -> unknown static graphs -> R1 validation. Basic diagnostics and validation apply throughout; MCMC is not an early milestone.

The [development execution plan](docs/DEVELOPMENT_PLAN.md) is the source of truth for stage dependencies, each file/function's responsibility, acceptance criteria, and deferred work. Its appendices contain the [mathematical reference](docs/DEVELOPMENT_PLAN.md#appendix-b-mathematical-and-modeling-reference) and [evaluation principles](docs/DEVELOPMENT_PLAN.md#appendix-c-evaluation-principles-across-releases). Planned APIs are explicitly distinguished from existing functionality.

## Current Limitations

- No graph fitting or calibrated parameter intervals yet. Automatic profile search is heuristic and budget-limited; unbounded joint optimization can still fail numerically.
- Adjacency uses source-to-target indexing: `adjacency[i,j]` means i→j. Planned drift K[i,j] acts from j→i; conversion must preserve this distinction.
- Zero-volatility Brownian bridges currently interpolate even incompatible distinct endpoints. This behavior needs correction before release; use positive volatility for stochastic bridge examples.
- Clean-environment installation verification and the remaining data-contract checks are release tasks recorded in the plan.

## Repository Layout

```text
src/dynsample/
    core/                       # State, Trajectory, Observation, Graph
    simulation/                 # Independent Brownian and scalar OU models
    estimation/                 # Scalar Brownian and OU fitting
    inference/reconstruction/   # Brownian bridge sampling
    metrics/                    # Placeholder
    sampling/                   # Placeholder
experiments/                    # Reproducible scripts and plots
tests/                          # Analytical and numerical-reference tests
docs/DEVELOPMENT_PLAN.md        # Execution plan and detailed references
pyproject.toml                  # Metadata and dependency extras
requirements.txt                # Editable installation with dev/plot extras
```
