# TPMO — Two-Phase Metamodel-Driven Optimization

[![Tests](https://github.com/mostafatsom/TPMO/actions/workflows/tests.yml/badge.svg)](https://github.com/mostafatsom/TPMO/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.22068%2Fijoce.2025.15.2.630-blue)](https://doi.org/10.22068/ijoce.2025.15.2.630)

TPMO is a standalone, problem-independent Python implementation of the two-phase metamodel-driven optimization methodology introduced in:

> M. Ilchi Ghazaan and M. Sharifi, “A Two-Phase Metamodel-Driven Approach for Topology and Size Optimization of Truss Structures,” *International Journal of Optimization in Civil Engineering*, 15(2), 181–201, 2025.  
> DOI: `10.22068/ijoce.2025.15.2.630`

The complete methodology is intentionally kept in one file: `tpmo.py`.

## Scope

TPMO implements the generic optimization methodology only:

- Latin Hypercube initial sampling
- XGBoost response metamodels
- Extensive Constraints (EC)
- sequential/adaptive expensive sampling
- GLOBAL/LOCAL FSGO orchestration
- Phase-I variable discovery
- inactive-variable pruning
- Phase-II reduced-space refinement

The structural case studies from the paper are intentionally excluded. This repository contains no 24-bar, 39-bar, or 72-bar finite-element models, no target topology, no target member areas, and no paper target objective values.

FSGO is maintained separately and is used as an upstream dependency:

`https://github.com/mostafatsom/FSGO`

No FSGO implementation or fallback is bundled in TPMO.

## Installation

Clone the repository and install it from the project root:

```bash
git clone https://github.com/mostafatsom/TPMO.git
cd TPMO
python -m pip install .
```

The installation pulls the standalone FSGO repository as a dependency.

## Quick start

```python
import numpy as np

from tpmo import Evaluation, TPMOConfig, minimize_tpmo


class SphereProblem:
    dimension = 5
    bounds = [(-5.0, 5.0)] * dimension
    constraints = []

    def evaluate(self, x):
        x = np.asarray(x, dtype=float)
        return Evaluation(
            objective=float(np.dot(x, x)),
            constraints=np.empty(0),
        )

    def inactive_mask(self, x):
        return np.zeros(self.dimension, dtype=bool)

    def phase2_bounds(self, active_indices):
        return [self.bounds[int(i)] for i in active_indices]


result = minimize_tpmo(
    SphereProblem(),
    config=TPMOConfig(seed=42),
)

print(result.fun)
print(result.x)
print(result.evaluations)
```

## Methodology

TPMO follows the two-stage workflow described in the associated paper:

1. Generate an initial design with Latin Hypercube Sampling.
2. Evaluate the expensive problem and train XGBoost response metamodels.
3. Start Phase I in GLOBAL FSGO mode under Extensive Constraints.
4. Search only the metamodel; evaluate only the selected candidate with the expensive model.
5. Append the new sample and retrain the metamodel.
6. Alternate GLOBAL and LOCAL FSGO modes when the expensive candidate does not improve the incumbent.
7. Identify and remove inactive variables after Phase I.
8. Retrain in the reduced space and perform LOCAL FSGO refinement in Phase II.
9. Stop Phase II after repeated relative-objective stagnation.

The paper requires `∇Ĝ` while using XGBoost. Because tree ensembles do not expose a conventional analytic input gradient, TPMO keeps this bridge explicit and uses central finite differences of the surrogate merit.

## Examples

| Example | Purpose | Command |
|---|---|---|
| `basic_problem.py` | Unconstrained problem | `python examples/basic_problem.py` |
| `constrained_problem.py` | Scalar constrained problem | `python examples/constrained_problem.py` |
| `inactive_variables.py` | Phase-I pruning and Phase-II refinement | `python examples/inactive_variables.py` |

## API

```python
minimize_tpmo(
    problem,
    *,
    config=None,
)
```

The problem object must provide:

| Member | Description |
|---|---|
| `dimension` | Number of design variables |
| `bounds` | Finite `(lower, upper)` pair for every variable |
| `constraints` | Sequence of `ConstraintSpec` objects |
| `evaluate(x)` | Expensive evaluation returning `Evaluation` |
| `inactive_mask(x)` | Boolean mask for variables removed before Phase II |
| `phase2_bounds(active_indices)` | Bounds for retained Phase-II variables |

`TPMOConfig` controls initial sampling, EC decay, stopping criteria, XGBoost parameters, the numerical gradient bridge, and FSGO settings.

## Result

`minimize_tpmo` returns a `TPMOResult`:

```text
x                    final design vector
fun                  final objective value
evaluation           complete final Evaluation object
evaluations          number of expensive evaluations
phase1_iterations    Phase-I adaptive iterations
phase2_iterations    Phase-II refinement iterations
active_indices       variables retained for Phase II
history              optimization history
```

## Comparison

A compact reproducible comparison is included on five generic functions that are unrelated to the structural case studies in the paper:

- Zakharov
- Elliptic
- Bent Cigar
- Styblinski–Tang
- Levy

Quick run:

```bash
python comparison.py --quick
```

Longer run:

```bash
python comparison.py --full
```

Results are written to `comparison_results.csv`. The comparison is provided as a reproducible software example, not as a claim of universal superiority.

## Testing

```bash
python -m unittest discover -s tests -v
```

GitHub Actions installs the upstream FSGO repository and runs the test suite on Python 3.10, 3.11, and 3.12.

To verify which FSGO installation TPMO is using:

```bash
python scripts/verify_upstream_fsgo.py
```

## FSGO dependency

TPMO uses [FSGO](https://github.com/mostafatsom/FSGO) as its optimization
search engine.

## Repository layout

```text
TPMO/
├── .github/workflows/tests.yml
├── examples/
├── scripts/
├── tests/
├── CITATION.cff
├── LICENSE
├── README.md
├── comparison.py
├── pyproject.toml
└── tpmo.py
```

## Related projects

- [FSGO](https://github.com/mostafatsom/FSGO): the standalone optimization search engine used by TPMO.
- [ANSO](https://github.com/mostafatsom/ANSO): Adaptive Neighborhood Surrogate Optimization for expensive black-box problems.

## Citation

GitHub reads citation metadata from `CITATION.cff`.

If TPMO is used in academic work, please cite the associated paper:

```bibtex
@article{ilchi2025two,
  title   = {A Two-Phase Metamodel-Driven Approach for Topology and Size Optimization of Truss Structures},
  author  = {Ilchi Ghazaan, M. and Sharifi, M.},
  journal = {International Journal of Optimization in Civil Engineering},
  volume  = {15},
  number  = {2},
  pages   = {181--201},
  year    = {2025},
  doi     = {10.22068/ijoce.2025.15.2.630}
}
```

## License

MIT License. See `LICENSE`.
