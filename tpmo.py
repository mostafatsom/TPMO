"""TPMO — Two-Phase Metamodel-Driven Optimization.

Problem-independent implementation of the two-phase surrogate-assisted
optimization methodology introduced by Ilchi Ghazaan and Sharifi (2025).

The implementation is intentionally kept in one file, following the layout of
the standalone FSGO repository. FSGO itself is not reimplemented here; the
implementation imports ``minimize_fsgo`` from the upstream ``mostafatsom/FSGO``
repository at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
from pathlib import Path
from typing import Literal, Protocol, Sequence

import numpy as np
from scipy.stats import qmc
import xgboost as xgb

__version__ = "1.0.0"

Array = np.ndarray
Sense = Literal["le", "ge", "eq"]


@dataclass(frozen=True)
class ConstraintSpec:
    """One scalar real constraint."""

    name: str
    sense: Sense
    limit: float


@dataclass
class Evaluation:
    """Result returned by the expensive problem evaluator."""

    objective: float
    constraints: Array
    valid: bool = True
    metadata: dict | None = None


class OptimizationProblem(Protocol):
    """Minimal problem interface required by TPMO."""

    @property
    def dimension(self) -> int: ...

    @property
    def bounds(self) -> Sequence[tuple[float, float]]: ...

    @property
    def constraints(self) -> Sequence[ConstraintSpec]: ...

    def evaluate(self, x: Array) -> Evaluation: ...

    def inactive_mask(self, x: Array) -> Array:
        """Return True for variables removed before Phase II."""
        ...

    def phase2_bounds(self, active_indices: Array) -> Sequence[tuple[float, float]]:
        """Return reduced-space bounds for retained Phase-II variables."""
        ...


@dataclass(frozen=True)
class XGBoostConfig:
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 1.0
    colsample_bytree: float = 1.0


@dataclass(frozen=True)
class GradientConfig:
    """Numerical bridge from XGBoost to the gradient required by FSGO."""

    relative_step: float = 0.02


@dataclass(frozen=True)
class FSGOConfig:
    population_size: int = 30
    K: int = 15000
    gamma_set: tuple[float, ...] | None = None
    precision_step: float = 1e-4


@dataclass(frozen=True)
class TPMOConfig:
    seed: int = 1
    initial_samples: int = 100
    alpha: float = 0.01
    ec_tolerance: float = 1e-6
    phase1_max_iterations: int = 5000
    phase2_stagnation: int = 100
    phase2_relative_tolerance: float = 1e-4
    phase2_max_iterations: int = 5000
    invalid_penalty: float = 1e12
    violation_penalty: float = 1e6
    record_history: bool = True
    xgboost: XGBoostConfig = XGBoostConfig()
    gradient: GradientConfig = GradientConfig()
    fsgo: FSGOConfig = FSGOConfig()


@dataclass
class TPMOResult:
    x: Array
    evaluation: Evaluation
    evaluations: int
    phase1_iterations: int
    phase2_iterations: int
    active_indices: Array
    history: list[dict]

    @property
    def fun(self) -> float:
        return float(self.evaluation.objective)


_REQUIRED_FSGO_PARAMETERS = {
    "fun",
    "grad",
    "bounds",
    "mode",
    "gamma_set",
    "population_size",
    "K",
    "x0",
    "seed",
    "precision_step",
    "record_history",
}


def _load_upstream_fsgo():
    """Import ``minimize_fsgo`` from the standalone upstream repository."""

    try:
        import fsgo  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "FSGO is not installed. Install the upstream repository with "
            "`python -m pip install git+https://github.com/mostafatsom/FSGO.git`."
        ) from exc

    minimize = getattr(fsgo, "minimize_fsgo", None)
    if minimize is None or not callable(minimize):
        raise ImportError("Installed `fsgo` module does not expose minimize_fsgo().")

    signature = inspect.signature(minimize)
    missing = sorted(_REQUIRED_FSGO_PARAMETERS.difference(signature.parameters))
    if missing:
        raise ImportError(
            "Installed FSGO API is incompatible with TPMO; missing parameters: "
            + ", ".join(missing)
        )

    module_path = getattr(fsgo, "__file__", None)
    if module_path:
        here = Path(__file__).resolve().parent
        if (here / "pyproject.toml").is_file():
            try:
                Path(module_path).resolve().relative_to(here)
            except ValueError:
                pass
            else:
                raise ImportError(
                    "Refusing to use an fsgo module vendored inside this repository. "
                    "Install the upstream GitHub repository instead."
                )

    return fsgo, minimize


def upstream_fsgo_info() -> dict:
    """Return runtime metadata for the imported FSGO package."""

    module, minimize = _load_upstream_fsgo()
    return {
        "module": module.__name__,
        "file": getattr(module, "__file__", None),
        "version": getattr(module, "__version__", None),
        "signature": str(inspect.signature(minimize)),
    }


def acceptable_margin(iteration: int, alpha: float) -> float:
    return 0.1 * math.exp(-float(alpha) * int(iteration))


def _extensive_limits(specs: Sequence[ConstraintSpec], am: float):
    out = []
    for spec in specs:
        limit = float(spec.limit)
        if spec.sense == "le":
            out.append(("le", limit * (1.0 + am), None))
        elif spec.sense == "ge":
            out.append(("ge", limit * (1.0 - am), None))
        elif spec.sense == "eq":
            out.append(("eq", limit * (1.0 - am), limit * (1.0 + am)))
        else:
            raise ValueError(f"Unsupported constraint sense: {spec.sense}")
    return out


def normalized_violation(values: Array, specs: Sequence[ConstraintSpec], am: float) -> Array:
    values = np.asarray(values, dtype=float)
    limits = _extensive_limits(specs, float(am))
    if values.shape[-1] != len(limits):
        raise ValueError("constraint vector length does not match ConstraintSpec list")

    violation = np.zeros_like(values, dtype=float)
    for j, (sense, lower_or_upper, upper) in enumerate(limits):
        scale = max(abs(float(specs[j].limit)), 1.0)
        if sense == "le":
            violation[..., j] = np.maximum(
                0.0, (values[..., j] - lower_or_upper) / scale
            )
        elif sense == "ge":
            violation[..., j] = np.maximum(
                0.0, (lower_or_upper - values[..., j]) / scale
            )
        else:
            low = float(lower_or_upper)
            high = float(upper)
            violation[..., j] = (
                np.maximum(
                    np.maximum(low - values[..., j], values[..., j] - high),
                    0.0,
                )
                / scale
            )
    return violation


class _ResponseSurrogate:
    """One XGBoost model for the objective and one for each constraint."""

    def __init__(
        self,
        bounds: Array,
        n_constraints: int,
        cfg: XGBoostConfig,
        grad_cfg: GradientConfig,
        seed: int,
    ):
        self.bounds = np.asarray(bounds, dtype=float)
        self.n_constraints = int(n_constraints)
        self.cfg = cfg
        self.grad_cfg = grad_cfg
        self.seed = int(seed)
        self.models = []

    def _train_one(self, X, y, seed):
        dm = xgb.DMatrix(np.asarray(X, dtype=float), label=np.asarray(y, dtype=float), nthread=1)
        params = {
            "objective": "reg:squarederror",
            "max_depth": int(self.cfg.max_depth),
            "eta": float(self.cfg.learning_rate),
            "subsample": float(self.cfg.subsample),
            "colsample_bytree": float(self.cfg.colsample_bytree),
            "tree_method": "hist",
            "nthread": 1,
            "verbosity": 0,
            "seed": int(seed),
        }
        return xgb.train(
            params,
            dm,
            num_boost_round=int(self.cfg.n_estimators),
            verbose_eval=False,
        )

    def fit(self, X, objective, constraints):
        X = np.asarray(X, dtype=float)
        objective = np.asarray(objective, dtype=float)
        constraints = np.asarray(constraints, dtype=float)
        if constraints.ndim != 2 or constraints.shape[1] != self.n_constraints:
            raise ValueError("constraints matrix has wrong shape")

        self.models = [self._train_one(X, objective, self.seed)]
        for j in range(self.n_constraints):
            self.models.append(
                self._train_one(X, constraints[:, j], self.seed + 1 + j)
            )
        return self

    @staticmethod
    def _predict_one(model, X):
        return np.asarray(
            model.inplace_predict(np.atleast_2d(np.asarray(X, dtype=float))),
            dtype=float,
        )

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        objective = self._predict_one(self.models[0], X)
        if self.n_constraints:
            constraints = np.column_stack(
                [self._predict_one(model, X) for model in self.models[1:]]
            )
        else:
            constraints = np.empty((len(X), 0), dtype=float)
        return objective, constraints

    def finite_difference_gradient(self, scalar_fun, x):
        x = np.asarray(x, dtype=float)
        dimension = len(x)
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        step = np.maximum(
            (upper - lower) * float(self.grad_cfg.relative_step), 1e-12
        )

        queries = np.repeat(x[None, :], 2 * dimension, axis=0)
        denominator = np.zeros(dimension, dtype=float)
        for j in range(dimension):
            xp = min(upper[j], x[j] + step[j])
            xm = max(lower[j], x[j] - step[j])
            queries[2 * j, j] = xp
            queries[2 * j + 1, j] = xm
            denominator[j] = xp - xm

        values = np.asarray(scalar_fun(queries), dtype=float).reshape(-1)
        gradient = np.zeros(dimension, dtype=float)
        valid = denominator > 1e-15
        gradient[valid] = (
            values[0::2][valid] - values[1::2][valid]
        ) / denominator[valid]
        gradient[~np.isfinite(gradient)] = 0.0
        return gradient


class _TwoPhaseOptimizer:
    def __init__(self, problem: OptimizationProblem, config: TPMOConfig):
        self.problem = problem
        self.cfg = config
        self.X: list[Array] = []
        self.E: list[Evaluation] = []
        self.history: list[dict] = []
        self.bounds = np.asarray(problem.bounds, dtype=float)
        if self.bounds.shape != (problem.dimension, 2):
            raise ValueError("problem.bounds shape must be (dimension, 2)")
        if np.any(~np.isfinite(self.bounds)) or np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError("all bounds must be finite and satisfy lower < upper")

    @property
    def evaluation_count(self) -> int:
        return len(self.E)

    def _record(self, item: dict):
        if self.cfg.record_history:
            self.history.append(item)

    def _evaluate(self, x):
        x = np.asarray(x, dtype=float)
        evaluation = self.problem.evaluate(x)
        constraints = np.asarray(evaluation.constraints, dtype=float).reshape(-1)
        if len(constraints) != len(self.problem.constraints):
            raise ValueError("problem returned a constraint vector with wrong length")
        if not np.isfinite(float(evaluation.objective)) and evaluation.valid:
            raise ValueError("valid evaluations must have a finite objective")
        self.X.append(x.copy())
        self.E.append(evaluation)
        return evaluation

    def _lhs_initial_design(self):
        n = int(self.cfg.initial_samples)
        if n <= 0:
            raise ValueError("initial_samples must be positive")
        sampler = qmc.LatinHypercube(d=self.problem.dimension, seed=int(self.cfg.seed))
        unit = sampler.random(n)
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        for row in unit:
            self._evaluate(lower + row * (upper - lower))

    def _training_arrays(self, active=None):
        X = np.vstack(self.X)
        if active is not None:
            X = X[:, active]
        objective = np.asarray(
            [e.objective if e.valid else self.cfg.invalid_penalty for e in self.E],
            dtype=float,
        )
        if len(self.problem.constraints):
            constraints = np.vstack(
                [np.asarray(e.constraints, dtype=float).reshape(1, -1) for e in self.E]
            )
        else:
            constraints = np.empty((len(self.E), 0), dtype=float)
        return X, objective, constraints

    def _fit(self, bounds, active=None, seed_offset=0):
        X, objective, constraints = self._training_arrays(active)
        return _ResponseSurrogate(
            bounds,
            len(self.problem.constraints),
            self.cfg.xgboost,
            self.cfg.gradient,
            self.cfg.seed + self.evaluation_count + seed_offset,
        ).fit(X, objective, constraints)

    def _predicted_merit(self, surrogate, X, am):
        objective, constraints = surrogate.predict(X)
        violation = normalized_violation(
            constraints, tuple(self.problem.constraints), am
        )
        return objective + float(self.cfg.violation_penalty) * np.sum(
            violation * violation, axis=1
        )

    def _true_merit(self, evaluation: Evaluation, am: float):
        if not evaluation.valid:
            return float(self.cfg.invalid_penalty)
        constraints = np.asarray(evaluation.constraints, dtype=float).reshape(1, -1)
        violation = normalized_violation(
            constraints, tuple(self.problem.constraints), am
        )[0]
        return float(evaluation.objective) + float(self.cfg.violation_penalty) * float(
            np.sum(violation * violation)
        )

    def _fsgo(self, surrogate, bounds, am, mode, x0=None, seed_offset=0):
        _, minimize_fsgo = _load_upstream_fsgo()

        def objective(x):
            return float(
                self._predicted_merit(
                    surrogate, np.asarray(x, dtype=float).reshape(1, -1), am
                )[0]
            )

        def gradient(x):
            return surrogate.finite_difference_gradient(
                lambda Q: self._predicted_merit(surrogate, Q, am),
                np.asarray(x, dtype=float),
            )

        return minimize_fsgo(
            objective,
            gradient,
            [tuple(map(float, b)) for b in np.asarray(bounds, dtype=float)],
            mode=mode,
            gamma_set=self.cfg.fsgo.gamma_set,
            population_size=self.cfg.fsgo.population_size,
            K=self.cfg.fsgo.K,
            x0=None if x0 is None else np.asarray(x0, dtype=float),
            seed=self.cfg.seed + self.evaluation_count + seed_offset,
            precision_step=self.cfg.fsgo.precision_step,
            record_history=False,
        )

    def _phase1(self):
        surrogate = self._fit(self.bounds)
        X = np.vstack(self.X)
        initial_merit = self._predicted_merit(surrogate, X, 0.1)
        best_index = int(np.argmin(initial_merit))
        z = self.X[best_index].copy()
        ez = self.E[best_index]

        mode = "GLOBAL"
        iteration = 0
        while iteration < self.cfg.phase1_max_iterations:
            am = acceptable_margin(iteration, self.cfg.alpha)
            if am <= self.cfg.ec_tolerance:
                break

            result = self._fsgo(
                surrogate,
                self.bounds,
                am,
                mode,
                z if mode == "LOCAL" else None,
                1000 + iteration,
            )
            candidate = np.asarray(result.x, dtype=float)
            candidate_evaluation = self._evaluate(candidate)
            surrogate = self._fit(self.bounds)

            improved = self._true_merit(
                candidate_evaluation, am
            ) < self._true_merit(ez, am)
            if improved:
                z = candidate.copy()
                ez = candidate_evaluation
            else:
                mode = "LOCAL" if mode == "GLOBAL" else "GLOBAL"

            self._record(
                {
                    "phase": 1,
                    "iteration": iteration,
                    "acceptable_margin": am,
                    "mode": mode,
                    "objective": float(ez.objective),
                    "evaluations": self.evaluation_count,
                }
            )
            iteration += 1

        inactive = np.asarray(self.problem.inactive_mask(z), dtype=bool)
        if inactive.shape != (self.problem.dimension,):
            raise ValueError("inactive_mask must have shape (dimension,)")
        active = np.flatnonzero(~inactive)
        return z, ez, active, iteration

    def _phase2(self, z, ez, active):
        if len(active) == 0:
            return z, ez, 0

        bounds = np.asarray(self.problem.phase2_bounds(active), dtype=float)
        if bounds.shape != (len(active), 2):
            raise ValueError("phase2_bounds returned the wrong shape")

        surrogate = self._fit(bounds, active=active, seed_offset=100000)
        y = np.asarray(z, dtype=float)[active].copy()
        previous = float(ez.objective)
        stagnation = 0
        iteration = 0
        current_evaluation = ez

        while (
            stagnation < self.cfg.phase2_stagnation
            and iteration < self.cfg.phase2_max_iterations
        ):
            result = self._fsgo(
                surrogate,
                bounds,
                0.0,
                "LOCAL",
                y,
                200000 + iteration,
            )
            y_new = np.asarray(result.x, dtype=float)
            full = np.asarray(z, dtype=float).copy()
            full[active] = y_new
            evaluation = self._evaluate(full)
            current = float(evaluation.objective)
            relative_change = abs(previous - current) / max(abs(previous), 1e-15)

            if relative_change < self.cfg.phase2_relative_tolerance:
                stagnation += 1
            else:
                stagnation = 0

            y = y_new
            previous = current
            current_evaluation = evaluation
            z = full
            iteration += 1
            self._record(
                {
                    "phase": 2,
                    "iteration": iteration,
                    "relative_change": relative_change,
                    "stagnation": stagnation,
                    "objective": current,
                    "evaluations": self.evaluation_count,
                }
            )

        return z, current_evaluation, iteration

    def run(self):
        self._lhs_initial_design()
        z, evaluation, active, phase1_iterations = self._phase1()
        z, evaluation, phase2_iterations = self._phase2(z, evaluation, active)
        return TPMOResult(
            x=np.asarray(z, dtype=float),
            evaluation=evaluation,
            evaluations=self.evaluation_count,
            phase1_iterations=phase1_iterations,
            phase2_iterations=phase2_iterations,
            active_indices=np.asarray(active, dtype=int),
            history=self.history.copy(),
        )


def minimize_tpmo(
    problem: OptimizationProblem,
    *,
    config: TPMOConfig | None = None,
) -> TPMOResult:
    """Run the Two-Phase Metamodel-Driven Optimization methodology on ``problem``.

    Parameters
    ----------
    problem:
        Object implementing the :class:`OptimizationProblem` protocol.
    config:
        Optional :class:`TPMOConfig`. Defaults to ``TPMOConfig()``.
    """

    return _TwoPhaseOptimizer(problem, config or TPMOConfig()).run()


__all__ = [
    "ConstraintSpec",
    "Evaluation",
    "OptimizationProblem",
    "XGBoostConfig",
    "GradientConfig",
    "FSGOConfig",
    "TPMOConfig",
    "TPMOResult",
    "acceptable_margin",
    "normalized_violation",
    "upstream_fsgo_info",
    "minimize_tpmo",
]
