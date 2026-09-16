"""Small reproducible TPMO comparison on generic non-paper functions.

The script reports the final TPMO objective and a same-seed initial-LHS baseline.
It is a software validation example, not a claim of universal superiority.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace

import numpy as np
from scipy.stats import qmc

from tpmo import (
    Evaluation,
    FSGOConfig,
    GradientConfig,
    TPMOConfig,
    XGBoostConfig,
    minimize_tpmo,
)

DIMENSION = 5
SHIFT = np.array([1.25, -1.10, 0.85, -0.65, 1.45], dtype=float)


def zakharov(z):
    z = np.asarray(z, dtype=float)
    t = np.sum(0.5 * np.arange(1, len(z) + 1) * z)
    return float(np.sum(z * z) + t * t + t**4)


def elliptic(z):
    z = np.asarray(z, dtype=float)
    a = np.arange(len(z)) / max(len(z) - 1, 1)
    return float(np.sum((1e6**a) * z * z))


def bent_cigar(z):
    z = np.asarray(z, dtype=float)
    return float(z[0] ** 2 + 1e6 * np.sum(z[1:] ** 2))


def levy(z):
    z = np.asarray(z, dtype=float)
    x = z + 1.0
    w = 1.0 + (x - 1.0) / 4.0
    value = np.sin(np.pi * w[0]) ** 2
    if len(x) > 1:
        value += np.sum(
            (w[:-1] - 1.0) ** 2
            * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + 1.0) ** 2)
        )
    value += (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[-1]) ** 2)
    return float(value)


def styblinski_tang(z):
    z = np.asarray(z, dtype=float) - 2.903534
    value = 0.5 * np.sum(z**4 - 16.0 * z**2 + 5.0 * z)
    optimum = -39.16616570377142 * len(z)
    return float(value - optimum)


FUNCTIONS = {
    "Zakharov": (zakharov, (-5.0, 10.0)),
    "Elliptic": (elliptic, (-5.0, 5.0)),
    "BentCigar": (bent_cigar, (-5.0, 5.0)),
    "StyblinskiTang": (styblinski_tang, (-5.0, 5.0)),
    "Levy": (levy, (-10.0, 10.0)),
}


class ShiftedBenchmark:
    constraints = []
    dimension = DIMENSION

    def __init__(self, name):
        self.name = name
        self.fun, domain = FUNCTIONS[name]
        self.bounds = [domain] * self.dimension

    def evaluate(self, x):
        z = np.asarray(x, dtype=float) - SHIFT
        return Evaluation(self.fun(z), np.empty(0))

    def inactive_mask(self, x):
        return np.zeros(self.dimension, dtype=bool)

    def phase2_bounds(self, active_indices):
        return [self.bounds[int(i)] for i in active_indices]


def lhs_baseline(problem, samples, seed):
    bounds = np.asarray(problem.bounds, dtype=float)
    unit = qmc.LatinHypercube(d=problem.dimension, seed=seed).random(samples)
    X = bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])
    return min(problem.evaluate(x).objective for x in X)


def make_config(seed, quick):
    return TPMOConfig(
        seed=seed,
        initial_samples=20 if quick else 30,
        alpha=0.25 if quick else 0.2,
        ec_tolerance=1e-6,
        phase1_max_iterations=10 if quick else 60,
        phase2_stagnation=3 if quick else 10,
        phase2_relative_tolerance=1e-4,
        phase2_max_iterations=5 if quick else 20,
        xgboost=XGBoostConfig(
            n_estimators=4 if quick else 8,
            max_depth=4,
            learning_rate=0.08,
        ),
        gradient=GradientConfig(relative_step=0.03),
        fsgo=FSGOConfig(
            population_size=10 if quick else 20,
            K=100 if quick else 200,
            precision_step=1e-4,
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--quick", action="store_true")
    group.add_argument("--full", action="store_true")
    args = parser.parse_args()

    quick = bool(args.quick)
    seeds = range(1001, 1003) if quick else range(2101, 2111)
    rows = []

    for name in FUNCTIONS:
        for seed in seeds:
            problem = ShiftedBenchmark(name)
            config = make_config(seed, quick)
            baseline = lhs_baseline(problem, config.initial_samples, seed)
            result = minimize_tpmo(problem, config=config)
            row = {
                "benchmark": name,
                "seed": seed,
                "initial_lhs": baseline,
                "tpmo": result.fun,
                "evaluations": result.evaluations,
                "phase1_iterations": result.phase1_iterations,
                "phase2_iterations": result.phase2_iterations,
            }
            rows.append(row)
            print(row)

    with open("comparison_results.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("wrote comparison_results.csv")


if __name__ == "__main__":
    main()
