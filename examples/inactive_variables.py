import numpy as np

from tpmo import Evaluation, FSGOConfig, TPMOConfig, minimize_tpmo


class SparseToyProblem:
    dimension = 5
    bounds = [(0.0, 4.0)] * dimension
    constraints = []

    def evaluate(self, x):
        x = np.asarray(x, dtype=float)
        target = np.array([2.0, 0.0, 1.5, 0.0, 0.8])
        return Evaluation(float(np.sum((x - target) ** 2)), np.empty(0))

    def inactive_mask(self, x):
        return np.asarray(x, dtype=float) < 0.25

    def phase2_bounds(self, active_indices):
        return [(0.25, 4.0) for _ in active_indices]


config = TPMOConfig(
    seed=11,
    initial_samples=25,
    alpha=0.25,
    phase1_max_iterations=20,
    phase2_stagnation=5,
    phase2_max_iterations=10,
    fsgo=FSGOConfig(population_size=10, K=100),
)
result = minimize_tpmo(SparseToyProblem(), config=config)

print(result.fun)
print(result.x)
print(result.active_indices)
