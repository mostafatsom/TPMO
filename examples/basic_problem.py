import numpy as np

from tpmo import Evaluation, FSGOConfig, TPMOConfig, minimize_tpmo


class SphereProblem:
    dimension = 4
    bounds = [(-5.0, 5.0)] * dimension
    constraints = []

    def evaluate(self, x):
        x = np.asarray(x, dtype=float)
        return Evaluation(float(np.dot(x, x)), np.empty(0))

    def inactive_mask(self, x):
        return np.zeros(self.dimension, dtype=bool)

    def phase2_bounds(self, active_indices):
        return [self.bounds[int(i)] for i in active_indices]


config = TPMOConfig(
    seed=42,
    initial_samples=20,
    alpha=0.25,
    phase1_max_iterations=20,
    phase2_stagnation=5,
    phase2_max_iterations=10,
    fsgo=FSGOConfig(population_size=10, K=100),
)
result = minimize_tpmo(SphereProblem(), config=config)

print(result.fun)
print(result.x)
print(result.evaluations)
