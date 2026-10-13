"""Evolutionary optimisation for multi-objective cost minimisation using DEAP.


This class is the runtime optimiser used by the live pipeline (app.py and
orchestration/). Its fitness function, `_evaluate_weights`, is a simplified
analytic surrogate: it scores a weight vector directly rather than by
simulating cargo outcomes.

The A2 experiment reported in Sect. 5.6 of the paper does NOT use this class.
Those numbers come from run_a2_experiment.py, which runs its own DEAP search
whose fitness is the measured cost over simulated cases, under common random
numbers across seeds. If you want to reproduce the GA results in the paper,
run that script, not this one.
"""
import random
import numpy as np
from deap import base, creator, tools, algorithms
from typing import List, Tuple, Dict, Any
import pickle

class EvolutionaryOptimiser:
    """
    Genetic Algorithm for multi‑objective weight optimisation.
    Based on Fyvie et al. (2025) – co‑evolves decision criteria weights.
    """

    def __init__(self, n_objectives: int = 4, population_size: int = 100, generations: int = 50):
        self.n_objectives = n_objectives
        self.population_size = population_size
        self.generations = generations
        self.best_weights = None
        self._setup_deap()

    def _setup_deap(self):
        creator.create("FitnessMin", base.Fitness, weights=tuple([-1.0] * self.n_objectives))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", random.uniform, 0.1, 1.0)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.attr_float, n=self.n_objectives)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self._evaluate_weights)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _evaluate_weights(self, weights: List[float]) -> Tuple[float, ...]:
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]

        # Simulate performance (simplified model)
        spoilage = 10 * (1 - weights[0]) + random.uniform(-2, 2)
        rerouting = 5 * (1 - weights[1]) + random.uniform(-1, 1)
        delay = 8 * (1 - weights[2]) + random.uniform(-2, 2)
        carbon = 3 * (1 - weights[3]) + random.uniform(-1, 1)

        return (max(0, spoilage), max(0, rerouting), max(0, delay), max(0, carbon))

    def train(self, simulation_function=None):
        if simulation_function:
            self.toolbox.register("evaluate", simulation_function)

        pop = self.toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(5)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        pop, log = algorithms.eaSimple(pop, self.toolbox,
                                       cxpb=0.7, mutpb=0.2, ngen=self.generations,
                                       stats=stats, halloffame=hof, verbose=True)

        best = hof[0]
        total = sum(best)
        self.best_weights = [w / total for w in best]
        return self.best_weights

    def get_weights(self) -> List[float]:
        if self.best_weights is None:
            self.best_weights = [0.25] * self.n_objectives
        return self.best_weights

    def compute_cost(self, metrics: Dict[str, float]) -> float:
        weights = self.get_weights()
        cost = (weights[0] * metrics.get('spoilage', 0) +
                weights[1] * metrics.get('rerouting', 0) +
                weights[2] * metrics.get('delay', 0) +
                weights[3] * metrics.get('carbon', 0))
        return cost

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'best_weights': self.best_weights}, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.best_weights = data['best_weights']