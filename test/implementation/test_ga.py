import numpy as np

from src.implementation.algorithm.algorithm import GeneticAlgorithm
from src.implementation.problems.knapsack_problem import KnapsackProblem


def test_ga():

    problem = KnapsackProblem(W=5, values=np.array([5, 1, 4, 3]), weights=np.array([4, 1, 3, 2]),
                              optimal=7)

    alg = GeneticAlgorithm(problem, population_size=10, iterations=5)

    alg.initialize()

    while not alg.is_done():
        alg.run_iteration()
        best = alg.get_best_solution()
        print(best.genome)