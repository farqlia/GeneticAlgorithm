import numpy as np

from src.implementation.individuals.individual import Individual


class GeneticAlgorithm:

    def __init__(self, problem, population_size, crossover_prob=0.6,
                 mutation_prob=0.1, iterations=100):
        self.problem = problem
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.iterations = iterations
        self.population = np.empty(population_size, dtype=Individual)

    def run_iteration(self):
        pass

    def get_best_solution(self):
        return None

    def is_done(self):
        return False

    def tournament_selection(self):
        np.random.default_rng().choice(self.population, size=int(self.population_size/4), replace=False)


    def initialize(self):

        def create_instance():
            return np.random.default_rng().random(self.problem.genome_length()) < 0.5

        for i in range(self.population_size):
            self.population[i] = create_instance()
