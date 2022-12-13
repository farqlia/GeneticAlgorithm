import numpy as np

from src.implementation.individuals.individual import Individual
from src.implementation.individuals.knapsack_problem_individual import KnapsackProblemIndividual


class GeneticAlgorithm:

    def __init__(self, problem, population_size, crossover_prob=0.6,
                 mutation_prob=0.1, iterations=100):
        self.problem = problem
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.iterations = iterations
        self.population = np.empty(population_size, dtype=Individual)
        self.best_solution = None
        self.current_iteration = 1

    def run_iteration(self):

        # evaluate each solution
        for ind in self.population:
            ind.evaluate_fitness()

        # generate new population
        new_population = np.empty(self.population_size, dtype=Individual)

        self.best_solution = max(self.population, key=lambda ind: ind.fitness)

        for i in range(0, self.population_size, 2):
            parent1, parent2 = self.select_parents()

            if np.random.default_rng().random() < self.crossover_prob:
                child1, child2 = parent1.crossover(parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            child1.mutate()
            child2.mutate()

            new_population[i] = child1
            new_population[i + 1] = child2

        self.population = new_population
        print(f"Iteration [{self.current_iteration}], best solution = {self.best_solution.fitness}")
        self.current_iteration += 1

    def get_best_solution(self):
        return self.best_solution

    def is_done(self):
        return self.current_iteration > self.iterations

    def select_parents(self):

        def get_parent():
            p1 = self.population[np.random.default_rng().integers(0, self.population_size)]
            p2 = self.population[np.random.default_rng().integers(0, self.population_size)]
            return p1 if p1.fitness >= p2.fitness else p2

        parent1 = get_parent()
        parent2 = get_parent()
        while parent2 == parent1:
            parent2 = get_parent()

        return parent1, parent2

    def initialize(self):

        def create_instance():
            genome = np.random.default_rng().random(self.problem.genome_length()) < 0.5
            return KnapsackProblemIndividual(genome, self.problem)

        for i in range(self.population_size):
            self.population[i] = create_instance()

        # evaluate each solution
        #for ind in self.population:

            # ind.evaluate_fitness()
