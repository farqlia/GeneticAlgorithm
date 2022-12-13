from abc import ABC

import numpy as np

from src.implementation.individuals.individual import Individual


class KnapsackProblemIndividual(Individual):

    def __init__(self, genome, problem):
        self.genome = genome
        self.length = problem.genome_length()
        self.fitness = -np.inf
        self.problem = problem

    def evaluate_fitness(self):
        self.fitness = sum(self.problem.values[self.genome]) if sum(self.problem.weights[self.genome]) <= self.problem.W else 0

    def mutate(self, mut_prob=0.1):
        was_mutated = False
        for i in range(self.length):
            if np.random.default_rng().random() < mut_prob:
                self.genome[i] = not self.genome[i]
                was_mutated = True
        return was_mutated

    def crossover(self, other):
        cross_point = np.random.default_rng().integers(0, self.length)

        def copy(destination, source, start, end):
            for i in range(start, end):
                destination[i] = source[i]

        child_genome1 = np.ndarray.copy(self.genome)
        child_genome2 = np.ndarray.copy(other.genome)
        copy(child_genome1, other.genome, cross_point, self.length)
        copy(child_genome2, self.genome, 0, cross_point)

        return KnapsackProblemIndividual(child_genome1, self.problem), KnapsackProblemIndividual(child_genome2, self.problem)

    def copy(self):
        genome = np.ndarray.copy(self.genome)
        ind = KnapsackProblemIndividual(genome, self.problem)
        ind.fitness = self.fitness
        return ind



