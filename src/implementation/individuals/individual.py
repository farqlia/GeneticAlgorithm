from abc import ABC, abstractmethod


class Individual:

    @abstractmethod
    def evaluate_fitness(self):
        pass

    @abstractmethod
    def mutate(self):
        return None

    @abstractmethod
    def crossover(self, other):
        return None, None

    @abstractmethod
    def copy(self):
        return None