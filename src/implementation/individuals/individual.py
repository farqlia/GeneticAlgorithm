from abc import ABC, abstractmethod


class Individual:

    @abstractmethod
    def fitness(self):
        return 0

    @abstractmethod
    def mutate(self):
        return None

    @abstractmethod
    def crossover(self, other):
        return None, None

    @abstractmethod
    def copy(self):
        return None