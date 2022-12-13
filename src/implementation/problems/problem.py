from abc import ABC, abstractmethod
from os import getcwd
from pathlib import Path

DIR = Path(getcwd()).parents[0].joinpath('data')
LARGE_SCALE = DIR.joinpath('large_scale')
LARGE_SCALE_OPTIMUM = DIR.joinpath('large_scale-optimum')
LOW_SCALE = DIR.joinpath('low-dimensional')
LOW_SCALE_OPTIMUM = DIR.joinpath('low-dimensional-optimum')


class Problem(ABC):

    @abstractmethod
    def genome_length(self):
        return 0

    @abstractmethod
    def fitness(self, x):
        return 0

    @abstractmethod
    def read_from_file(self, instance, solution):
        pass