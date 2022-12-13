import numpy as np
from src.implementation.problems.problem import Problem


class KnapsackProblem(Problem):

    def __init__(self, W=0, values=None, weights=None, optimal=0):
        assert (values is None and weights is None) or len(values) == len(weights)
        self.n = 0 if values is None else len(values)
        self.W = W
        self.values = values
        self.weights = weights
        self.optimal = optimal

    def genome_length(self):
        return self.n

    # x is a boolean array
    def fitness(self, mask):
        return sum(self.values[mask]) if sum(self.weights[mask]) <= self.W else 0

    # assumes that the file is in form
    # n wmax
    # v1 w1
    # ...
    # vn wn
    # where vi is the value and wi is the weight
    def read_from_file(self, instance, solution):

        def number(x):
            try:
                x = float(x)
                return x > 0
            except ValueError:
                return False

        def read_next(line):
            if not line:
                return -1, -1
            entries = line.strip().split(" ")
            if len(entries) != 2 or not number(entries[0]) or not number(entries[1]):
                return -1, -1
            return float(entries[0]), float(entries[1])

        try:
            with open(instance) as f:
                line = f.readline()
                n, wmax = read_next(line)
                if n == - 1 and wmax == -1:
                    return False
                self.n = int(n)
                self.W = wmax

                self.values = np.empty(self.n, dtype='float32')
                self.weights = np.empty(self.n, dtype='float32')

                for i in range(self.n):
                    v, w = read_next(f.readline())
                    if w == - 1 and w == -1:
                        return False
                    self.values[i] = v
                    self.weights[i] = w

                with open(solution) as f:
                    line = f.readline()
                    if not line or not number(line):
                        return False
                    else:
                        self.optimal = float(line)

        except OSError:
            return False
        return True

