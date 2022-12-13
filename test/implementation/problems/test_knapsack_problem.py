import pytest
import numpy as np
import numpy.testing as np_test
from src.implementation.problems.knapsack_problem import KnapsackProblem

problem = KnapsackProblem(W=5, values=np.array([5, 1, 4, 3]), weights=np.array([4, 1, 3, 2]),
                          optimal=7)


class TestFitness:

    @pytest.mark.parametrize('solution,fitness', [(np.array([False, True, False, True]), 4),
                                                  (np.array([False, True, True, False]), 5)])
    def test_fitness_for_feasible(self, solution, fitness):
        assert problem.fitness(solution) == fitness

    @pytest.mark.parametrize('solution', [np.array([True, True, True, False])])
    def test_fitness_for_unfeasible(self, solution):
        assert problem.fitness(solution) == 0


@pytest.fixture()
def valid_instance(tmpdir):
    filename = tmpdir.join("valid_instance.txt")
    filename.write('4 5\n5 4\n1 1\n4 3\n3 2')
    solution = tmpdir.join("valid_instance_sol.txt")
    solution.write('7')
    return filename, solution


@pytest.fixture()
def instance_empty(tmpdir):
    filename = tmpdir.join("not_valid_instance.txt")
    filename.write('')
    solution = tmpdir.join("valid_instance_sol.txt")
    solution.write('')
    return filename, solution


@pytest.fixture()
def instance_with_not_enough_data(tmpdir):
    filename = tmpdir.join("valid_instance.txt")
    filename.write('4 5')
    solution = tmpdir.join("valid_instance_sol.txt")
    solution.write('7')
    return filename, solution


@pytest.fixture()
def instance_with_floats(tmpdir):
    filename = tmpdir.join("not_valid_instance.txt")
    filename.write('4 5\n5.4 4.4\n1 1.4\n4.34 3\n3.5 2')
    solution = tmpdir.join("valid_instance_sol.txt")
    solution.write('7')
    return filename, solution


class TestReadFromFile:

    def test_read_from_existing_file(self, valid_instance):
        instance, solution = valid_instance
        p = KnapsackProblem()
        assert p.read_from_file(instance, solution)
        assert p.W == problem.W
        assert p.n == problem.n
        assert p.optimal == problem.optimal
        np_test.assert_array_equal(p.values, problem.values)
        np_test.assert_array_equal(p.weights, problem.weights)

    def test_read_from_invalid_file2(self, instance_with_floats):
        instance, solution = instance_with_floats
        p = KnapsackProblem()
        assert p.read_from_file(instance, solution)
        assert p.W == 5
        assert p.n == 4
        assert p.optimal == 7

    def test_read_from_valid_file(self, instance_with_not_enough_data):
        instance, solution = instance_with_not_enough_data
        p = KnapsackProblem()
        assert not p.read_from_file(instance, solution)

    def test_read_from_non_existing_file(self):
        p = KnapsackProblem()
        assert not p.read_from_file("no_exist.txt", "")


    def test_read_from_invalid_file3(self, instance_empty):
        instance, solution = instance_empty
        p = KnapsackProblem()
        assert not p.read_from_file(instance, solution)
