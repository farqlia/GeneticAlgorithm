import numpy as np
import numpy.testing as np_test
import pytest

from src.implementation.individuals.knapsack_problem_individual import KnapsackProblemIndividual
from src.implementation.problems.knapsack_problem import KnapsackProblem

problem = KnapsackProblem(W=5, values=np.array([5, 1, 4, 3]), weights=np.array([4, 1, 3, 2]),
                          optimal=7)


def test_genotype():
    genome = np.array([True, False, False, True])
    ind = KnapsackProblemIndividual(genome=genome, problem=problem)
    np_test.assert_array_equal(ind.genome, genome)


@pytest.mark.parametrize("genome,fit", [(np.array([False, True, False, True]), 4),
                                        (np.array([False, True, True, False]), 5),
                                        (np.array([True, False, True, False]), 0)])
def test_fitness(genome, fit):
    ind = KnapsackProblemIndividual(genome=genome, problem=problem)
    assert ind.evaluate_fitness() == fit


def test_mutate_all_genes():
    genome = np.array([True, False, False, True])
    ind = KnapsackProblemIndividual(genome=genome, problem=problem)
    assert ind.mutate(mut_prob=1.0)
    mutated_genome = np.array([False, True, True, False])
    np_test.assert_array_equal(ind.genome, mutated_genome)


def test_dont_mutate_any_gene():
    genome = np.array([True, False, False, True])
    ind = KnapsackProblemIndividual(genome=genome, problem=problem)
    assert not ind.mutate(mut_prob=0.0)
    original_genome = np.array([True, False, False, True])
    np_test.assert_array_equal(ind.genome, original_genome)


def test_crossover():
    genome1 = np.array([False, False, True, False])
    ind1 = KnapsackProblemIndividual(genome=genome1, problem=problem)
    genome2 = np.array([False, True, False, True])
    ind2 = KnapsackProblemIndividual(genome=genome2, problem=problem)

    child1, child2 = ind1.crossover(ind2)
    with np_test.assert_raises(AssertionError):
        np_test.assert_array_equal(child1.genome, genome1)

    with np_test.assert_raises(AssertionError):
        np_test.assert_array_equal(child2.genome, genome2)


def test_copy():
    genome = np.array([True, False, False, True])
    ind = KnapsackProblemIndividual(genome=genome, problem=problem)
    copy_ind = ind.copy()
    np_test.assert_array_equal(ind.genome, copy_ind.genome)
