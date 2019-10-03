import pygmo as pg
import numpy as np


class GeneticAlgoProblem:
    def __init__(self, q_function):
        # self.q_function = q_function
        # self.domain = domain
        self.objective_function = q_function

    def fitness(self, x):
        return [-self.objective_function(x)]

    def get_bounds(self):
        domain = np.array([[0, -20, -1, -1], [10, 0, 1, 1]])
        return (domain[0].tolist(), domain[1].tolist())


def genetic_algorithm(q_function, total_evals=100):
    prob = pg.problem(GeneticAlgoProblem(q_function))
    population_size = 20

    generations = total_evals / population_size
    algo = pg.algorithm(pg.sade(gen=generations))

    pop = pg.population(prob, size=population_size)
    pop = algo.evolve(pop)
    print -pop.champion_f

    return pop.champion_x, -pop.champion_f
