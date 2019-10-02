import pygmo as pg
import numpy as np


class GeneticAlgoProblem:
    def __init__(self, q_function, domain):
        self.q_function = q_function
        self.domain = domain
        self.objective_function = lambda x: self.q_function.predict()

    def hinge_function(self, x):
        return max(0, x)

    def fitness(self, x):
        return [self.objective_function(x)]

    def get_bounds(self):
        # domain = np.array([[-600.] * dim_x, [600.] * dim_x])
        # return (domain[0].tolist(), domain[1].tolist())
        return (self.domain[0], self.domain[1])


def genetic_algorithm(q_function, domain):
    prob = pg.problem(GeneticAlgoProblem(q_function, domain))
    population_size = 5

    total_evals = 1000
    generations = total_evals / population_size
    optimizer = pg.cmaes(gen=generations, ftol=1e-20, xtol=1e-20)
    algo = pg.algorithm(optimizer)
    algo.set_verbosity(1)

    pop = pg.population(prob, size=population_size)
    pop = algo.evolve(pop)
    print pop.champion_f

    champion_x = pop.champion_x
    uda = algo.extract(pg.cmaes)
    log = np.array(uda.get_log())
    n_fcn_evals = log[:, 1]
    pop_best_at_generation = -log[:, 2]
    evaled_x = None
    evaled_y = pop_best_at_generation
    import pdb;pdb.set_trace()

    max_y = [pop_best_at_generation[0]]
    for y in pop_best_at_generation[1:]:
        if y > max_y[-1]:
            max_y.append(y)
        else:
            max_y.append(max_y[-1])

    return evaled_y, max_y
