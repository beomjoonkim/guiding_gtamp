import pygmo as pg
import numpy as np


class GeneticAlgoProblem:
    def __init__(self, q_function):
        # self.q_function = q_function
        #self.domain = domain
        self.objective_function = q_function

    def fitness(self, x):
        return [float(self.objective_function(x)[0, 0])]

    def get_bounds(self):
        domain = np.array([[0, -20, -1, -1], [10, 0, 1, 1]])
        return (domain[0].tolist(), domain[1].tolist())

def genetic_algorithm(q_function, domain):
    prob = pg.problem(GeneticAlgoProblem(q_function))
    import pdb;pdb.set_trace()
    population_size = 20

    total_evals = 1000
    generations = total_evals / population_size
    #optimizer = pg.cmaes(gen=generations, ftol=1e-20, xtol=1e-20)
    algo = pg.algorithm(pg.sade(gen=total_evals))
    #algo = pg.algorithm(optimizer)
    algo.set_verbosity(1)

    pop = pg.population(prob, size=population_size)
    pop = algo.evolve(pop)
    champion_x = pop.champion_x
    print pop.champion_f
    print pop.champion_x
    import pdb;pdb.set_trace()

    uda = algo.extract(pg.cmaes)
    log = np.array(uda.get_log())
    n_fcn_evals = log[:, 1]
    pop_best_at_generation = -log[:, 2]
    evaled_x = None
    evaled_y = pop_best_at_generation
    import pdb;
    pdb.set_trace()

    max_y = [pop_best_at_generation[0]]
    for y in pop_best_at_generation[1:]:
        if y > max_y[-1]:
            max_y.append(y)
        else:
            max_y.append(max_y[-1])

    return evaled_y, max_y
