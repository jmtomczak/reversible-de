import numpy as np
from algorithms.recombination import DifferentialRecombination
from algorithms.selections import SelectBest


# ----------------------------------------------------------------------------------------------------------------------
class OptimizationAlgorithm(object):
    def __init__(self, pop_algorithm, name=None, num_epochs=1):
        self.pop_algorithm = pop_algorithm

        self.num_epochs = num_epochs

        if name is None:
            self.name = self.pop_algorithm.name
        else:
            self.name = name

    def optimize(self, directory_name):

        x = self.pop_algorithm.x0.copy()
        f = self.pop_algorithm.evaluate_objective(x)

        f_min = np.min(f)
        ind_min = np.argmin(f)

        f_best_so_far = [f_min]

        x_sample = x[[ind_min]]
        f_sample = np.asarray([f_min])

        for i in range(self.num_epochs):
            x, f = self.pop_algorithm.step(x, f)

            f_min = np.min(f)
            if f_min < f_best_so_far[-1]:
                f_best_so_far.append(f_min)

                ind_min = np.argmin(f)

                x_sample = np.concatenate((x_sample, x[[ind_min]]), 0)
                f_sample = np.concatenate((f_sample, np.asarray([f_min])), 0)
            else:
                f_best_so_far.append(f_best_so_far[-1])

        np.save(directory_name + '/' + 'f_best', np.array(f_best_so_far))

        np.save(directory_name + '/' + 'last_x', np.array(x))
        np.save(directory_name + '/' + 'last_f', np.array(f))

        return x_sample, f_sample


# ----------------------------------------------------------------------------------------------------------------------
class PopulationAlgorithm(object):
    def __init__(self, fun, args):
        self.fun = fun
        self.params = args[0]
        self.y_real = args[1]

    def step(self, x, f):
        raise NotImplementedError

    def objective_function(self, x):
        return self.fun(x, self.params, self.y_real)

    def evaluate_objective(self, x):
        if self.params['evaluate_objective_type'] == 'single':
            f = np.zeros((x.shape[0],))
            for i in range(x.shape[0]):
                f[i] = self.objective_function(x[i])
        elif self.params['evaluate_objective_type'] == 'full':
            f = self.objective_function(x)
        else:
            raise ValueError('Wrong evaluation type!')
        return f


# ----------------------------------------------------------------------------------------------------------------------
class DifferentialEvolution(PopulationAlgorithm):
    def __init__(self, fun, args, x0, bounds=(-np.infty, np.infty),
                 population_size=None,
                 de_type='de'):
        super().__init__(fun, args)

        self.name = 'de_rand'

        self.differential = DifferentialRecombination(type=de_type, bounds=bounds, params=self.params)

        self.selection = SelectBest()

        self.x0 = x0
        self.bounds = bounds

        if population_size is None:
            self.population_size = self.x0.shape[0]
        else:
            self.population_size = population_size

    def step(self, x, f):

        # mutation + crossover
        x_new, _ = self.differential.recombination(x)
        if not (self.differential.type in ['de']):
            x_new = np.concatenate(x_new, 0)

        # evaluate only new points
        f_new = self.evaluate_objective(x_new)
        x_new = np.concatenate((x_new, x), 0)
        f_new = np.concatenate((f_new, f))

        # select
        x, f = self.selection.select(x_new, f_new, population_size=self.population_size)

        return x, f
