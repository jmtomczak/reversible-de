import numpy as np
from utils.distributions import bernoulli

# general idea: if probability is set to None, then using an operator is random (we sample uniformly prob)
# but if we set prob to 0, then the operator is not used


class Recombination(object):
    def __init__(self, type='uniform', prob=None, num_cutting_points=0):
        assert type in ['k-points', 'uniform']
        self.type = type

        if prob is not None:
            assert (prob >= 0) and (prob <= 1), 'Probability must be in [0, 1]!'
        self.prob = prob # the probability of applying recombination (the lower it is, the lower chance it's applied)

        if type in ['k-points']:
            assert num_cutting_points > 0, 'There must be at least 1 cutting point!'
        self.num_cutting_points = num_cutting_points

    def recombine(self, x):
        # pick which points to recombine
        if self.prob is None:
            b = bernoulli(np.random.rand(1), (x.shape[0],))
        else:
            b = bernoulli(self.prob, (x.shape[0],))

        if np.sum(b) > 0:
            x_remain = x[(1. - b).astype(bool)]

            # take first parent
            parent_1 = x[b.astype(bool)]
            # assign second parent (shuffle x for that)
            indices = np.arange(parent_1.shape[0])
            np.random.shuffle(indices)
            parent_2 = parent_1[indices, :]

            if self.type == 'uniform':
                p = bernoulli(0.5, (1, parent_1.shape[1]))

            elif self.type == 'k-points':
                if self.num_cutting_points >= parent_1.shape[1]:
                    self.num_cutting_points = parent_1.shape[1] - 1

                cutting_points = np.random.permutation(np.arange(parent_1.shape[1]))[:self.num_cutting_points]
                cutting_points = np.sort(cutting_points)

                p = np.zeros((1, parent_1.shape[1]))
                if self.num_cutting_points == 1:
                    r = 1
                else:
                    r = 0
                start = 0
                for j in cutting_points:
                    p[0, start:j] = r
                    r = np.mod(r + 1, 2)
                    start = j

            child = p * parent_1 + (1. - p) * parent_2

            # return indices of remaining points (in order to avoid re-evaluating fitness for them again!) and children
            return (1. - b).astype(bool), child
        else:
            return (1. - b).astype(bool), None


class Mutation(object):
    def __init__(self, type='binary', prob=None, cov=None):
        if prob is not None:
            assert (prob >= 0) and (prob <= 1), 'Probability must be in [0, 1]!'
        assert type in ['binary', 'gaussian']
        # if type == 'gaussian':
        #     assert (std is not None) and (std > 0.)
        # check positive-definite

        self.prob = prob  # probability of CHANGING a variable
        self.type = type
        self.cov = cov

    def mutate(self, x):
        x_shape = x.shape

        if self.prob == 0.:
            return x
        else:
            if self.prob is None:
                mask = bernoulli(np.random.rand(1), x_shape)
            else:
                mask = bernoulli(self.prob, x_shape)

            if self.type == 'binary':
                x = np.mod(x + mask, 2)
            elif self.type == 'gaussian':
                gauss = np.random.multivariate_normal(np.zeros(x_shape[1]), self.cov)
                x = x + mask * gauss
            return x


class Selection(object):
    def __init__(self, type='proportional', elitism=0.):
        assert type in ['proportional'], 'Wrong type!'
        self.type = type
        assert (elitism >= 0.) and (elitism < 1.), 'Elitism must be in [0, 1).'
        self.elitism = elitism

    def select(self, x, f, population_size=1):
        assert (population_size > 0) and (population_size <= x.shape[0]), 'Too small or too large population size!'

        if self.elitism > 0.:
            indices_sort = np.argsort(f)
            elite_indx = int(self.elitism * x.shape[0])
            if elite_indx == 0:
                elite_indx = 1

            x = x[indices_sort]
            f = f[indices_sort]

            x_elite = x[:(elite_indx)]
            f_elite = f[:(elite_indx)]

            x_rest = x[(elite_indx):]
            f_rest = f[(elite_indx):]
        else:
            x_rest = x
            f_rest = f
            elite_indx = 0

        if self.type == 'proportional':
            exp_f = np.exp(-f_rest)
            probability = exp_f / np.sum(exp_f)

            indices = np.random.choice(x_rest.shape[0], population_size - elite_indx, replace=False, p=probability)

        if self.elitism > 0.:
           return np.concatenate((x_elite, x_rest[indices]), 0), np.concatenate((f_elite, f_rest[indices]), 0)
        else:
            return x_rest[indices], f_rest[indices]


class EvolutionaryAlgorithm(object):
    def __init__(self, fun, args, x_0, add_fun=None, recombination_type='uniform', recombination_prob=0.5,
                 recombination_num_cutting_points=1, mutation_type='gaussian', mutation_prob=0.5, mutation_cov=0.1,
                 selection_type='proportional', selection_elitism=0., population_size=100, num_iters=100):
        self.recombination = Recombination(type=recombination_type, prob=recombination_prob,
                                           num_cutting_points=recombination_num_cutting_points)
        self.mutation = Mutation(type=mutation_type, prob=mutation_prob, cov=mutation_cov)
        self.selection = Selection(type=selection_type, elitism=selection_elitism)

        self.population_size = population_size
        self.num_iters = num_iters

        self.x_shape = x_0.shape

        self.fun = fun
        self.add_fun = add_fun
        self.args = args

        self.x_0 = x_0

    def fitness(self, x):
        if self.add_fun is None:
            return self.fun(x, self.args[0], self.args[1])
        else:
            raise ValueError('Not implemented!')

    def evaluate_fitness(self, x):
        f = np.zeros((x.shape[0],))
        for i in range(x.shape[0]):
            f[i] = self.fitness(x[i])
        return f

    def recombine(self, x, f):
        if (self.recombination.prob is None) or (self.recombination.prob > 0.):
            indices, child = self.recombination.recombine(x)
            x_new = x[indices]
            f_new = f[indices]

            # if there are children, we need to evaluate them
            if child is not None:
                f_child = self.evaluate_fitness(child)
                x_new = np.concatenate((x_new, child), 0)
                f_new = np.concatenate((f_new, f_child), 0)

            return x_new, f_new
        else:
            return x, f

    def mutate(self, x, f):
        if (self.mutation.prob is None) or (self.mutation.prob > 0.):
            x_new = self.mutation.mutate(x)
            # evaluate new fitness
            f_new = self.evaluate_fitness(x_new)
            return x_new, f_new
        else:
            return x, f

    def select(self, x, f):
        x_new, f_new = self.selection.select(x, f, self.population_size)
        return x_new, f_new

    def step(self, x, f):
        x_old = x.copy()
        f_old = f.copy()
        if (self.recombination.prob is None) or (self.recombination.prob > 0):
            x, f = self.recombine(x, f)

        if (self.mutation.prob is None) or (self.mutation.prob > 0):
            x, f = self.mutate(x, f)

        x, f = self.select(np.concatenate((x_old, x), 0), np.concatenate((f_old, f), 0))

        return x, f

    def optimize(self):
        x = self.x_0.copy()
        f = self.evaluate_fitness(x)

        for i in range(self.num_iters):
            x, f, = self.step(x, f)

        indx = np.argmin(f)

        return x[indx], f[indx]

    def sample(self, epsilon = 0.01):
        x = self.x_0.copy()
        f = self.evaluate_fitness(x)

        x_sample = None
        for i in range(self.num_iters):
            x, f = self.step(x, f)

            indx = f < epsilon
            if np.sum(indx) > 0:
                if x_sample is None:
                    x_sample = x[indx]
                else:
                    x_sample = np.concatenate((x_sample, x[indx]), 0)

        return x_sample