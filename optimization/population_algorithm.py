from collections import Set
import numpy as np
import scipy.optimize as opt
from optimization.proposals import ContinuousProposal, MixingProposal, DifferentialProposal
from optimization.selections import AcceptanceSelection, ProportionalSelection, LikelihoodFreeAcceptanceUniformSelection, LikelihoodFreeAcceptanceGreedySelection
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------------------------------------------------
class LikelihoodFreeInference(object):
    def __init__(self, pop_algorithm, name=None, num_epochs=1):
        self.pop_algorithm = pop_algorithm

        self.num_epochs = num_epochs

        if name is None:
            self.name = self.pop_algorithm.name
        else:
            self.name = name

    def lf_inference(self, epsilon=0.1, save_figs_folder='../results/'):
        x_sample = None

        while x_sample is None:
            print(self.name, epsilon)
            x_sample = None
            f_sample = None

            x = self.pop_algorithm.x0.copy()
            f = self.pop_algorithm.evaluate_objective(x)

            for i in range(self.num_epochs):
                x, f = self.pop_algorithm.step(x, f, epsilon)

                # save figs
                if len(save_figs_folder) > 0:
                    plt.scatter(x[:, 0] / 60., x[:, 1], c=f, vmin=0., vmax=2.)
                    plt.xlim(self.pop_algorithm.bounds[0][0], self.pop_algorithm.bounds[1][0] / 60.)
                    plt.ylim(self.pop_algorithm.bounds[0][1], self.pop_algorithm.bounds[1][1])
                    plt.colorbar()
                    plt.savefig(save_figs_folder + '/' + self.name + '/' + str(i))
                    plt.close()

                indices = f < epsilon
                if np.sum(indices) > 0:
                    if x_sample is None:
                        x_sample = x[indices]
                        f_sample = f[indices]
                    else:
                        x_sample = np.concatenate((x_sample, x[indices]), 0)
                        f_sample = np.concatenate((f_sample, f[indices]), 0)

            epsilon = epsilon * 2.

        x_sample = np.unique(x_sample, axis=0)
        f_sample = np.unique(f_sample, axis=0)

        return x_sample, f_sample


# ----------------------------------------------------------------------------------------------------------------------
class PopulationAlgorithm(object):
    def __init__(self, fun, args):
        self.fun = fun
        self.params = args[0]
        self.y_real = args[1]

    def step(self, x, f, epsilon):
        raise NotImplementedError

    def objective_function(self, x):
        return self.fun(x, self.params, self.y_real)

    def evaluate_objective(self, x):
        f = np.zeros((x.shape[0],))
        for i in range(x.shape[0]):
            f[i] = self.objective_function(x[i])
        return f


# ----------------------------------------------------------------------------------------------------------------------
class MetropolisHastings(PopulationAlgorithm):
    def __init__(self, fun, args, x0, burn_in_phase = 0, num_epochs=100, bounds=(-np.infty, np.infty), objective_is_probability=False):
        super().__init__(fun, args)

        self.proposal = ContinuousProposal(type='gaussian', bounds=bounds, params=self.params)
        self.selection = AcceptanceSelection()

        self.objective_is_probability = objective_is_probability

        self.x0 = x0

        self.burn_in_phase = burn_in_phase
        self.num_epochs = num_epochs

    def sample(self, epsilon = 0.01):
        x = self.x0.copy()
        f = self.evaluate_objective(x)

        x_sample = None
        f_sample = None

        for i in range(self.num_epochs):
            # Sample new points
            x_new = self.proposal.sample(x)
            f_new = self.evaluate_objective(x_new)
            # Transform them to unnormalized probabilities if necessary
            # MH acceptance rule
            x, f = self.selection.select(x, f, x_new, f_new, objective_is_probability=self.objective_is_probability)

            if i > self.burn_in_phase:
                indx = f < epsilon
                if np.sum(indx) > 0:
                    if x_sample is None:
                        x_sample = x[indx]
                        f_sample = f[indx]
                    else:
                        x_sample = np.concatenate((x_sample, x[indx]), 0)
                        f_sample = np.concatenate((f_sample, f[indx]), 0)

        if self.objective_is_probability is False:
            f_sample = np.exp(-f_sample)

        return x_sample, f_sample

    def optimize(self):
        pass


class Evolutionary(PopulationAlgorithm):
    def __init__(self, fun, args, x0, burn_in_phase = 0, num_epochs=100, bounds=(-np.infty, np.infty),
                 objective_is_probability=False, continuous_proposal_type='gaussian',
                 mixing_proposal_type='uniform'):
        super().__init__(fun, args)

        self.proposal = ContinuousProposal(type=continuous_proposal_type, bounds=bounds, params=self.params)
        self.mixing = MixingProposal(type=mixing_proposal_type, num_cutting_points=self.params['num_cutting_points'])

        # self.selection = ProportionalSelection()
        self.selection = AcceptanceSelection()

        self.objective_is_probability = objective_is_probability

        self.x0 = x0

        self.burn_in_phase = burn_in_phase
        self.num_epochs = num_epochs

    def step(self, epsilon = 0.01):
        x = self.x0.copy()
        f = self.evaluate_objective(x)

        x_sample = None
        f_sample = None

        for i in range(self.num_epochs):
            # Sample new points
            x_new = self.mixing.sample(x, prob=self.params['mixing_prob'])
            x_new = self.proposal.sample(x_new)
            f_new = self.evaluate_objective(x_new)
            # Transform them to unnormalized probabilities if necessary
            # MH acceptance rule
            # x, f = self.selection.select(np.concatenate((x, x_new), 0),
            #                              np.concatenate((f, f_new), 0),
            #                              objective_is_probability=self.objective_is_probability,
            #                              population_size=x.shape[0])
            x, f = self.selection.select(x, f, x_new, f_new, objective_is_probability=self.objective_is_probability)



            if i > self.burn_in_phase:
                indx = f < epsilon
                if np.sum(indx) > 0:
                    if x_sample is None:
                        x_sample = x[indx]
                        f_sample = f[indx]
                    else:
                        x_sample = np.concatenate((x_sample, x[indx]), 0)
                        f_sample = np.concatenate((f_sample, f[indx]), 0)

        if self.objective_is_probability is False:
            f_sample = np.exp(-f_sample)

        return x_sample, f_sample

    def optimize(self):
        pass


# ----------------------------------------------------------------------------------------------------------------------
class Powell(PopulationAlgorithm):
    def __init__(self, fun, args, x0, bounds=(-np.infty, np.infty), max_iter=None):
        super().__init__(fun, args)

        self.name = 'powell'

        self.x0 = x0
        self.bounds = bounds
        self.args = args
        self.max_iter = max_iter

    def step(self, x, f, epsilon):
        x0 = np.asarray([np.random.uniform(self.bounds[0][i], self.bounds[1][i]) for i in range(len(self.bounds[0]))])

        # Sample new points by optimization
        res = opt.minimize(self.fun, x0, args=self.args, method='powell',
                           options={'maxiter': self.max_iter, 'xtol': 1e-8, 'disp': False})
        x = np.expand_dims(res.x, 0)
        f = self.evaluate_objective(x)

        return x, f


# ----------------------------------------------------------------------------------------------------------------------
class PopLiFe(PopulationAlgorithm):
    def __init__(self, fun, args, x0, burn_in_phase = 0, num_epochs=100, bounds=(-np.infty, np.infty),
                 objective_is_probability=False, continuous_proposal_type='gaussian',
                 mixing_proposal_type='uniform', differential_proposal_type='differential_evolution', population_size=None):
        super().__init__(fun, args)

        self.name = 'poplife'

        self.proposal = ContinuousProposal(type=continuous_proposal_type, bounds=bounds, params=self.params)
        self.mixing = MixingProposal(type=mixing_proposal_type, num_cutting_points=self.params['num_cutting_points'])
        self.differential = DifferentialProposal(type=differential_proposal_type, bounds=bounds, params=self.params)

        self.selection = LikelihoodFreeAcceptanceGreedySelection()

        self.objective_is_probability = objective_is_probability

        self.x0 = x0
        self.bounds = bounds

        if population_size is None:
            self.population_size = self.x0.shape[0]
        else:
            self.population_size = population_size

        self.burn_in_phase = burn_in_phase
        self.num_epochs = num_epochs

    def step(self, x, f, epsilon):

        # sample
        x_new = self.differential.sample(x)
        x_new = self.mixing.sample(x_new, prob=self.params['mixing_prob'])
        x_new = self.proposal.sample(x_new)

        # evaluate
        f_new = self.evaluate_objective(x_new)

        # select
        x, f = self.selection.select(x, f, x_new, f_new, epsilon=epsilon, population_size=self.population_size)

        return x, f
