import numpy as np
import scipy.optimize as opt
from optimization.proposals import ContinuousProposal, MixingProposal, DifferentialProposal
from optimization.selections import AcceptanceSelection, ProportionalSelection, LikelihoodFreeAcceptanceUniformSelection, \
    LikelihoodFreeAcceptanceGreedySelection, SelectBest, RevGreedy
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

    def lf_inference(self, directory_name, epsilon=0.1):

        x_sample = None

        while x_sample is None:
            x_sample = None
            f_sample = None

            x = self.pop_algorithm.x0.copy()
            f = self.pop_algorithm.evaluate_objective(x)

            f_best_so_far = [np.min(f)]

            for i in range(self.num_epochs):
                x, f = self.pop_algorithm.step(x, f, epsilon=epsilon)

                f_best_so_far.append(np.min(f))

                # save figs
                if (x.shape[1] == 2):
                    plt.scatter(x[:, 0] / 60., x[:, 1], c=f, vmin=0., vmax=2.)
                    plt.xlim(self.pop_algorithm.bounds[0][0], self.pop_algorithm.bounds[1][0] / 60.)
                    plt.ylim(self.pop_algorithm.bounds[0][1], self.pop_algorithm.bounds[1][1])
                    plt.colorbar()
                    plt.savefig(directory_name + '/' + str(i))
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

        np.save(directory_name + '/' + 'f_best', np.array(f_best_so_far))

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
class MetropolisHastings(PopulationAlgorithm):
    def __init__(self, fun, args, x0, burn_in_phase = 0, num_epochs=100, bounds=(-np.infty, np.infty),
                 objective_is_probability=False, name='mh'):
        super().__init__(fun, args)
        self.name = name

        self.proposal = ContinuousProposal(type='gaussian', bounds=bounds, params=self.params)
        # self.selection = AcceptanceSelection()
        self.selection = LikelihoodFreeAcceptanceUniformSelection()

        self.objective_is_probability = objective_is_probability

        self.x0 = x0
        self.bounds = bounds

        self.burn_in_phase = burn_in_phase
        self.num_epochs = num_epochs

    def step(self, x, f, epsilon):
        # Sample new points
        x_new = self.proposal.sample(x)
        f_new = self.evaluate_objective(x_new)
        # MH acceptance rule
        # x, f = self.selection.select(x, f, x_new, f_new, objective_is_probability=self.objective_is_probability)
        x, f = self.selection.select(x, f, x_new, f_new, epsilon=epsilon)

        return x, f

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
                 mixing_proposal_type='uniform', differential_proposal_type='differential_evolution',
                 population_size=None, elitism=0.):
        super().__init__(fun, args)

        self.name = 'poplife'

        self.proposal = ContinuousProposal(type=continuous_proposal_type, bounds=bounds, params=self.params)
        self.mixing = MixingProposal(type=mixing_proposal_type, num_cutting_points=self.params['num_cutting_points'])
        self.differential = DifferentialProposal(type=differential_proposal_type, bounds=bounds, params=self.params)

        self.selection = LikelihoodFreeAcceptanceGreedySelection()
        self.best_selection = SelectBest()

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
        x_new = self.proposal.sample(x_new, prob=self.params['gaussian_prob'])

        # evaluate
        f_new = self.evaluate_objective(x_new)

        # select
        # x, f = self.best_selection.select(x_new, f_new, population_size=self.population_size)
        x, f = self.selection.select(x, f, x_new, f_new, epsilon=epsilon, population_size=self.population_size)

        return x, f


# ----------------------------------------------------------------------------------------------------------------------
class ReversiblePopLiFe(PopulationAlgorithm):
    def __init__(self, fun, args, x0, burn_in_phase = 0, num_epochs=100, bounds=(-np.infty, np.infty),
                 de_proposal_type='proper_differential_1',
                 continuous_proposal_type='gaussian',
                 selection='best',
                 population_size=None, elitism=0.):
        super().__init__(fun, args)

        self.name = 'revpoplife'

        self.differential = DifferentialProposal(type=de_proposal_type, bounds=bounds, params=self.params)
        self.proposal = ContinuousProposal(type=continuous_proposal_type, bounds=bounds, params=self.params)

        if selection == 'best':
            self.selection = SelectBest()
        elif selection == 'proportional':
            self.selection = ProportionalSelection(elitism=elitism)
        # elif selection == 'likelihood_greedy':
        #     self.selection = LikelihoodFreeAcceptanceGreedySelection()
        else:
            raise ValueError('Wrong selection mechanism!')

        self.objective_is_probability = False

        self.x0 = x0
        self.bounds = bounds

        if population_size is None:
            self.population_size = self.x0.shape[0]
        else:
            self.population_size = population_size

        self.burn_in_phase = burn_in_phase
        self.num_epochs = num_epochs

    def step(self, x, f, epsilon):

        x_new, _ = self.differential.sample(x)
        if not(self.differential.type in ['differential_1', 'differential_1_dist', 'de_single']):
            x_new = np.concatenate(x_new, 0)

        if self.params['gaussian_prob'] > 0.:
            x_new = np.concatenate((x_new, x), 0)
            x_new = self.proposal.sample(x_new, prob=self.params['gaussian_prob'])
            # evaluate all, because we add Gaussian noise to all points
            f_new = self.evaluate_objective(x_new)
        else:
            # evaluate only new points
            f_new = self.evaluate_objective(x_new)
            x_new = np.concatenate((x_new, x), 0)
            f_new = np.concatenate((f_new, f))

        # select
        x, f = self.selection.select(x_new, f_new, population_size=self.population_size)

        return x, f


# ----------------------------------------------------------------------------------------------------------------------
class RevPopLiFe(PopulationAlgorithm):
    def __init__(self, fun, args, x0, burn_in_phase = 0, num_epochs=100, bounds=(-np.infty, np.infty),
                 continuous_proposal_type='gaussian',
                 population_size=None, elitism=0.):
        super().__init__(fun, args)

        self.name = 'revpoplife'

        self.differential = DifferentialProposal(bounds=bounds, params=self.params)
        self.proposal = ContinuousProposal(type=continuous_proposal_type, bounds=bounds, params=self.params)

        # self.selection = RevGreedy()
        self.selection = SelectBest()

        self.objective_is_probability = False

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
        x_new, indices = self.differential.sample3(x)
        # x_new = np.concatenate(x_new, 0)

        # evaluate
        f_new = self.evaluate_objective(x_new)

        # select
        # x, f = self.selection.select2(x, f, np.concatenate(x_new, 0), f_new)
        x, f = self.selection.select(np.concatenate((x_new, x[indices]), 0), np.concatenate((f_new, f[indices]), 0),
                                     population_size=self.population_size)

        return x, f
