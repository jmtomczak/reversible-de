import numpy as np
from utils.distributions import bernoulli


class Proposal(object):
    def __init__(self):
        pass

    def sample(self, x):
        pass


class EvolutionaryProposal(object):
    def __init__(self):
        self.recombination = None
        self.mutation = None

    def ea_proposal(self, x):
        pass


class ContinuousProposal(Proposal):
    def __init__(self, type='gaussian', bounds=(-np.infty, np.infty), params=None):
        super().__init__()
        self.type = type
        self.bounds = bounds

        self.params = params

    def sample(self, x, prob=1.):
        mask = np.random.binomial(1, prob, x.shape)

        if self.type == 'gaussian' and prob > 0.:
            gauss = np.random.multivariate_normal(np.zeros(x.shape[1]), self.params['cov'], x.shape[0])
            x = x + mask * gauss

            return np.clip(x, self.bounds[0], self.bounds[1])
        else:
            return x


class DifferentialProposal(Proposal):
    def __init__(self, type='differential_evolution', bounds=(-np.infty, np.infty), params=None):
        super().__init__()
        self.type = type
        self.bounds = bounds

        assert (0 <= params['F'] <= 2), 'F must be in [0, 2]'
        assert (0 < params['CR'] <= 1), 'CR must be in (0, 1]'

        self.F = params['F']
        self.CR = params['CR']

    def sample(self, x):
        if self.type == 'differential_evolution':
            indices = np.arange(x.shape[0])
            np.random.shuffle(indices)
            # take first parent
            parent_1 = x[indices]
            # assign second parent (shuffle x for that)
            np.random.shuffle(indices)
            parent_2 = parent_1[indices, :]
            # assign third parent
            np.random.shuffle(indices)
            parent_3 = parent_1[indices, :]

            p = bernoulli(self.CR, (1, parent_1.shape[1]))

            x = (1-p) * x + p * (parent_1 + self.F * (parent_2 - parent_3))

        elif self.type == 'reversible_evolution':
            indices = np.arange(x.shape[0])
            np.random.shuffle(indices)
            # take first parent
            x_1 = x[indices]
            # assign second parent (shuffle x for that)
            np.random.shuffle(indices)
            x_2 = x_1[indices, :]
            # assign third parent
            np.random.shuffle(indices)
            x_3 = x_2[indices, :]

            y_1 = x_1 + self.F * (x_2 - x_3)
            y_2 = x_2 + self.F * (y_1 - x_3)
            y_3 = x_3 + self.F * (y_2 - y_1)

            x = np.concatenate((y_1, y_2, y_3), 0)

        return np.clip(x, self.bounds[0], self.bounds[1])


class ProperDifferentialProposal(Proposal):
    def __init__(self, type='proper_differential_3', bounds=(-np.infty, np.infty), params=None):
        super().__init__()
        self.type = type
        self.bounds = bounds

        assert (0 <= params['F'] <= 2), 'F must be in [0, 2]'
        assert (0 < params['CR'] <= 1), 'CR must be in (0, 1]'

        self.randomized_F = params['randomized_F']
        self.F = params['F'] / 4.
        self.CR = params['CR']

    def sample(self, x):
        if self.randomized_F:
            self.F = np.random.rand(1)

        indices_1 = np.arange(x.shape[0])
        # take first parent
        x_1 = x[indices_1]
        # assign second parent (ensure)
        indices_2 = np.random.permutation(x.shape[0])
        while sum(indices_1 == indices_2) > 0:
            indices_2 = np.random.permutation(x.shape[0])
        x_2 = x_1[indices_2]
        # assign third parent
        indices_3 = np.random.permutation(x.shape[0])
        while sum(indices_1 == indices_3) > 0 and sum(indices_2 == indices_3) > 0:
            indices_3 = np.random.permutation(x.shape[0])
        x_3 = x_2[indices_3]

        if self.type == 'proper_differential_1':
            y_1 = np.clip(x_1 + self.F * (x_2 - x_3), self.bounds[0], self.bounds[1])

            return (y_1), (indices_1, indices_2, indices_3)

        elif self.type == 'proper_differential_2':
            y_1 = np.clip(x_1 + self.F * (x_2 - x_3), self.bounds[0], self.bounds[1])
            y_2 = np.clip(x_2 + self.F * (y_1 - x_3), self.bounds[0], self.bounds[1])

            return (y_1, y_2), (indices_1, indices_2, indices_3)

        elif self.type == 'proper_differential_3':
            y_1 = np.clip(x_1 + self.F * (x_2 - x_3), self.bounds[0], self.bounds[1])
            y_2 = np.clip(x_2 + self.F * (y_1 - x_3), self.bounds[0], self.bounds[1])
            y_3 = np.clip(x_3 + self.F * (y_2 - y_1), self.bounds[0], self.bounds[1])

            return (y_1, y_2, y_3), (indices_1, indices_2, indices_3)

        elif self.type == 'antisymmetric_differential':
            y_1 = np.clip(x_1 + self.F * (x_2 - x_3), self.bounds[0], self.bounds[1])
            y_2 = np.clip(x_2 + self.F * (x_3 - x_1), self.bounds[0], self.bounds[1])
            y_3 = np.clip(x_3 + self.F * (x_1 - x_2), self.bounds[0], self.bounds[1])

            return (y_1, y_2, y_3), (indices_1, indices_2, indices_3)

        if self.type == 'de_times_3':
            # y1
            y_1 = np.clip(x_1 + self.F * (x_2 - x_3), self.bounds[0], self.bounds[1])

            # y2
            indices_1p = np.arange(x.shape[0])
            # take first parent
            x_1 = x[indices_1p]
            # assign second parent (ensure)
            indices_2p = np.random.permutation(x.shape[0])
            x_2 = x_1[indices_2p]
            # assign third parent
            indices_3p = np.random.permutation(x.shape[0])
            x_3 = x_2[indices_3p]

            y_2 = np.clip(x_1 + self.F * (x_2 - x_3), self.bounds[0], self.bounds[1])

            # y3
            indices_1p = np.arange(x.shape[0])
            # take first parent
            x_1 = x[indices_1p]
            # assign second parent (ensure)
            indices_2p = np.random.permutation(x.shape[0])
            x_2 = x_1[indices_2p]
            # assign third parent
            indices_3p = np.random.permutation(x.shape[0])
            x_3 = x_2[indices_3p]

            y_3 = np.clip(x_1 + self.F * (x_2 - x_3), self.bounds[0], self.bounds[1])

            return (y_1, y_2, y_3), (indices_1, indices_2, indices_3)

        elif self.type == 'proper_differential_1_dist':
            u = x_2 - x_3
            dis2 = np.exp(-np.linalg.norm(x_1 - x_2, axis=1, keepdims=True))
            dis3 = np.exp(-np.linalg.norm(x_1 - x_3, axis=1, keepdims=True))
            m = np.minimum(dis2, dis3)
            y_1 = np.clip(x_1 + self.F * m * u, self.bounds[0], self.bounds[1])

            return (y_1), (indices_1, indices_2, indices_3)

        elif self.type == 'de_single':
            u = x_2[[0], :] - x_3[[0], :]
            y_1 = np.clip(x_1 + self.F * u, self.bounds[0], self.bounds[1])

            return (y_1), (indices_1, indices_2, indices_3)

    def sample2(self, x):
        if self.randomized_F:
            self.F = np.random.rand(1)

        indices = np.random.permutation(x.shape[0])

        x_1, x_2, x_3 = np.split(x[indices], 3)

        y_1 = np.clip(x_1 + self.F * (x_2 - x_3), self.bounds[0], self.bounds[1])
        y_2 = np.clip(x_2 + self.F * (y_1 - x_3), self.bounds[0], self.bounds[1])
        y_3 = np.clip(x_3 + self.F * (y_2 - y_1), self.bounds[0], self.bounds[1])

        return (y_1, y_2, y_3), indices

    def sample3(self, x):
        if self.randomized_F:
            self.F = np.random.rand(1)

        indices = np.random.permutation(x.shape[0])

        indices_1 = np.arange(x.shape[0])
        # take first parent
        x_1 = x[indices_1]
        # assign second parent (ensure)
        indices_2 = np.random.permutation(x.shape[0])
        while sum(indices_1 == indices_2) > 0:
            indices_2 = np.random.permutation(x.shape[0])
        x_2 = x_1[indices_2]
        # assign third parent
        indices_3 = np.random.permutation(x.shape[0])
        while sum(indices_1 == indices_3) > 0 and sum(indices_2 == indices_3) > 0:
            indices_3 = np.random.permutation(x.shape[0])
        x_3 = x_2[indices_3]

        y_1 = np.clip(x_1 + self.F * (x_2 - x_3), self.bounds[0], self.bounds[1])
        y_2 = np.clip(x_2 + self.F * (y_1 - x_3), self.bounds[0], self.bounds[1])
        y_3 = np.clip(x_3 + self.F * (y_2 - y_1), self.bounds[0], self.bounds[1])

        return y_3, indices


class BinaryProposal(Proposal):
    def __init__(self, type='flip', args=None):
        super().__init__()
        self.type = type
        self.args = args

    def sample(self, x, prob=1.):
        mask = np.random.binomial(1, prob, x.shape)

        if self.type == 'flip':
            x = np.mod(x + mask, 2)

        return x


class MixingProposal(Proposal):
    def __init__(self, type='uniform', num_cutting_points=0):
        super().__init__()
        self.type = type
        self.num_cutting_points = num_cutting_points

    def sample(self, x, prob=0.5):

        if self.type in ['k-points', 'uniform']:
            b = bernoulli(prob, (x.shape[0],))

            x_remain = x[(1. - b).astype(bool)]

            if np.sum(b) > 0:
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

                return np.concatenate((x_remain, child), 0)
            else:
                return x_remain
        else:
            return x