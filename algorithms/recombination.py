import numpy as np
from ..utils.distributions import bernoulli


# ----------------------------------------------------------------------------------------------------------------------
class Recombination(object):
    def __init__(self):
        pass

    def recombination(self, x):
        pass


# ----------------------------------------------------------------------------------------------------------------------
class DifferentialRecombination(Recombination):
    def __init__(self, type='de', bounds=(-np.infty, np.infty), params=None):
        super().__init__()
        self.type = type
        self.bounds = bounds

        assert (0. <= params['F'] <= 2.), 'F must be in [0, 2]'
        assert (0. < params['CR'] <= 1.), 'CR must be in (0, 1]'
        assert type in ['de', 'ade', 'revde', 'dex3'], 'type must be one in {de, dex3, ade, revde}'

        self.F = params['F']
        self.CR = params['CR']

    def recombination(self, x):
        indices_1 = np.arange(x.shape[0])
        # take first parent
        x_1 = x[indices_1]
        # assign second parent (ensure)
        indices_2 = np.random.permutation(x.shape[0])
        x_2 = x_1[indices_2]
        # assign third parent
        indices_3 = np.random.permutation(x.shape[0])
        x_3 = x_2[indices_3]

        if self.type == 'de':
            y_1 = np.clip(x_1 + self.F * (x_2 - x_3), self.bounds[0], self.bounds[1])

            # uniform crossover
            if self.CR < 1.:
                p_1 = bernoulli(self.CR, y_1.shape)
                y_1 = p_1 * y_1 + (1. - p_1) * x_1

            return (y_1), (indices_1, indices_2, indices_3)

        elif self.type == 'revde':
            y_1 = np.clip(x_1 + self.F * (x_2 - x_3), self.bounds[0], self.bounds[1])
            y_2 = np.clip(x_2 + self.F * (x_3 - y_1), self.bounds[0], self.bounds[1])
            y_3 = np.clip(x_3 + self.F * (y_1 - y_2), self.bounds[0], self.bounds[1])

            # uniform crossover
            if self.CR < 1.:
                p_1 = bernoulli(self.CR, y_1.shape)
                p_2 = bernoulli(self.CR, y_2.shape)
                p_3 = bernoulli(self.CR, y_3.shape)
                y_1 = p_1 * y_1 + (1. - p_1) * x_1
                y_2 = p_2 * y_2 + (1. - p_2) * x_2
                y_3 = p_3 * y_3 + (1. - p_3) * x_3

            return (y_1, y_2, y_3), (indices_1, indices_2, indices_3)

        elif self.type == 'ade':
            y_1 = np.clip(x_1 + self.F * (x_2 - x_3), self.bounds[0], self.bounds[1])
            y_2 = np.clip(x_2 + self.F * (x_3 - x_1), self.bounds[0], self.bounds[1])
            y_3 = np.clip(x_3 + self.F * (x_1 - x_2), self.bounds[0], self.bounds[1])

            # uniform crossover
            if self.CR < 1.:
                p_1 = bernoulli(self.CR, y_1.shape)
                p_2 = bernoulli(self.CR, y_2.shape)
                p_3 = bernoulli(self.CR, y_3.shape)
                y_1 = p_1 * y_1 + (1. - p_1) * x_1
                y_2 = p_2 * y_2 + (1. - p_2) * x_2
                y_3 = p_3 * y_3 + (1. - p_3) * x_3

            return (y_1, y_2, y_3), (indices_1, indices_2, indices_3)

        if self.type == 'dex3':
            # y1
            y_1 = np.clip(x_1 + self.F * (x_2 - x_3), self.bounds[0], self.bounds[1])

            # uniform crossover
            if self.CR < 1.:
                p_1 = bernoulli(self.CR, y_1.shape)
                y_1 = p_1 * y_1 + (1. - p_1) * x_1

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

            # uniform crossover
            if self.CR < 1.:
                p_2 = bernoulli(self.CR, y_2.shape)
                y_2 = p_2 * y_2 + (1. - p_2) * x_1

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

            # uniform crossover
            if self.CR < 1.:
                p_3 = bernoulli(self.CR, y_3.shape)
                y_3 = p_3 * y_3 + (1. - p_3) * x_1

            return (y_1, y_2, y_3), (indices_1, indices_2, indices_3)
        else:
            raise ValueError('Wrong name of the differential mutation!')