import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
class Selection(object):
    def __init__(self, elitism=0.):
        assert (elitism >= 0.) and (elitism < 1.), 'Elitism must be in [0, 1).'
        self.elitism = elitism

    def select_elite(self, x, f, population_size=None):
        if self.elitism > 0.:
            if population_size is None:
                population_size = x.shape[0]
            indices_sort = np.argsort(f)
            elite_indx = int(self.elitism * population_size)
            if elite_indx == 0:
                elite_indx = 1

            x = x[indices_sort]
            f = f[indices_sort]

            x_elite = x[:elite_indx]
            f_elite = f[:elite_indx]

            x_rest = x[elite_indx:]
            f_rest = f[elite_indx:]
        else:
            x_elite = None
            f_elite = None
            x_rest = x
            f_rest = f
            elite_indx = 0

        return x_rest, f_rest, x_elite, f_elite, elite_indx


# ----------------------------------------------------------------------------------------------------------------------
class SelectBest(Selection):
    def __init__(self):
        super().__init__(elitism=0.)

    def select(self, x, f, population_size=None):
        if population_size is None:
            population_size = x.shape[0]

        indices = np.argsort(f)

        x_new = x[indices]
        f_new = f[indices]

        return x_new[0:population_size], f_new[0:population_size]
