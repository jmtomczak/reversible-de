import numpy as np


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


class ProportionalSelection(Selection):
    def __init__(self, elitism=0.):
        super().__init__(elitism=elitism)

    def select(self, x, f, objective_is_probability, epsilon=np.infty, population_size=None, inv_temp=1.):
        if population_size is None:
            population_size = x.shape[0]

        # select an elite
        x_rest, f_rest, x_elite, f_elite, elite_indx = self.select_elite(x, f, population_size=population_size)

        # calculate probability by using softmax
        if objective_is_probability is False:
            exp_f_rest = np.exp(-inv_temp*f_rest)
        else:
            exp_f_rest = f_rest
        probability = exp_f_rest / np.sum(exp_f_rest)

        indices = np.random.choice(x_rest.shape[0], population_size - elite_indx, replace=False, p=probability)

        if self.elitism > 0.:
            return np.concatenate((x_elite, x_rest[indices]), 0), np.concatenate((f_elite, f_rest[indices]), 0)
        else:
            return x_rest[indices], f_rest[indices]


class AcceptanceSelection(Selection):
    def __init__(self):
        super().__init__(elitism=0.)

    def select(self, x_old, f_old, x_new, f_new, objective_is_probability, population_size=None):
        if objective_is_probability is False:
            exp_f_old = np.exp(-f_old)
            exp_f_new = np.exp(-f_new)
        else:
            exp_f_old = f_old
            exp_f_new = f_new

        # Sample u ~ Uniform[0,1]
        u = np.random.rand(x_old.shape[0], )

        # Calculate acceptance probability
        A = np.minimum(1., np.exp(np.log(exp_f_new) - np.log(exp_f_old)))

        # If u < A, then accept new, otherwise take old
        accepted = u < A
        previous = ~accepted

        x = np.concatenate((x_old[previous], x_new[accepted]), 0)
        f = np.concatenate((f_old[previous], f_new[accepted]), 0)

        return x, f


class LikelihoodFreeAcceptanceSelection(Selection):
    def __init__(self):
        super().__init__(elitism=0.)

    def select(self, x_old, f_old, x_new, f_new, objective_is_probability, epsilon=np.infty, population_size=None):
        if objective_is_probability is False:
            exp_f_old = np.exp(-f_old)
            exp_f_new = np.exp(-f_new)
        else:
            exp_f_old = f_old
            exp_f_new = f_new

        # ||f_old - f_new||
        indices = f_new < epsilon

        if np.sum(indices > 0):
            x_old_lf = x_old[indices]
            x_new_lf = x_new[indices]
            f_old_lf = f_old[indices]
            f_new_lf = f_new[indices]
            exp_f_old_lf = exp_f_old[indices]
            exp_f_new_lf = exp_f_new[indices]

            # Sample u ~ Uniform[0,1]
            u = np.random.rand(x_old_lf.shape[0], )

            # Calculate acceptance probability
            A = np.minimum(1., np.exp(np.log(exp_f_new_lf) - np.log(exp_f_old_lf)))

            # If u < A, then accept new, otherwise take old
            accepted = u < A
            previous = ~accepted

            x_lf = np.concatenate((x_old_lf[previous], x_new_lf[accepted]), 0)
            f_lf = np.concatenate((f_old_lf[previous], f_new_lf[accepted]), 0)

            return np.concatenate((x_lf, x_old[~indices]), 0), np.concatenate((f_lf, f_old[~indices]), 0)


class LikelihoodFreeAcceptanceUniformSelection(Selection):
    def __init__(self):
        super().__init__(elitism=0.)

    def select(self, x_old, f_old, x_new, f_new, epsilon=np.infty, population_size=None):
        # ||f_old - f_new|| < epsilon
        indices = f_new < epsilon

        if np.sum(indices) > 0:
            return np.concatenate((x_new[indices], x_old[~indices]), 0), np.concatenate((f_new[indices], f_old[~indices]), 0)
        else:
            return x_old, f_old


class LikelihoodFreeAcceptanceGreedySelection(Selection):
    def __init__(self):
        super().__init__(elitism=0.)

    def select(self, x_old, f_old, x_new, f_new, epsilon=np.infty, population_size=None):
        # f_new < epsilon
        indices = np.array(f_new < epsilon)
        # f_new < f_old
        indices_greedy = np.array(f_new < f_old)
        indices_greedy[indices] = False

        if np.sum(indices) > 0 and np.sum(indices_greedy) > 0:
            return np.concatenate((x_new[indices], x_new[indices_greedy], x_old[~indices_greedy]), 0), np.concatenate((f_new[indices], f_new[indices_greedy], f_new[~indices_greedy]), 0)
        elif np.sum(indices) == 0 and np.sum(indices_greedy) > 0:
            return np.concatenate((x_new[indices_greedy], x_old[~indices_greedy]), 0), np.concatenate((f_new[indices_greedy], f_new[~indices_greedy]), 0)
        elif np.sum(indices) > 0 and np.sum(indices_greedy) == 0:
            return np.concatenate((x_new[indices], x_old[~indices]), 0), np.concatenate((f_new[indices], f_new[~indices]), 0)
        else:
            return x_old, f_old


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


class RevGreedy(Selection):
    def __init__(self):
        super().__init__(elitism=0.)

    def select(self, x, f, x_new, f_new, indices, population_size=None):
        if population_size is None:
            population_size = x.shape[0]

        x_1 = x[indices[0]]
        x_2 = x[indices[1]]
        x_3 = x[indices[2]]

        f_1 = f[indices[0]]
        f_2 = f[indices[0]]
        f_3 = f[indices[0]]

        x_new_1 = x_new[0]
        x_new_2 = x_new[1]
        x_new_3 = x_new[2]

        f_new_1 = f_new[0]
        f_new_2 = f_new[1]
        f_new_3 = f_new[2]

        indx = (f_new_1 * f_new_2 * f_new_3) < (f_1 * f_2 * f_3)

        y = np.concatenate((x_1[~indx], x_2[~indx], x_3[~indx]), 0)
        f_y = np.concatenate((f_1[~indx], f_2[~indx], f_3[~indx]), 0)

        z = np.concatenate((x_new_1[indx], x_new_2[indx], x_new_3[indx]), 0)
        f_z = np.concatenate((f_new_1[indx], f_new_2[indx], f_new_3[indx]), 0)

        x_cat = np.concatenate((y, z), 0)
        f_cat = np.concatenate((f_y, f_z), 0)

        # final greedy selection
        indices_final = np.argsort(f_cat)

        x_final = x_cat[indices_final]
        f_final = f_cat[indices_final]

        return x_final[0:population_size], f_final[0:population_size]

    def select2(self, x, f, x_new, f_new):
        x_1, x_2, x_3 = np.split(x, 3)
        f_1, f_2, f_3 = np.split(f, 3)

        x_new_1, x_new_2, x_new_3 = np.split(x_new, 3)
        f_new_1, f_new_2, f_new_3 = np.split(f_new, 3)

        indx = (f_new_1 * f_new_2 * f_new_3) < (f_1 * f_2 * f_3)

        y = np.concatenate((x_1[~indx], x_2[~indx], x_3[~indx]), 0)
        f_y = np.concatenate((f_1[~indx], f_2[~indx], f_3[~indx]), 0)

        z = np.concatenate((x_new_1[indx], x_new_2[indx], x_new_3[indx]), 0)
        f_z = np.concatenate((f_new_1[indx], f_new_2[indx], f_new_3[indx]), 0)

        x_cat = np.concatenate((y, z), 0)
        f_cat = np.concatenate((f_y, f_z), 0)

        # final greedy selection
        indices_final = np.argsort(f_cat)

        x_final = x_cat[indices_final]
        f_final = f_cat[indices_final]

        return x_final, f_final
        # return x_final[0:population_size], f_final[0:population_size]