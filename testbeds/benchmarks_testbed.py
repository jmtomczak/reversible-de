import numpy as np

from .testbed import TestBed


# ----------------------------------------------------------------------------------------------------------------------
class BenchmarkFun(TestBed):
    def __init__(self, name='schwefel'):
        super().__init__()
        self.name = name

    def objective(self, x, *args):
        if self.name == 'schwefel':
            f = 418.9829 * x.shape[1] - np.sum(x * np.sin(np.sqrt(np.abs(x))), 1)
        elif self.name == 'rastrigin':
            f = 10 * x.shape[1] + np.sum(x**2 - 10 * np.cos(2. * np.pi * x), 1)
        elif self.name == 'griewank':
            denumerator = np.sqrt(np.arange(x.shape[1]) + 1.)
            f = 1 + np.sum(x ** 2 / 4000., 1) - np.prod(np.cos(x / denumerator), 1)
        elif self.name == 'salomon':
            xx = np.sqrt(np.sum(x**2, 1))
            f = 1 - np.cos(2.*np.pi*xx) + 0.1 * xx
        else:
            raise ValueError('Wrong name!')
        return f
