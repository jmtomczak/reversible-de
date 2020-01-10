import csv
import numpy as np
from scipy.integrate import solve_ivp
from math import ceil
from json import load

from testbeds.testbed import TestBed


class Repressilator(TestBed):
    def __init__(self):
        super().__init__()

    @staticmethod
    def repressilator_model(t, y, params):

        m1, m2, m3, p1, p2, p3 = y[0], y[1], y[2], y[3], y[4], y[5]

        alpha0 = params['alpha0']
        n = params['n']
        beta = params['beta']
        alpha = params['alpha']

        dm1_dt = -m1 + alpha / (1. + p3**n) + alpha0
        dp1_dt = -beta * (p1 - m1)
        dm2_dt = -m2 + alpha / (1. + p1**n) + alpha0
        dp2_dt = -beta * (p2 - m2)
        dm3_dt = -m3 + alpha / (1. + p2**n) + alpha0
        dp3_dt = -beta * (p3 - m3)

        return dm1_dt, dm2_dt, dm3_dt, dp1_dt, dp2_dt, dp3_dt

    def solve_repressilator(self, params):
        # we need to use lambda function if we want to pass some parameters
        solution = solve_ivp(lambda t, y: self.repressilator_model(t, y, params),
                             t_span=(params['t0'], params['t1']), y0=params['y0'],
                             method='RK45', t_eval=params['t_points'])
        y_points = np.asarray(solution.y)
        return params['t_points'], y_points

    @staticmethod
    def config2args(config):
        args = {}
        args['h'] = float(config["experiment_details"]["h"]) / 60.
        args['exp_last'] = float(config["experiment_details"]["exp_last"])

        args['low_alpha0'] = float(config["abc_details"]["low_alpha0"])
        args['high_alpha0'] = float(config["abc_details"]["high_alpha0"])

        args['low_n'] = float(config["abc_details"]["low_n"])
        args['high_n'] = float(config["abc_details"]["high_n"])

        args['low_beta'] = float(config["abc_details"]["low_beta"])
        args['high_beta'] = float(config["abc_details"]["high_beta"])

        args['low_alpha'] = float(config["abc_details"]["low_alpha"])
        args['high_alpha'] = float(config["abc_details"]["high_alpha"])

        args['bounds'] = [[args['low_alpha0'], args['low_n'], args['low_beta'], args['low_alpha']],
                          [args['high_alpha0'], args['high_n'], args['high_beta'], args['high_alpha']]]

        return args

    @staticmethod
    def args2params(args, t_points, x0, y0):
        params = {}

        params['x0'] = x0

        params['alpha0'] = x0[0, 0]
        params['n'] = x0[0, 1]
        params['beta'] = x0[0, 2]
        params['alpha'] = x0[0, 3]

        params['bounds'] = args['bounds']

        params['t0'] = t_points[0]
        params['t1'] = t_points[-1] + 1.
        params['y0'] = y0
        params['t_points'] = t_points

        return params

    @staticmethod
    def smooth_data(args, t_points, P_real):
        # linear function
        def fun(x, a):
            return a * x

        # fitting
        x = np.expand_dims(t_points, 1)
        y = np.expand_dims(P_real, 1)
        I = 0.0000001 * np.diag(np.ones(shape=(x.shape[1],)))

        a = np.squeeze(np.dot(np.dot(np.linalg.pinv(np.dot(x.T, x) + I), x.T), y), 1)

        # smoothed data
        # P_fit = np.repeat( np.expand_dims(fun(t_points, a),0), args['N'], axis=0 )
        P_fit = fun(t_points, a)
        return P_fit

    @staticmethod
    def loss(y_real, y_model):
        # we assume only m's are observed!
        y_r = y_real[0:3]
        y_m = y_model[0:3]
        if y_r.shape[1] == y_m.shape[1]:
            return np.mean(np.sum(np.sqrt((y_r - y_m)**2), 0))
        else:
            return np.infty

    def create_data(self, pop_size=10, alpha0=1, n=2., beta=5, alpha=1000., json_name='repressilator.json'):
        with open(json_name) as f:
            config = load(f)
        args = self.config2args(config)

        # time steps
        t_points = np.arange(0, args['h'] * args['exp_last'] // args['h'], args['h'])

        # init candidate solutions
        x0 = np.random.uniform(low=args['bounds'][0], high=args['bounds'][1], size=(pop_size, 4))

        params = self.args2params(args, t_points, x0=x0, y0=[0., 0., 0., 2., 1., 3.])

        # # SYNTHETIC DATA
        params['alpha0'] = alpha0
        params['n'] = n
        params['beta'] = beta
        params['alpha'] = alpha

        _, y_real = self.solve_repressilator(params)

        y_real = y_real + np.random.rand(y_real.shape[0], y_real.shape[1]) * 5.

        return y_real, params

    def objective(self, x, *args):
        params = args[0].copy()
        params['alpha0'] = x[0]
        params['n'] = x[1]
        params['beta'] = x[2]
        params['alpha'] = x[3]

        _, y_model = self.solve_repressilator(params)
        return self.loss(args[1], y_model)


if __name__ == '__main__':

    r = Repressilator()

    y_real, params = r.create_data()

    print(y_real.shape)
    print(params)

    import matplotlib.pyplot as plt

    plt.plot(params['t_points'], y_real[0])
    plt.plot(params['t_points'], y_real[1])
    plt.plot(params['t_points'], y_real[2])
    plt.show()
    plt.close()

    plt.plot(params['t_points'], y_real[3])
    plt.plot(params['t_points'], y_real[4])
    plt.plot(params['t_points'], y_real[5])
    plt.show()
    plt.close()