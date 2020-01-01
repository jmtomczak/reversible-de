import csv
import numpy as np
from scipy.integrate import solve_ivp
from math import ceil
from json import load

from testbeds.testbed import TestBed


class MichaelisMenten(TestBed):
    def __init__(self):
        super().__init__()

    @staticmethod
    def MichaelisMenten(t, y, params):
        S = params['S_0'] - y
        dP_dt = ((params['k_cat'] * params['E_0']) * S) / (params['K_M'] + S)
        return dP_dt

    def solve_MichaelisMenten(self, params):
        # we need to use lambda function if we want to pass some parameters
        solution = solve_ivp(lambda t, y: self.MichaelisMenten(t, y, params),
                             t_span=(params['t0'], params['t1']), y0=params['y0'],
                             method='RK45', t_eval=params['t_points'])
        y_points = np.asarray(solution.y[0, :])
        return params['t_points'], y_points

    @staticmethod
    def config2args(config):
        args = {}
        args['file'] = str(config["file_name"]["file"])
        args['E_0'] = float(config["experiment_details"]["E_0"])
        args['S_0'] = float(config["experiment_details"]["S_0"])
        args['S_1'] = float(config["experiment_details"]["S_1"])
        args['h'] = float(config["experiment_details"]["h"]) / 60.
        args['exp_last'] = float(config["experiment_details"]["exp_last"])

        args['low_K_M'] = float(config["abc_details"]["low_K_M"])
        args['high_K_M'] = float(config["abc_details"]["high_K_M"])
        args['low_k_cat'] = float(config["abc_details"]["low_k_cat"])
        args['high_k_cat'] = float(config["abc_details"]["high_k_cat"])

        args['N'] = int(config["abc_details"]["N"])
        args['No_sim'] = int(config["abc_details"]["No_sim"])
        args['points_prior'] = int(config["abc_details"]["points_prior"])

        return args

    @staticmethod
    def args2params(args, t_points, x0, y0):
        params = {}
        params['E_0'] = args['E_0']
        params['S_0'] = args['S_0']

        params['k_cat'] = x0[0, 0]
        params['K_M'] = x0[0, 1]

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
    def load_data(args, row_value=0):
        file_name = args['file']

        # LOAD DATA
        with open(file_name, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            counter = 0
            v = []
            for row in reader:
                if counter == row_value:
                    for i in range(len(row)):
                        v.append(float(row[i]))
                    break
                counter += 1

        if len(v) == 0:
            raise Exception("Something wrong with data loading! Please check csv file.")

        # set as numpy array
        v = np.asarray(v)

        # how long the experiment lasts (max is 15 min)
        if args['exp_last'] > 15.:
            exp_last = 15.
        else:
            exp_last = args['exp_last']

        # get data
        A = v[0:int(ceil(exp_last / args['h']))]
        # repeat data
        #     P_real = np.repeat(np.expand_dims(np.asarray(A),0), args['N'], axis=0)
        P_real = A
        # time points
        #     t_points = np.arange(0, args['h']*P_real.shape[1], args['h'])
        t_points = np.arange(0, args['h'] * P_real.shape[0], args['h'])

        return P_real, t_points

    @staticmethod
    def loss(y_real, y_model):
        return np.linalg.norm(y_real - y_model, ord=2) / y_real.shape[0]

    def create_data(self, x0, S_0=200., k_cat=40.5 * 60., K_M=15.7, json_name='hAPN.json'):
        with open(json_name) as f:
            config = load(f)
        args = self.config2args(config)

        y_real_raw, t_real = self.load_data(args, 0)
        y_real = self.smooth_data(args, t_real, y_real_raw)

        params = self.args2params(args, t_real, x0=x0, y0=[0.])

        # # SYNTHETIC DATA
        params['S_0'] = S_0
        params['k_cat'] = k_cat
        params['K_M'] = K_M
        params['pop_size'] = x0.shape[0]

        _, y_real = self.solve_MichaelisMenten(params)

        return y_real, params

    def objective(self, x, *args):
        params = args[0].copy()
        params['k_cat'] = x[0]
        params['K_M'] = x[1]
        _, y_model = self.solve_MichaelisMenten(params)
        return self.loss(args[1], y_model)