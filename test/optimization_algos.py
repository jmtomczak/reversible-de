import time
import csv
from json import load
from math import ceil
import numpy as np
import scipy
import scipy.optimize as opt
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from optimization.evolutionary_algorithm import EvolutionaryAlgorithm as EA
from optimization.population_algorithm import MetropolisHastings, Powell, Evolutionary


def MichaelisMenten(t, y, params):
    S = params['S_0'] - y
    dP_dt = ( (params['k_cat'] * params['E_0']) * S ) / ( params['K_M'] + S )
    return dP_dt


def solve_MichaelisMenten(params):
    # we need to use lambda function if we want to pass some parameters
    solution = solve_ivp(lambda t, y: MichaelisMenten(t, y, params),
                         t_span = (params['t0'], params['t1']), y0 = params['y0'],
                         method='RK45', t_eval=params['t_points'])
    y_points = np.asarray(solution.y[0,:])
    return params['t_points'], y_points


def loss(y_real, y_model):
    return np.linalg.norm(y_real - y_model, ord=2) / y_real.shape[0]


def objective(x, *args):
    params = args[0].copy()
    params['k_cat'] = x[0]
    params['K_M'] = x[1]
    _, y_model = solve_MichaelisMenten(params)
    return loss(args[1], y_model)


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


def args2params(args, t_points, x0, y0):
    params = {}
    params['E_0'] = args['E_0']
    params['S_0'] = args['S_0']

    params['k_cat'] = x0[0]
    params['K_M'] = x0[1]

    params['t0'] = t_points[0]
    params['t1'] = t_points[-1] + 1.
    params['y0'] = y0
    params['t_points'] = t_points

    return params


def smooth_data(args, t_points, P_real):
    # linear function
    def fun(x, a):
        return a*x

    # fitting
    x = np.expand_dims( t_points, 1 )
    y = np.expand_dims( P_real, 1 )
    I = 0.0000001*np.diag( np.ones(shape=(x.shape[1],) ) )

    a = np.squeeze(np.dot( np.dot( np.linalg.pinv( np.dot(x.T, x) + I ), x.T ), y), 1)

    # smoothed data
    # P_fit = np.repeat( np.expand_dims(fun(t_points, a),0), args['N'], axis=0 )
    P_fit = fun(t_points, a)
    return P_fit


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

if __name__ == '__main__':
    json_name = 'hAPN.json'
    with open(json_name) as f:
        config = load(f)
    args = config2args(config)

    y_real, t_real = load_data(args, 0)
    y_real = smooth_data(args, t_real, y_real)

    # bounds = [(0., 100. * 60.), (0., 200.)]
    # x0 = np.array([np.random.uniform(bounds[0][0], bounds[0][1]),  # k_cat
    #                np.random.uniform(bounds[1][0], bounds[1][1])])  # K_M)

    # POWELL
    # tic = time.time()
    # res = opt.minimize(objective, x0, args=(params, y_real), method='powell', options={'xtol': 1e-8, 'disp': False}) # OK
    # toc = time.time()
    # params['k_cat'] = res.x[0]
    # params['K_M'] = res.x[1]
    # _, y_powell = solve_MichaelisMenten(params)
    # print('Powell method: k_cat=', params['k_cat']/60., 'K_M=', params['K_M'], 'time elapsed=', toc-tic)

    # # DIFFERENTIAL EVOLUTION
    # tic = time.time()
    # res = opt.differential_evolution(objective, bounds, args=(params, y_real))  # OK
    # toc = time.time()
    # params['k_cat'] = res.x[0]
    # params['K_M'] = res.x[1]
    # _, y_de = solve_MichaelisMenten(params)
    # print('Differential evolution: k_cat=', params['k_cat'] / 60., 'K_M=', params['K_M'], 'time elapsed=', toc - tic)
    #
    # # DUAL ANNEALING
    # tic = time.time()
    # res = opt.dual_annealing(objective, bounds, args=(params, y_real))  # OK
    # toc = time.time()
    # params['k_cat'] = res.x[0]
    # params['K_M'] = res.x[1]
    # _, y_annealing = solve_MichaelisMenten(params)
    # print('Dual annealing: k_cat=', params['k_cat'] / 60., 'K_M=', params['K_M'], 'time elapsed=', toc - tic)

    # INIT
    bounds = [[0., 0], [100.*60, 200.]]

    pop_size = 100

    x0 = np.concatenate(
        (np.asarray([np.random.uniform(bounds[0][i], bounds[1][i], (pop_size, 1)) for i in range(len(bounds[0]))])), 1)

    params = args2params(args, t_real, x0=x0, y0=[0.])

    # x0 = np.concatenate((np.random.uniform(bounds[0][0], bounds[0][1], (pop_size, 1)),  # k_cat
    #                np.random.uniform(bounds[1][0], bounds[1][1], (pop_size, 1))), 1)  # K_M

    # x0 = np.asarray([np.random.uniform(bounds[0][i], bounds[1][i]) for i in range(len(bounds[0]))])


    cov_mat = np.eye(2, 2)
    cov_mat[0, 0] *= 20. ** 2
    cov_mat[1, 1] *= 5. ** 2

    algos = ['ea'] #['powell', 'mh', 'ea']

    if 'powell' in algos:
        tic = time.time()
        pow = Powell(fun=objective, x0=x0[0,:], bounds=bounds, args=(params, y_real), num_epochs=300)
        res, f = pow.sample(epsilon=1.)

        plt.hexbin(res[:,0] / 60., res[:,1], gridsize=20)
        plt.show()

        toc = time.time()
        params['k_cat'] = np.mean(res[:,0])
        params['K_M'] = np.mean(res[:,1])
        _, y_powell = solve_MichaelisMenten(params)
        print('Powell method: k_cat=', params['k_cat'] / 60., ' (', np.std(res[:,0]) / 60., ') ', 'K_M=', params['K_M'], ' (', np.std(res[:,1]), ') ', 'time elapsed=', toc - tic)


    # tic = time.time()
    # ea = EA(objective, args=(params, y_real), x_0=x0, population_size=pop_size, num_iters=100, mutation_prob=1.,
    #         mutation_cov=cov_mat, recombination_prob=0.5, recombination_type='k-points', recombination_num_cutting_points=1, selection_elitism=0.2)
    # res, f_value = ea.optimize()
    # toc = time.time()
    # params['k_cat'] = res[0]
    # params['K_M'] = res[1]
    # print('EA: k_cat=', params['k_cat'] / 60., 'K_M=', params['K_M'], 'time elapsed=', toc - tic)

    # x_sample = ea.sample(epsilon=0.0005)
    # x_sample[:, 0] /= 60
    # print(np.mean(x_sample, 0))
    # print(np.std(x_sample, 0))

    if 'mh' in algos:
        tic = time.time()
        params['cov'] = cov_mat
        mh = MetropolisHastings(objective, args=(params, y_real), x0=x0, burn_in_phase=48, num_epochs=50, bounds=([bounds[0][0], bounds[0][1]], [bounds[1][0], bounds[1][1]]) )
        res, f_res = mh.sample(epsilon=1.)
        toc = time.time()

        # plt.hist2d(res[:,0] / 60., res[:,1], bins=50)
        plt.hexbin(res[:,0] / 60., res[:,1], gridsize=20)
        plt.show()

        params['k_cat'] = np.mean(res[:,0])
        params['K_M'] = np.mean(res[:,1])
        print('Metropolis-Hastings: k_cat=', params['k_cat'] / 60., ' (', np.std(res[:,0]) / 60., ') ', 'K_M=', params['K_M'], ' (', np.std(res[:,1]), ') ', 'time elapsed=', toc - tic)

    if 'ea' in algos:
        tic = time.time()
        params['cov'] = cov_mat
        params['mixing_prob'] = 1.
        params['num_cutting_points'] = 4
        ea = Evolutionary(objective, args=(params, y_real), x0=x0, burn_in_phase=48, num_epochs=50,
                          bounds=([bounds[0][0], bounds[0][1]], [bounds[1][0], bounds[1][1]]),
                          mixing_proposal_type='k-points')
        res, f_res = ea.sample(epsilon=1.)
        toc = time.time()

        # plt.hist2d(res[:,0] / 60., res[:,1], bins=50)
        plt.hexbin(res[:, 0] / 60., res[:, 1], gridsize=20)
        plt.show()

        params['k_cat'] = np.mean(res[:, 0])
        params['K_M'] = np.mean(res[:, 1])
        print('Evolutionary: k_cat=', params['k_cat'] / 60., ' (', np.std(res[:, 0]) / 60., ') ', 'K_M=',
              params['K_M'], ' (', np.std(res[:, 1]), ') ', 'time elapsed=', toc - tic)
