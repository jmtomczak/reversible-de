import os
import time
import pickle
import matplotlib

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from optimization.evolutionary_algorithm import EvolutionaryAlgorithm as EA
from optimization.population_algorithm import MetropolisHastings, Powell, Evolutionary, PopLiFe, \
    LikelihoodFreeInference, \
    ReversiblePopLiFe

from testbeds.mm_testbed import MichaelisMenten

if __name__ == '__main__':

    # INIT: general hyperparams
    D = 2

    bounds = [[0., 0], [100. * 60, 200.]]

    pop_size = 50

    num_epochs = 20

    max_iter = 500

    F = 2.0

    cov_mat = np.eye(2, 2)
    cov_mat[0, 0] *= 25. ** 2
    cov_mat[1, 1] *= 10. ** 2

    epsilon = 0.0001

    # run experiments
    num_repetitions = 10

    proposal_types = ['differential_1', 'de_times_3', 'antisymmetric_differential', 'differential_3']

    results_dir = '../results/MichaelisMenten_F_' + str(F) + '_pop_' + str(pop_size)

    final_results = {}

    for de_proposal_type in proposal_types:
        print(f"------- Now runs: {de_proposal_type} -------")
        for rep in range(num_repetitions):
            print(f"\t-> repetition {rep}")

            np.random.seed(seed=rep)

            # x0 = np.concatenate(
            #     (np.asarray([np.random.uniform(bounds[0][i], bounds[1][i], (pop_size, 1)) for i in range(len(bounds[0]))])), 1)
            x0 = np.random.uniform(low=bounds[0], high=bounds[1], size=(pop_size, D))

            # Michaelis-Menten experiment
            mm = MichaelisMenten()
            y_real, params = mm.create_data(x0)
            objective = mm.objective

            params['evaluate_objective_type'] = 'single'

            params['cov'] = cov_mat
            params['gaussian_prob'] = .0
            params['mixing_prob'] = 0.0
            params['num_cutting_points'] = 0
            params['CR'] = 0.999
            params['F'] = F
            params['randomized_F'] = False

            revpoplife = ReversiblePopLiFe(objective, args=(params, y_real), x0=x0, burn_in_phase=0, num_epochs=num_epochs,
                                           bounds=([bounds[0][0], bounds[0][1]], [bounds[1][0], bounds[1][1]]),
                                           population_size=pop_size,
                                           elitism=0.,
                                           de_proposal_type=de_proposal_type)

            lf_poplife = LikelihoodFreeInference(pop_algorithm=revpoplife, num_epochs=num_epochs)

            specific_folder = '-' + de_proposal_type + '-F-' + str(params['F']) + '-pop_size-' + str(params['pop_size']) + '-epochs-' + str(num_epochs)
            directory_name = results_dir + '/' + lf_poplife.name + specific_folder + '-r' + str(rep)

            if os.path.exists(directory_name):
                directory_name = directory_name + str(datetime.now())

            directory_name = directory_name + '/'
            os.makedirs(directory_name)

            tic = time.time()
            res, f = lf_poplife.lf_inference(directory_name=directory_name, epsilon=epsilon)
            toc = time.time()

            # Histogram
            plt.hexbin(res[:, 0] / 60., res[:, 1], gridsize=20)
            plt.colorbar()
            plt.savefig(directory_name + 'histogram')
            plt.close()

            # Plot of best results
            f_best = np.load(directory_name + 'f_best.npy')

            plt.plot(np.arange(0, len(f_best)), np.array(f_best))
            plt.grid()
            plt.savefig(directory_name + '/' + 'best_f')
            plt.close()

            params['k_cat'] = np.mean(res[:, 0])
            params['K_M'] = np.mean(res[:, 1])
            print('\tResult: k_cat=', params['k_cat'] / 60., ' (', np.std(res[:, 0]) / 60., ') ', 'K_M=',
                  params['K_M'], ' (', np.std(res[:, 1]), ') ', 'time elapsed=', toc - tic)

        # Average best results
        directory_name_avg = directory_name[:-4]
        for r in range(num_repetitions):
            dir = directory_name_avg + '-r' + str(r)
            if r == 0:
                f_best_avg = np.load(dir + '/' + 'f_best.npy')
            else:
                f_best_avg = np.concatenate((f_best_avg, np.load(dir + '/' + 'f_best.npy')), 0)

        f_best_avg = np.reshape(f_best_avg, (num_repetitions, num_epochs+1))

        # plotting
        x_epochs = np.arange(0, f_best_avg.shape[1])
        y_f = f_best_avg.mean(0)
        y_f_std = f_best_avg.std(0)

        final_results[de_proposal_type + '_avg'] = y_f
        final_results[de_proposal_type + '_std'] = y_f_std

        plt.plot(x_epochs, y_f)
        plt.fill_between(x_epochs, y_f - y_f_std, y_f + y_f_std)
        plt.grid()
        plt.savefig(results_dir + '/' + lf_poplife.name + de_proposal_type + '_best_f_avg')
        plt.close()

    # save final results (just in case!)
    f = open(results_dir + '/' + 'michaelis_menten.pkl', "wb")
    pickle.dump(final_results, f)
    f.close()

    colors = ['#e6194B', '#ffe119', '#3cb44b', '#4363d8']
    linestyles = ['-', '-', '-.', 'dotted']
    labels = ['DE', 'DEx3', 'ADE', 'RevDE']
    lw = 3.
    iter = 0
    for de_proposal_type in proposal_types:
        plt.plot(x_epochs, final_results[de_proposal_type + '_avg'], colors[iter], ls=linestyles[iter], lw=lw,
                 label=labels[iter])
        plt.fill_between(x_epochs, final_results[de_proposal_type + '_avg'] - final_results[de_proposal_type + '_std'],
                         final_results[de_proposal_type + '_avg'] + final_results[de_proposal_type + '_std'],
                         color=colors[iter], alpha=0.5)

        iter += 1

    plt.grid()
    plt.xlabel('epochs')
    plt.ylabel('objective')
    plt.legend(loc=0)
    plt.savefig(results_dir + '/' + '_best_f_comparison')
    plt.close()