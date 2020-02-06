import os
import sys
import time
import pickle
import matplotlib

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

PYTHONPATH = '/home/jakub/Dev/github/life/experiments'  # ripper5
# PYTHONPATH = '/Users/jmt/Dev/github/life/experiments'  #mac

sys.path.append(os.path.dirname(os.path.expanduser(PYTHONPATH)))

from optimization.population_algorithm import LikelihoodFreeInference, ReversiblePopLiFe

from testbeds.benchmarks_testbed import BenchmarkFun

if __name__ == '__main__':

    # INIT: general hyperparams
    names = ['griewank', 'rastrigin', 'salomon', 'schwefel']
    # names = ['schwefel']

    Ds = [10, 30, 100]
    # Fs = [1.0, 1.5, 2.0]
    Fs = [2.5]

    for name in names:
        for D in Ds:
            for F in Fs:

                if name == 'schwefel':
                    bounds = [[200.] * D, [500] * D]
                elif name == 'rastrigin':
                    bounds = [[-5.] * D, [5.] * D]
                elif name == 'griewank':
                    bounds = [[-5.] * D, [5.] * D]
                elif name == 'salomon':
                    bounds = [[-5.] * D, [5.] * D]
                else:
                    raise ValueError('Wrong name')

                pop_size = 500

                num_epochs = 150

                max_iter = 500

                cov_mat = np.eye(D, D)

                epsilon = np.infty

                # run experiments
                num_repetitions = 10

                # proposal_types = ['differential_1', 'de_times_3', 'antisymmetric_differential', 'differential_3']
                proposal_types = ['antisymmetric_differential', 'de_times_3']

                results_dir = '../results/' + name + '_D' + str(D) + '_F_' + str(F) + '_pop_' + str(pop_size)

                final_results = {}

                for de_proposal_type in proposal_types:
                    print(f"------- Now runs: {de_proposal_type} -------")
                    for rep in range(num_repetitions):
                        print(f"\t-> repetition {rep}")

                        np.random.seed(seed=rep)

                        # x0 = np.concatenate(
                        #     (np.asarray([np.random.uniform(bounds[0][i], bounds[1][i], (pop_size, 1)) for i in range(len(bounds[0]))])), 1)
                        x0 = np.random.uniform(low=bounds[0], high=bounds[1], size=(pop_size, D))

                        # Experiment
                        b_fun = BenchmarkFun(name=name)
                        objective = b_fun.objective
                        params = {}

                        params['evaluate_objective_type'] = 'full'

                        params['pop_size'] = pop_size
                        params['cov'] = cov_mat
                        params['gaussian_prob'] = .0
                        params['mixing_prob'] = 0.0
                        params['num_cutting_points'] = 0
                        params['CR'] = 0.9
                        params['F'] = F
                        params['randomized_F'] = False

                        revpoplife = ReversiblePopLiFe(objective, args=(params, None), x0=x0, burn_in_phase=0, num_epochs=num_epochs,
                                                       bounds=(bounds[0], bounds[1]),
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

                        # Plot of best results
                        f_best = np.load(directory_name + 'f_best.npy')

                        plt.plot(np.arange(0, len(f_best)), np.array(f_best))
                        plt.grid()
                        plt.savefig(directory_name + '/' + 'best_f.pdf')
                        plt.close()

                        # print('\tResult: ', res.mean(0), 'time elapsed=', toc - tic)
                        print('\tTime elapsed:', toc - tic)

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
                    plt.xlim(0, num_epochs)
                    plt.savefig(results_dir + '/' + lf_poplife.name + de_proposal_type + '_best_f_avg.pdf')
                    plt.close()

                # save final results (just in case!)
                f = open(results_dir + '/' + name + 'D' + str(D) + '.pkl', "wb")
                pickle.dump(final_results, f)
                f.close()

                # colors = ['b', 'r', 'c', 'g']
                # colors = ['#f58231', '#3cb44b', '#4363d8', '#f032e6']
                colors = ['#e6194B', '#ffe119', '#3cb44b', '#4363d8']
                linestyles = ['-', '-', '-.', 'dotted']
                labels = ['DE', 'DEx3', 'ADE', 'RevDE']
                lw = 3.
                iter = 0
                for de_proposal_type in proposal_types:
                    plt.plot(x_epochs, final_results[de_proposal_type + '_avg'], colors[iter], ls=linestyles[iter], lw=lw, label=labels[iter])
                    plt.fill_between(x_epochs, final_results[de_proposal_type + '_avg'] - final_results[de_proposal_type + '_std'],
                                     final_results[de_proposal_type + '_avg'] + final_results[de_proposal_type + '_std'], color=colors[iter], alpha=0.5)

                    iter += 1

                plt.grid()
                plt.xlabel('epochs')
                plt.ylabel('objective')
                plt.legend(loc=0)
                plt.xlim(0, num_epochs)
                plt.savefig(results_dir + '/' + '_best_f_comparison.pdf')
                plt.close()