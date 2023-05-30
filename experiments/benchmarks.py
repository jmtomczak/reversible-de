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

PYTHONPATH = '/Users/jmt/Dev/github/reversible-de/experiments'

sys.path.append(os.path.dirname(os.path.expanduser(PYTHONPATH)))

from ..algorithms.population_algorithm import OptimizationAlgorithm, DifferentialEvolution
from ..testbeds.benchmarks_testbed import BenchmarkFun


if __name__ == '__main__':

    # INIT: general hyperparams
    names = ['griewank', 'rastrigin', 'salomon', 'schwefel']

    Ds = [10, 30, 100]
    Fs = [0.125, 0.25, 0.375, 0.5, 0.6, 0.625, 0.675, 0.75]

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

                num_generations = 150

                num_repetitions = 10

                de_types = ['de', 'dex3', 'ade', 'revde']

                results_dir = '../results/' + name + '_D' + str(D) + '_F_' + str(F) + '_pop_' + str(pop_size)

                final_results = {}

                for de_type in de_types:
                    print(f"------- Now runs: {de_type} -------")
                    for rep in range(num_repetitions):
                        print(f"\t-> repetition {rep}")

                        np.random.seed(seed=rep)

                        x0 = np.random.uniform(low=bounds[0], high=bounds[1], size=(pop_size, D))

                        # Experiment
                        b_fun = BenchmarkFun(name=name)
                        objective = b_fun.objective
                        params = {}

                        params['evaluate_objective_type'] = 'full'
                        params['pop_size'] = pop_size
                        params['CR'] = 0.9
                        params['F'] = F

                        de = DifferentialEvolution(objective,
                                                   args=(params, None),
                                                   x0=x0,
                                                   bounds=(bounds[0], bounds[1]),
                                                   population_size=pop_size,
                                                   de_type=de_type)

                        opt_alg = OptimizationAlgorithm(pop_algorithm=de, num_epochs=num_generations)

                        specific_folder = '-' + de_type + '-F-' + str(params['F']) + '-pop_size-' + str(params['pop_size']) + '-epochs-' + str(num_generations)
                        directory_name = results_dir + '/' + opt_alg.name + specific_folder + '-r' + str(rep)

                        if os.path.exists(directory_name):
                            directory_name = directory_name + str(datetime.now())

                        directory_name = directory_name + '/'
                        os.makedirs(directory_name)

                        tic = time.time()
                        res, f = opt_alg.optimize(directory_name=directory_name)
                        toc = time.time()

                        # Plot of best results
                        f_best = np.load(directory_name + 'f_best.npy')

                        plt.plot(np.arange(0, len(f_best)), np.array(f_best))
                        plt.grid()
                        plt.savefig(directory_name + '/' + 'best_f.pdf')
                        plt.close()

                        print('\tTime elapsed:', toc - tic)

                    # Average best results
                    directory_name_avg = directory_name[:-4]
                    for r in range(num_repetitions):
                        dir = directory_name_avg + '-r' + str(r)
                        if r == 0:
                            f_best_avg = np.load(dir + '/' + 'f_best.npy')
                        else:
                            f_best_avg = np.concatenate((f_best_avg, np.load(dir + '/' + 'f_best.npy')), 0)

                    f_best_avg = np.reshape(f_best_avg, (num_repetitions, num_generations + 1))

                    # plotting
                    x_epochs = np.arange(0, f_best_avg.shape[1])
                    y_f = f_best_avg.mean(0)
                    y_f_std = f_best_avg.std(0)

                    final_results[de_type + '_avg'] = y_f
                    final_results[de_type + '_std'] = y_f_std

                    plt.plot(x_epochs, y_f)
                    plt.fill_between(x_epochs, y_f - y_f_std, y_f + y_f_std)
                    plt.grid()
                    plt.xlim(0, num_generations)
                    plt.savefig(results_dir + '/' + opt_alg.name + de_type + '_best_f_avg.pdf')
                    plt.close()

                # save final results (just in case!)
                f = open(results_dir + '/' + name + 'D' + str(D) + '.pkl', "wb")
                pickle.dump(final_results, f)
                f.close()

                colors = ['#e6194B', '#ffe119', '#3cb44b', '#4363d8']
                linestyles = ['-', '-', '-.', 'dotted']
                labels = ['DE', 'DEx3', 'ADE', 'RevDE']
                lw = 3.
                iter = 0
                for de_type in de_types:
                    plt.plot(x_epochs, final_results[de_type + '_avg'], colors[iter], ls=linestyles[iter], lw=lw, label=labels[iter])
                    plt.fill_between(x_epochs, final_results[de_type + '_avg'] - final_results[de_type + '_std'],
                                     final_results[de_type + '_avg'] + final_results[de_type + '_std'], color=colors[iter], alpha=0.5)

                    iter += 1

                plt.grid()
                plt.xlabel('epochs')
                plt.ylabel('objective')
                plt.legend(loc=0)
                plt.xlim(0, num_generations)
                plt.savefig(results_dir + '/' + '_best_f_comparison.pdf')
                plt.close()