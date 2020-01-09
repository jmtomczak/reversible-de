import pickle
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sp = '_'
save_dir_name = '../results/'


def load_from_file(file_to_load):
    f = open(file_to_load, "rb")
    final_results = pickle.load(f)
    f.close()
    return final_results


def plot_F(dataset_name='griewank', D=10, pop=500, Fs=[1.0], proposal_types=['differential_1']):

    for proposal in proposal_types:
        linestyles = ['-', '-.', 'dotted']
        iter = 0
        for F in Fs:
            dir_name = '../results/' + dataset_name + sp + 'D' + str(D) + sp + 'F' + sp + str(F) + sp + 'pop' + sp + str(pop)
            file_name = dataset_name + 'D' + str(D) + '.pkl'
            # # For Michaelis-Menten
            # dir_name = '../results/' + dataset_name + sp + 'F' + sp + str(F) + sp + 'pop' + sp + str(pop)
            # file_name = 'michaelis_menten' + '.pkl'

            # load data
            file_to_load = dir_name + '/' + file_name
            final_results = load_from_file(file_to_load)

            x_epochs = np.arange(0, final_results[proposal + '_avg'].shape[0])

            plt.plot(x_epochs, final_results[proposal + '_avg'], lw=3., ls=linestyles[iter], label=str(F))
            plt.fill_between(x_epochs, final_results[proposal + '_avg'] - final_results[proposal + '_std'],
                             final_results[proposal + '_avg'] + final_results[proposal + '_std'], alpha=0.5)
            iter += 1

        plt.grid()
        plt.xlabel('epochs')
        plt.ylabel('objective')
        plt.legend(loc=0)
        plt.savefig(save_dir_name + '/' + dataset_name + sp + 'D' + str(D) + sp + str(proposal) + sp + '_F_comparison.pdf')
        plt.close()


def plot_best(Fs, dataset_name, D=10, pop=500,
              proposal_types=['differential_1', 'de_times_3', 'antisymmetric_differential', 'differential_3'],
              num_epochs=None):
    # Fs are best Fs for proposals in the same order as proposal_types!

    colors = ['#e6194B', '#ffe119', '#3cb44b', '#4363d8']
    linestyles = ['-', '-', '-.', 'dotted']
    labels = ['DE', 'DEx3', 'ADE', 'RevDE']
    iter = 0

    for proposal in proposal_types:
        F = Fs[iter]

        dir_name = '../results/' + dataset_name + sp + 'D' + str(D) + sp + 'F' + sp + str(F) + sp + 'pop' + sp + str(pop)
        file_name = dataset_name + 'D' + str(D) + '.pkl'

        # load data
        file_to_load = dir_name + '/' + file_name
        final_results = load_from_file(file_to_load)

        if num_epochs is None:
            num_epochs = final_results[proposal + '_avg'].shape[0]
        x_epochs = np.arange(0, num_epochs)

        plt.plot(x_epochs, final_results[proposal + '_avg'][:num_epochs], colors[iter], lw=3., ls=linestyles[iter], label=labels[iter])
        plt.fill_between(x_epochs, final_results[proposal + '_avg'][:num_epochs] - final_results[proposal + '_std'][:num_epochs],
                         final_results[proposal + '_avg'][:num_epochs] + final_results[proposal + '_std'][:num_epochs], color=colors[iter], alpha=0.5)
        iter += 1

    plt.grid()
    plt.xlabel('epochs')
    plt.ylabel('objective')
    plt.legend(loc=0)
    plt.savefig(save_dir_name + '/' + dataset_name + sp + 'D' + str(D) + sp + '_best.pdf')
    plt.close()


if __name__ == '__main__':

    # plottings = ['F', 'best']
    plottings = ['best']
    datasets = ['salomon', 'griewank', 'rastrigin', 'schwefel']
    Ds = [10, 30, 100]
    pop = 500

    # For Michaelis-Menten
    # datasets = ['MichaelisMenten']
    # Ds = [2]
    # pop = 50

    proposal_types = ['differential_1', 'de_times_3', 'antisymmetric_differential', 'differential_3']

    for plotting in plottings:
        for data in datasets:
            for D in Ds:
                if plotting == 'F':
                    Fs = [1., 1.5, 2.]
                    plot_F(dataset_name=data, D=D, pop=pop, Fs=Fs, proposal_types=proposal_types)

                elif plotting == 'best':
                    if data == 'griewank':
                        if D == 10:
                            Fs = [1., 1.5, 1., 2.] #D=10
                        elif D == 30:
                            Fs = [1.5, 1.5, 1.5, 2.]  # D=30
                        elif D == 100:
                            Fs = [1.5, 1.5, 1.5, 2.]  # D=100
                        plot_best(dataset_name=data, D=D, pop=pop, Fs=Fs, proposal_types=proposal_types, num_epochs=None)
                    elif data == 'rastrigin':
                        if D == 10:
                            Fs = [1., 1., 1., 2.] #D=10
                        elif D == 30:
                            Fs = [1., 1., 1., 2.]  # D=30
                        elif D == 100:
                            Fs = [1., 1., 1., 2.]  # D=100
                        plot_best(dataset_name=data, D=D, pop=pop, Fs=Fs, proposal_types=proposal_types)
                    elif data == 'salomon':
                        if D == 10:
                            Fs = [1.5, 1.5, 1.5, 1.5] #D=10
                        elif D == 30:
                            Fs = [1., 1.5, 1.5, 1.5]  # D=30
                        elif D == 100:
                            Fs = [1., 1.5, 1.5, 2.0]  # D=100
                        plot_best(dataset_name=data, D=D, pop=pop, Fs=Fs, proposal_types=proposal_types)
                    elif data == 'schwefel':
                        if D == 10:
                            Fs = [1., 1.5, 1., 2.0] #D=10
                        elif D == 30:
                            Fs = [1., 1.5, 1.5, 2.0]  # D=30
                        elif D == 100:
                            Fs = [1., 1.5, 1.5, 2.0]  # D=100
                        plot_best(dataset_name=data, D=D, pop=pop, Fs=Fs, proposal_types=proposal_types)
                    else:
                        raise ValueError('Wrong data name!')