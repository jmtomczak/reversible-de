import pickle
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# # rc('text', usetex=True)

sp = '_'
save_dir_name = '../results/'


def load_from_file(file_to_load):
    f = open(file_to_load, "rb")
    final_results = pickle.load(f)
    f.close()
    return final_results


def plot_populations(dir_name, x_axis=3, y_axis=2, x_label='alpha', y_label='beta', epochs=None):

    x_populations = load_from_file(dir_name + '/' + 'populations.pkl')

    if epochs is None:
        epochs = range(-1, len(x_populations)-1 )

    for i in epochs:
        xi = x_populations[str(i)][:,x_axis]
        yi = x_populations[str(i)][:, y_axis]

        plt.scatter(x=xi, y=yi, label=str(i), alpha=0.5)

    plt.grid()

    if x_label is 'n':
        plt.xlabel('$' + x_label + '$')
    else:
        plt.xlabel('$\\' + x_label + '$')
    if y_label is 'n':
        plt.ylabel('$' + y_label + '$')
    else:
        plt.ylabel('$\\' + y_label + '$')

    plt.legend(loc=1)
    plt.savefig(dir_name + '/' + x_label + sp + y_label + sp + 'plot.pdf')
    plt.close()


if __name__ == '__main__':

    x_axes = [3, 0, 1, 0]
    y_axes = [2, 2, 2, 1]
    x_labels = ['alpha', 'alpha_0', 'n', 'alpha_0']
    y_labels = ['beta', 'beta','beta', 'n']

    for i in range(len(x_axes)):
        plot_populations(dir_name='../results/Repressilator_F_1.5_pop_100/revpoplife-differential_1-F-1.5-pop_size-100-epochs-20-r0',
                         x_axis=x_axes[i], y_axis=y_axes[i],
                         x_label=x_labels[i], y_label=y_labels[i],
                         epochs=[-1, 5, 10, 19])