import pickle
import numpy as np
from scipy.special import softmax
from scipy.stats import mode
from optimization.population_algorithm import ReversiblePopLiFe, LikelihoodFreeInference
from testbeds.mnist_testbed2 import MNIST


def load_from_file(file_to_load):
    f = open(file_to_load, "rb")
    final_results = pickle.load(f)
    f.close()
    return final_results


def single_nn(hidden_units, image_size, w, data_x, data_y, batch_size):
    im_shape = image_size[0] * image_size[1]

    y_pred = np.zeros((data_y.shape[0],))

    for i in range(data_x.shape[0] // batch_size):
        w1 = w[0: im_shape * hidden_units]
        w2 = w[im_shape * hidden_units:]

        W1 = np.reshape(w1, (im_shape, hidden_units))
        W2 = np.reshape(w2, (hidden_units, 10))

        # First layer
        h = np.dot(data_x[i * batch_size: (i + 1) * batch_size], W1)
        # ReLU
        h = np.maximum(h, 0.)
        # Second layer
        logits = np.dot(h, W2)
        # Softmax
        prob = softmax(logits, -1)

        y_pred[i * batch_size: (i + 1) * batch_size] = np.argmax(prob, -1)

    return y_pred


def bagging(x_sol, hidden_units, image_size, data_x, data_y, batch_size):

    for i in range(x_sol.shape[0]):
        w = x_sol[i,:]

        y_pred_i = single_nn(hidden_units, image_size, w, data_x, data_y, batch_size)
        y_pred_i = np.expand_dims(y_pred_i, axis=0)

        if i == 0:
            ens = y_pred_i
        else:
            ens = np.concatenate((ens, y_pred_i), 0)

    y_pred = np.zeros((data_y.shape[0],))
    for n in range(data_x.shape[0] // batch_size):
        m, _ = mode(ens[:, n * batch_size: (n + 1) * batch_size], 0)
        y_pred[n * batch_size: (n + 1) * batch_size] = m[0,:]

    class_error = 1. - np.mean(data_y == y_pred)
    return class_error


def ensemble(x_sol, f_sol, hidden_units, image_size, data_x, data_y, batch_size):

    for i in range(x_sol.shape[0]):
        w = x_sol[i,:]

        y_pred_i = single_nn(hidden_units, image_size, w, data_x, data_y, batch_size)
        y_pred_i = np.expand_dims(y_pred_i, axis=0)

        if i == 0:
            ens = y_pred_i
        else:
            ens = np.concatenate((ens, y_pred_i), 0)

    y_pred = np.zeros((data_y.shape[0],))
    for n in range(data_x.shape[0] // batch_size):
        m, _ = mode(ens[:, n * batch_size: (n + 1) * batch_size], 0)
        y_pred[n * batch_size: (n + 1) * batch_size] = m[0,:]

    class_error = 1. - np.mean(data_y == y_pred)
    return class_error


if __name__ == '__main__':

    F = 1.5
    de_proposal_types = ['differential_1', 'de_times_3', 'antisymmetric_differential', 'differential_3']

    # EXPERIMENT SETUP
    name = 'mnist_nn'
    image_size = (14, 14)
    hidden_units = 20

    D = image_size[0] * image_size[1] * hidden_units + hidden_units * 10

    bounds = [[-3.] * D, [3.] * D]

    pop_size = 500

    num_epochs = 500

    epsilon = np.infty

    # xavier init
    xavier = np.asarray([np.sqrt(2. / (image_size[0] * image_size[1] + 20.))] * image_size[0] * image_size[1] * 20 + [
        np.sqrt(2. / (20. + 10.))] * 20 * 10)
    x0 = np.random.randn(pop_size, D) * xavier

    params = {}

    params['evaluate_objective_type'] = 'single'
    params['evaluate'] = False

    params['image_size'] = image_size
    params['hidden_units'] = hidden_units
    params['pop_size'] = pop_size
    params['gaussian_prob'] = .0
    params['mixing_prob'] = 0.0
    params['num_cutting_points'] = 0
    params['CR'] = 0.9
    params['F'] = F
    params['randomized_F'] = False

    # Experiment
    b_fun = MNIST(name=name, image_size=image_size, train_size=1000)
    objective = b_fun.objective

    revpoplife = ReversiblePopLiFe(objective, args=(params, None), x0=x0, burn_in_phase=0, num_epochs=num_epochs,
                                   bounds=(bounds[0], bounds[1]),
                                   population_size=pop_size,
                                   elitism=0.,
                                   de_proposal_type=de_proposal_types[0])

    lf_poplife = LikelihoodFreeInference(pop_algorithm=revpoplife, num_epochs=num_epochs)

    final_results = {}

    for de_proposal_type in de_proposal_types:
        score = []
        for rep in range(3):
            print(de_proposal_type, rep)
            # FOLDER NAME
            results_dir = '../results/' + name + '_D' + str(D) + '_F_' + str(F) + '_pop_' + str(pop_size)

            specific_folder = '-' + de_proposal_type + '-F-' + str(params['F']) + '-pop_size-' + str(
                params['pop_size']) + '-epochs-' + str(num_epochs)
            directory_name = results_dir + '/' + lf_poplife.name + specific_folder + '-r' + str(rep)

            x_sol = np.load(directory_name + '/' + 'solutions.npy')

            # EVALUATE ON TRAINSET
            revpoplife.params['evaluate'] = False
            f_sol = revpoplife.evaluate_objective(x_sol)

            revpoplife.params['evaluate'] = True

            x_sol = x_sol[x_sol.shape[0] - 5:x_sol.shape[0],:]
            f_sol = f_sol[x_sol.shape[0] - 5:x_sol.shape[0]]

            ens_err = bagging(x_sol, hidden_units, image_size, b_fun.x_test, b_fun.y_test, batch_size=1000)
            # ens_err = ensemble(x_sol, f_sol, hidden_units, image_size, b_fun.x_test, b_fun.y_test, batch_size=1000)

            score.append(ens_err)

        score = np.asarray(score)
        # saving best results to the file
        f = open(results_dir + '/' + 'bagging_best_test_results.txt', "a")
        f.writelines(
            de_proposal_type + ': ' + str(np.mean(score)) + ' (' + str(np.std(score)) + ')' + '\n')
        f.close()