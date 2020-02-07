import os
import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from urllib import request
from scipy.special import softmax

from skimage.transform import resize

from testbeds.testbed import TestBed

# ----------------------------------------------------------------------------------------------------------------------
PYTHONPATH = '/Users/jmt/Dev/github/life'  #mac

filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]


# ----------------------------------------------------------------------------------------------------------------------
class MNIST(TestBed):
    def __init__(self, name='mnist', image_size=(14, 14), batch_size=1000, train_size=5000):
        super().__init__()
        self.name = name
        self.image_size = image_size

        if not(os.path.isfile(PYTHONPATH + '/data/' + 'mnist.pkl')):
            self.download_mnist(location=PYTHONPATH + '/data/')
            self.save_mnist(location=PYTHONPATH +'/data/')

        x_train, y_train, x_test, y_test = self.load(location=PYTHONPATH + '/data/')

        x_train = x_train / 255.
        x_test = x_test / 255.

        # TRAIN DATA
        x_train = np.reshape(x_train[0:train_size], (train_size, 28, 28))
        self.x_train = np.zeros((x_train.shape[0], image_size[0], image_size[1]))
        for i in range(x_train.shape[0]):
            self.x_train[i] = resize(x_train[i], image_size, anti_aliasing=True)
        self.x_train = np.reshape(self.x_train, (train_size, image_size[0] * image_size[1]))
        self.y_train = y_train[0:train_size]

        # TEST DATA
        x_test = np.reshape(x_test, (x_test.shape[0], 28, 28))
        self.x_test = np.zeros((x_test.shape[0], image_size[0], image_size[1]))
        for i in range(x_test.shape[0]):
            self.x_test[i] = resize(x_test[i], image_size, anti_aliasing=True)
        self.x_test = np.reshape(self.x_test, (x_test.shape[0], image_size[0] * image_size[1]))
        self.y_test = y_test

        self.batch_size = batch_size

    @staticmethod
    def download_mnist(location):
        base_url = "http://yann.lecun.com/exdb/mnist/"
        for name in filename:
            print("Downloading " + name[1] + "...")
            request.urlretrieve(base_url + name[1], location + name[1])
        print("Download complete.")

    @staticmethod
    def save_mnist(location):
        mnist = {}
        for name in filename[:2]:
            with gzip.open(location + name[1], 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
        for name in filename[-2:]:
            with gzip.open(location + name[1], 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
        with open(location + "mnist.pkl", 'wb') as f:
            pickle.dump(mnist, f)
        print("Save complete.")

    @staticmethod
    def load(location):
        with open(location + "mnist.pkl", 'rb') as f:
            mnist = pickle.load(f)
        return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

    def visualize(self, mode='train', size_x=5, size_y=5, file_name='mnist_images'):
        fig = plt.figure(figsize=(size_x, size_y))
        gs = gridspec.GridSpec(size_x, size_y)
        gs.update(wspace=0.05, hspace=0.05)

        if mode == 'train':
            x_sample = self.x_train[0:size_x * size_y]
        elif mode == 'test':
            x_sample = self.x_test[0:size_x * size_y]
        else:
            raise ValueError('Mode could be train OR test, nothing else!')

        for i, sample in enumerate(x_sample):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            sample = np.expand_dims(sample, 0)
            sample = sample.reshape((1, self.image_size[0], self.image_size[1]))
            sample = sample.swapaxes(0, 2)
            sample = sample.swapaxes(0, 1)
            sample = sample[:, :, 0]
            plt.imshow(sample, cmap='gray')

        plt.savefig(file_name + '_' + mode + '.pdf', bbox_inches='tight')
        plt.close(fig)

    def objective(self, w, *args):
        hidden_units = args[0]['hidden_units']
        image_size = args[0]['image_size']
        im_shape = image_size[0] * image_size[1]

        if args[0]['evaluate'] is False:
            data_x = self.x_train
            data_y = self.y_train
        else:
            data_x = self.x_test
            data_y = self.y_test

        y_pred = np.zeros((data_y.shape[0],))

        for i in range(data_x.shape[0] // self.batch_size):
            w1 = w[0: im_shape * hidden_units]
            w2 = w[im_shape * hidden_units:]

            W1 = np.reshape(w1, (im_shape, hidden_units))
            W2 = np.reshape(w2, (hidden_units, 10))

            # First layer
            h = np.dot(data_x[i * self.batch_size: (i + 1) * self.batch_size], W1)
            # ReLU
            h = np.maximum(h, 0.)
            # Second layer
            logits = np.dot(h, W2)
            # Softmax
            prob = softmax(logits, -1)

            y_pred[i * self.batch_size : (i+1) * self.batch_size] = np.argmax(prob, -1)

        class_error = 1. - np.mean(data_y == y_pred)
        return class_error
