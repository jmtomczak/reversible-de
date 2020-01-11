import os
import gzip
import pickle
import numpy as np
from urllib import request
from scipy.special import softmax

from skimage.transform import resize

from testbeds.testbed import TestBed

# PYTHONPATH = '/home/jakub/Dev/github/life'  # ripper5
PYTHONPATH = '/Users/jmt/Dev/github/life'  #mac

filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]

class MNIST(TestBed):
    def __init__(self, name='mnist', image_size=(14, 14), batch_size=1000, train_size=5000):
        super().__init__()
        self.name = name

        if not(os.path.isfile(PYTHONPATH + '/data/' + 'mnist.pkl')):
            self.download_mnist(location=PYTHONPATH + '/data/')
            self.save_mnist(location=PYTHONPATH +'/data/')

        x_train, y_train, x_test, y_test = self.load(location=PYTHONPATH + '/data/')

        x_train = x_train / 255.
        x_test = x_test / 255.

        # TRAIN DATA
        x_train = x_train[0:train_size]
        self.x_train = np.zeros((x_train.shape[0], image_size[0], image_size[1]))
        for i in range(x_train.shape[0]):
            self.x_train[i] = resize(x_train[i], image_size, anti_aliasing=True)
        self.x_train = np.reshape(self.x_train, (x_train.shape[0], image_size[0] * image_size[1]))
        self.y_train = y_train[0:train_size]

        # TEST DATA
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


if __name__ == '__main__':
    import time

    mnist = MNIST()

    print(mnist.x_train.shape)
    print(mnist.y_train.shape)

    print(mnist.x_test.shape)
    print(mnist.y_test.shape)