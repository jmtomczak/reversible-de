import numpy as np
import torch
from torchvision import datasets, transforms
from scipy.special import softmax

from skimage.transform import resize

from testbeds.testbed import TestBed


class MNIST(TestBed):
    def __init__(self, name='mnist', image_size=(14, 14), batch_size=1000, train_size=5000):
        super().__init__()
        self.name = name

        train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True,
                                                                 transform=transforms.Compose([transforms.ToTensor()])),
                                                  batch_size=batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False,
                                                                 transform=transforms.Compose([transforms.ToTensor()])),
                                                  batch_size=batch_size, shuffle=False)

        # TRAIN DATA
        x_train = train_loader.dataset.data.float().numpy() / 255.
        np.random.shuffle(x_train)
        x_train = x_train[0:train_size]
        self.x_train = np.zeros((x_train.shape[0], image_size[0], image_size[1]))
        for i in range(x_train.shape[0]):
            self.x_train[i] = resize(x_train[i], image_size, anti_aliasing=True)
        self.x_train = np.reshape(self.x_train, (x_train.shape[0], image_size[0] * image_size[1]))
        self.y_train = np.array(train_loader.dataset.targets.float().numpy(), dtype=int)

        # TEST DATA
        x_test = test_loader.dataset.data.float().numpy() / 255.
        self.x_test = np.zeros((x_test.shape[0], image_size[0], image_size[1]))
        for i in range(x_test.shape[0]):
            self.x_test[i] = resize(x_test[i], image_size, anti_aliasing=True)
        self.x_test = np.reshape(self.x_test, (x_test.shape[0], image_size[0] * image_size[1]))

        self.y_test = np.array(test_loader.dataset.targets.float().numpy(), dtype=int)

        self.batch_size = batch_size

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
    print(mnist.x_test.shape)

    w = np.random.randn(14*14 * 300 + 300 * 10)

    tic = time.time()
    err = mnist.objective(w=w)
    toc = time.time()

    print(err, toc-tic)