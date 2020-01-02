import numpy as np
import torch
from torchvision import datasets, transforms
from scipy.special import softmax

from skimage.transform import resize

from testbeds.testbed import TestBed


class MNIST(TestBed):
    def __init__(self, name='mnist', image_size=(14, 14), batch_size=1000):
        super().__init__()
        self.name = name

        test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False,
                                                                 transform=transforms.Compose([transforms.ToTensor()])),
                                                  batch_size=batch_size, shuffle=False)

        x_test = test_loader.dataset.data.float().numpy() / 255.
        # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
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

        y_pred = np.zeros((self.y_test.shape[0],))

        for i in range(self.x_test.shape[0] // self.batch_size):
            w1 = w[0: im_shape * hidden_units]
            w2 = w[im_shape * hidden_units:]

            W1 = np.reshape(w1, (im_shape, hidden_units))
            W2 = np.reshape(w2, (hidden_units, 10))

            # h = np.einsum('nd,dm->nm', self.x_test[i * 1000 : (i+1) * 1000], W1)
            h = np.dot(self.x_test[i * self.batch_size: (i + 1) * self.batch_size], W1)
            h = np.maximum(h, 0.)
            # logits = np.einsum('nm,ml->nl', h, W2)
            logits = np.dot(h, W2)

            prob = softmax(logits, -1)

            y_pred[i * self.batch_size : (i+1) * self.batch_size] = np.argmax(prob, -1)

        class_error = 1. - np.mean(self.y_test == y_pred)
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