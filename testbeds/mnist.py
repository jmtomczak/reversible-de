import os
import gzip
import pickle
import numpy as np
from urllib import request

PYTHONPATH = '/home/jakub/Dev/github/life'  # ripper5
# PYTHONPATH = '/Users/jmt/Dev/github/life'  #mac

filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]


def download_mnist(location):
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print(name)
        print("Downloading " + name[1] + "...")
        request.urlretrieve(base_url + name[1], location + name[1])
    print("Download complete.")


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


if __name__ == '__main__':
    download_mnist(location=PYTHONPATH + '/data/')
    save_mnist(location=PYTHONPATH + '/data/')