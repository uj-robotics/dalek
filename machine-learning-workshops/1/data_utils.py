import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros,reshape
import csv
import numpy as np

def get_exam_dataset():
    X = []
    Y = []
    with open("./data/exam_data.csv","r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            X.append([float(row[0]), float(row[1])])
            Y.append([float(row[2])])

    X,Y = np.array(X), np.array(Y)
    # Musimy usrednic dane i przeskalowac
    return X,Y

def get_MNIST_dataset(digits, dataset = "training", path = "./data"):
    """
    Loads MNIST files into 3D numpy arrays

    @param digits list of numbers (0 to 9)
    @param dataset "training" or "testing"
    @returns 2 numpy arrays : images, labels

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in xrange(size) if lbl[k] in digits ]
    N = len(ind)

    #images = zeros((N, rows, cols), dtype=uint8)
    #labels = zeros((N, 1), dtype=int8)
    #
    images = zeros((N, rows, cols), dtype="float64")
    labels = zeros((N, 1), dtype="float64")
    for i in xrange(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels