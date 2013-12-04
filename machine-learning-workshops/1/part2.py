from utils import *
from part1 import get_MNIST_testing_normalized, get_MNIST_training_normalized, show_as_image, normalize_data
from data_utils import get_exam_dataset
import math

import numpy as np
import pylab as pl

from scipy.optimize import fmin_bfgs

# Potrzebujemy lepszej optymalizacji!
from scipy.optimize import fmin_bfgs

#1. Zaimplementujemy regresje logistyczna
class LogisticRegression():
    """ Basic logistic regression """

    def __init__(self, cls=1):
        """
            @param cls - Class that classifier will treat as positive
        """
        self.W = None # Learned weight
        self.features = 0 # Number of features
        self.cls = cls
        self.alpha = 0.13
        self.stopping_criterion = 0.00001
        self.max_it = 1000

    @staticmethod
    def sigmoid(x):
        return 1.0/(1.0 + np.exp(-x))

    @staticmethod
    def compute_cost(W, X, Y, cls):
        L = None
        # ..
        return np.float64(L[0,0])

    @staticmethod
    def compute_grad(W, X, Y, cls):
        grad = None
        # ..
        return grad

    @timed
    def fit_optimized(self, X, Y):
        # ..
        pass

    @timed
    def fit(self, X, Y):
        """
            @param X - numpy 2d matrix
            @param Y - numpy 2d matrix of labels as column vector
        """
        #1. Inicjalizacja wag

        # To sie przyda
        self.features = X.shape[1]
        M = X.shape[0]

        # A to sa wagi regresji logistcznej
        self.W = np.zeros((self.features+1,1), dtype=float)

        #2. Zmiana Y na 0 v 1, w zaleznosci czy klasa zgadza sie z klasa klasyfikatora
        Y = np.array(Y) # copy array
        Y = (Y == self.cls).astype(np.float64)

        # 3. Implementacja
        #TODO: fill in
        iteration = 0
        batch_size = 600
        while True:
            iteration += 1
            break

        # Tu maja byc spoko wagi :)

        print "Stopped training after ",iteration


    def predict(self, x):
        """
            @param x - numpy vector
            @returns predicted label
        """
        # 4. TODO: fill in
        return 0

    def predict_probability(self, x):
        """
            @param x - numpy vector
            @returns predicted probability of label
        """
        # 5. TODO: fillin
        return 0

def plot_exam_data_decision_boundary(X, Y, model , cls = 1):
    """ Plot exam data
        @param X - Nx2 iterable (numpy matrix)
        @param Y - N iterable (numpy vector)
    """

    if Y.ndim > 1: Y = Y.reshape([Y.shape[0]])

    # Plot boundary
    u = np.linspace(X.min(), X.max(), 50)
    v = np.linspace(X.min(), X.max(), 50)
    z = np.zeros(shape=(len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i, j] = model.predict([u[i], v[j]])

    pl.figure(1)
    pl.contour(u,v,z)
    pl.plot(X[Y==cls,0], X[Y==cls,1], 'o', mfc='none')
    pl.plot(X[Y!=cls,0], X[Y!=cls,1], 'x', mfc='none')
    pl.show()

def plot_exam_data(X, Y, cls):
    """ Plot exam data
        @param X - Nx2 iterable (numpy matrix)
        @param Y - N iterable (numpy vector)
    """
    pl.figure(1)
    pl.plot(X[Y==cls,0], X[Y==cls,1], 'o', mfc='none')
    pl.plot(X[Y!=cls,0], X[Y!=cls,1], 'x', mfc='none')
    pl.show()

def evaluate(model, X_test, Y_test):
    """
        @returns accuracy of the model on test set
    """
    predictions = []
    for example in X_test: predictions.append(model.predict(example))
    return sum([1 for i in xrange(len(X_test)) if predictions[i] == Y_test[i]]) / float(len(predictions))


def test_part2():
    """ Should be working after implementing succesfully part 2 """
    X,Y = get_exam_dataset()
    X,Y = normalize_data(X,Y)
    log_reg = LogisticRegression(cls=1)
    log_reg.fit(X,Y)
    accuracy =  evaluate(log_reg, X, Y)
    plot_exam_data_decision_boundary(X,Y,log_reg, 1)
    assert(accuracy > 0.8)

if __name__ == "__main__":
    # 0. Odkomentuj
    # plot_exam_data()
    # 6.test_part2()
    pass