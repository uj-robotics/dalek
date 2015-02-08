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
        h = LogisticRegression.sigmoid(X.dot(W[1:]) + W[0]).astype(np.float64)
        h.shape = (h.shape[0], 1)
        M = X.shape[0]
        features = X.shape[1]


        L = (1.0/M) * (  (-Y.T.dot(np.log(h))) - ((1.0 - Y.T)).dot(np.log(1.0 - h))) #Todo <- add regularization
        print L[0,0]

        return np.float64(L[0,0])

    @staticmethod
    def compute_grad(W, X, Y, cls):
        ## Niezbyt szybka wersja
        #M = X.shape[0]
        #h = np.zeros((M,1), dtype="float64")
        #for i in xrange(M):
        #    h[i] = (LogisticRegression.sigmoid(X[i,:].dot(W[1:]) + W[0]) - Y[i])
        #h.shape = (h.shape[0],1)
        #features = X.shape[1]
        #grad = np.zeros((features + 1, 1), dtype='float64')
        #grad[0] = (1.0/M)*h.sum()
        #sumd = np.zeros((features,), dtype="float64")
        #for i in xrange(features):
        #    sumd[i] = X[:, i].T.dot(h).astype(np.float64)
        #sumd.shape = (features, )
        #sumd.shape = (features, 1)
        #grad[1:] = (1.0/M)*sumd
        #grad.shape = (features+1,)


        # Da sie szybciej ale trzeba uzywac -= += operatorow a nie pelnych operacji, poniewaz numpy nie wie
        # czy moze nie kopiowac
        M, features = X.shape[0], X.shape[1]
        h = LogisticRegression.sigmoid(X.dot(W[1:]) + W[0]).astype(np.float64)
        h.shape = (h.shape[0],1 )
        h -= Y
        grad = np.zeros((features+1,1 ), dtype="float64")
        grad[1:] = (1.0/M)*X.T.dot(h).astype("float64")
        h.shape = (h.shape[0],)
        grad[0] = (1.0/M)*sum(h)
        grad.shape = (grad.shape[0], )

        #print grad[200:220]
        return grad

    @timed
    def fit_optimized(self, X, Y):
        # Required decoration
        Y = np.array(Y) # copy array
        Y = (Y == self.cls).astype(np.float64)

        def cost(W):
            return LogisticRegression.compute_cost(W, X, Y, self.cls)
        def grad(W):
            return LogisticRegression.compute_grad(W, X, Y, self.cls)

        features = X.shape[1]
        self.W = (np.random.random_sample((features+1,))-0.5)*0.001
        #self.W = np.zeros((features+1,), dtype='float64')

        self.W= fmin_bfgs(cost, self.W, grad, disp=True, maxiter=300)
        self.W.shape = (self.W.shape[0], 1)
    @timed
    def fit(self, X, Y):
        """
            @param X - numpy 2d matrix
            @param Y - numpy 2d matrix of labels as column vector
        """
        #1. Inicjalizacja wag
        self.features = X.shape[1]
        M = X.shape[0]
        self.W = np.zeros((self.features+1,1), dtype=float)

        #2. Zmiana Y na 0 v 1, w zaleznosci czy klasa zgadza sie z klasa klasyfikatora



        print Y[1:10]
        print "Fitting logistic regression to ",M, " examples with ", self.features

        # 3. Implementacja - wersja szybka
        iteration = 0
        #while True:
        #    iteration += 1
        #
        #    if iteration % 1 == 0: print "Iteration ",iteration
        #
        #    grad = np.zeros((self.features+1, 1))
        #    h = (Y - self.sigmoid(X.dot(self.W[1:]) + self.W[0]))
        #    grad[1:] = X.T.dot(h)
        #    grad[0] = h.sum()
        #    self.W = self.W + self.alpha*(1./M)*grad
        #
        #    if math.fabs(grad.max()) < self.stopping_criterion : break
        #    if self.max_it <= iteration: break

        # 3. Implementacja - wersja szybka i miesci sie w pamieci
        iteration = 0
        batch_size = 600
        while True:
            iteration += 1



            grad = np.zeros((self.features+1, 1))
            for i in xrange(M/batch_size): #TODO append last
                X_batch = X[i*batch_size:((i+1)*batch_size), :]
                Y_batch = Y[i*batch_size:((i+1)*batch_size), :]
                h = (LogisticRegression.sigmoid(X_batch.dot(self.W[1:]) + self.W[0]) - Y_batch)
                grad[1:] = X_batch.T.dot(h)
                grad[0] = h.sum()
                self.W = self.W - self.alpha*(1./batch_size)*grad


            if iteration % 100 == 0:
                print "Iteration ",iteration
                print math.fabs(grad.max())
                print self.W.max()

            if math.fabs(grad.max()) < self.stopping_criterion : break
            if self.max_it <= iteration: break

        # 3. Implementacja - wersja wolna
        #while True:
        #    iteration += 1
        #    if iteration % 10 == 0: print "Iteration ",iteration
        #    grad = np.zeros((self.features+1, 1))
        #    for id, example in enumerate(X):
        #        h = Y[id, 0] - self.sigmoid(np.dot(X[id,:], self.W[1:]))
        #        grad[1:, 0] = grad[1:, 0] + (Y[id, 0] - self.sigmoid(np.dot(X[id, :], self.W[1:, 0]) + self.W[0,0]))*X[id, :]
        #        grad[0, 0] = grad[0, 0] + (Y[id, 0] - self.sigmoid(np.dot(X[id, :], self.W[1:, 0]) + self.W[0,0]))
        #    self.W = self.W + self.alpha*(1.0/X.shape[0])*grad
        #    if math.fabs(grad.max()) < self.stopping_criterion : break
        #    if self.max_it <= iteration: break

        print "Stopped training after ",iteration


    def predict(self, x):
        """
            @param x - numpy vector
            @returns predicted label
        """
        return 1 if (self.sigmoid(np.dot(x, self.W[1:,0])+ self.W[0,0]) > 0.5) else 0

    def predict_probability(self, x):
        """
            @param x - numpy vector
            @returns predicted probability of label
        """
        return self.sigmoid(np.dot(x, self.W[1:,0])+ self.W[0,0])

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

import random
def train_and_evaluate(model, X, Y):
    """
    Splits data into training and testing datasets
    and then evaluates model
        @returns accuracy of the model on test set
    """
    datapoints = range(X.shape[0])



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
    test_part2()