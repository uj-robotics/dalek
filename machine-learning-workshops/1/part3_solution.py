""" Implement digit recognition """

from part2 import LogisticRegression, normalize_data, evaluate
import numpy as np
from part1 import get_MNIST_testing_normalized, get_MNIST_training_normalized, show_as_image
from data_utils import get_exam_dataset

#2. Rozszerzymy na rozpoznawanie wielu klas (klas cyfr)
class MulticlassLogisitcRegression():

    def __init__(self, classes = 0):
        self.classifiers = []
        self.classes = classes

        pass

    def fit(self, X, Y):
        # Uzywamy 1 vs all, i wybieramy najbardziej pewna klasyfikacje

        #1. Stworz 10 klasyfikatorow LogisticRegression
        for i in xrange(self.classes): self.classifiers.append(LogisticRegression(cls=i))

        #2. Trenuj je uzywajac X i Y
        for i in xrange( self.classes):
            print "Fitting {0}-th classifier".format(i)
            self.classifiers[i].fit_optimized(X, Y)


    def predict(self, x):
        #3. Zwroc najbardziej prawdopodobny wynik
        x.shape = (x.shape[0], )
        return np.array([cls.predict_probability(x) for cls in self.classifiers]).argmax()


def test_part3_1():
    """ Test from part2 should be working """
    X,Y = get_exam_dataset()
    X,Y = normalize_data(X,Y)
    log_reg = MulticlassLogisitcRegression(classes = 2)
    log_reg.fit(X,Y)
    accuracy =  evaluate(log_reg, X, Y)
    print "Accuracy ",accuracy
    assert(accuracy > 0.8)


def test_part3_2():
    X, Y = get_MNIST_training_normalized()
    print "Normalized MNIST dataset loaded"
    ml_log_reg = MulticlassLogisitcRegression(10)
    ml_log_reg.fit(X,Y)
    print "Fitted logistic regression"
    X_test, Y_test = get_MNIST_testing_normalized()
    print "Normalized MNIST testing dataset loaded"
    accuracy = evaluate(ml_log_reg, X_test, Y_test)
    print "Accuracy on test dataset = ", accuracy
    assert(accuracy > 0.8)
    return ml_log_reg


def show_results(trained_model ,X_test, Y_test):
    """ Plot some results """
    ml_log_reg = cPickle.load(open("trained_model.pkl","r"))

import cPickle
if __name__ == "__main__":
    #test_part3_1()
    cPickle.dump(test_part3_2(), open("trained_model.pkl","w"))
