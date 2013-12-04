"""
Part 1:
    Convert MNIST to generic dataset that we will use in part2
"""

import itertools
import pylab as pl # Matlab w pythonie!
import numpy as np
import os
# Jest juz zaprogramowane pobieranie cyfr z danych binarnych MNIST
from data_utils import get_MNIST_dataset
from utils import *
import random
import cPickle

# 1. Podstawy numpy
def numpy_basics():
    """
        Przyklady uzycia numpy
        Numpy pozwala na szybkie operacje macierzowe w pythonie. Numpy uzywa pakietu BLAS (fortran), oraz
        w wiekszosci jest napisana w C++.
    """
    # a) tworzenie macierzy
    X = np.zeros([3,3]) # macierz 0
    print np.array([[1.1]]) #tworzenie macierzy z listy python
    Y = np.eye(3,3) # macierz jednostkowa
    X[0,0] = 10.0 # ustawienie elementu
    print "Array dimensions ",X.shape #wymiar macierzy

    # b) dodawanie macierzowe
    print (X+Y)

    # c) mnozenie macierzowe
    print np.dot(X,Y)

    # d) pobieranie elementow
    print X[1,1] #element X_22
    print X[1,:] #caly drugi wiersz, zwraca tez np.array

    # e) w kazdym wierszu najwiekszy element macierzy X+Y
    #TODO: fill in
    print (X+Y).max(axis=1) # "zjadamy" 2 wymiar, czyli kolumny

#2. Mozemy tez rysowac, uzywajac Matplotlib (pylab zapewnia fajny interfejs)
def plotting_examples():
    # a) Definiujemy funkcje ktora bedziemy rysowac, nie zeby to mialo znaczenie
    def f(t):
        return np.exp(-t) * np.cos(2*np.pi*t)

    # b) Np.range tworzy wektor z elementami w rownych odstepach
    t1 = np.arange(0.0, 5.0, 0.1)
    t2 = np.arange(0.0, 5.0, 0.02)

    # c) Odrazu narysujemy sobie 2 naraz
    pl.figure(1)

    # d) Przelaczamy sie do 1 wykresu
    pl.subplot(211) #rows =2, columns = 1, index = 1

    # e) Matlabowe rysowanie.
    pl.plot(t1, f(t1), 'bo')
    pl.plot(t2, f(t2), 'k') # ps. mozna tez pl.plot(t1,f(t1),'bo', t2, f(t2), 'k')

    # f) Przelaczamy sie do 2 wykresu
    pl.subplot(212) #rows =2, columns =1, index = 2

    print "Rysujemy ", t2
    print ", z ", np.cos(2*np.pi*t2)

    pl.plot(t2, np.cos(2*np.pi*t2), 'r--')

    # g) Renderujemy
    pl.show()

#3. Zbior danych MNIST
def MNIST_data():
    """
        Pobieramy wszystkie obrazki 28x28 (indeksowane pierwszym wymiarem) oraz etykiety
        Images jest 3-wymiarowa macierzy numpy.

        Wiecej do samodzielnego wykonania bo jestesmy juz swietni w numpy i rysowaniu wykresow
    """

    # Pobieramy macierze numpy z cyframi
    # images[i,j,k] <=> piksel (j,k) z i-tego obrazka w zbiorze danych
    images, labels = get_MNIST_dataset(range(10), "training") #pierwszy argument to

    # a) Ilosc przykladow i rozmiary danych
    print "Raw training data dimensions ", images.shape
    print "Labels dimensions ",labels.shape

    # b) Ile jest cyfr 2?
    print "Counting 2 in training dataset ",len(filter(lambda x: x == 2, labels))

    # c) Jaki jest sredni obrazek 2 ? (Usrednienie wszystkich macierzy ktore sa 2)

    #1. Pobierzmy wszystkie dwojki, fajny sposob indeksowania
    print labels == 2
    only_2 = images[labels == 2, :, :]
    print "Checking number of 2s ", only_2.shape

    #2. TODO: Usrednienie (matrix.mean moze byc przydatne)

    #3. TODO: narysowanie usrednionej cyfry (zobacz pl.imshow)

    # d) Ostatnie - przetworzmy ostatnia cyfre do 1 wymiarowego wektora
    vectorized = np.reshape(images[-1], newshape=(images[-1].shape[0]*images[-1].shape[1]))
    print "Vectorized last digit ", vectorized


#4. Tworzymy zbior ktory bedziemy uzywac z nastepnej czesci
def get_dataset(dataset_type = "training"):
    """
        We will use only this one to get data in part2, which will be
        a generic logistic regression model

        @param dataset "training" or "test"
        @returns standard X,Y or X
    """
    X, Y = None, None
    #1. Pobieramy obrazki i etykiety
    images, labels = get_MNIST_dataset(range(10), dataset_type)
    #2. Konwersja do X,Y (czyli wiersz to jeden przyklad, X - dwuwymiarowa macierz), Y po prostu przepisujemy
    # hint : reshape, np.zeros, przez macierz mozna latwo iterowac w petli (for row in matrix)
    #TODO: fill in
    Y = labels
    X = images.reshape((images.shape[0], images.shape[1]*images.shape[2]))

    #for id, digit in enumerate(images):
    #    pl.imshow(digit, cmap=pl.cm.gray)
    #    X[id, :] = digit.reshape((digit.shape[0]*digit.shape[1],))
    #    if id % 1000 == 0: print "Converted ",id

    # Czy da sie szybciej?

    #3. Return X,Y
    return X, Y



def normalize_data(X, Y):
    return ((X-X.mean())/X.std()), Y

"""
5. Czesto przydaje sie zapisywac posrednie rezultaty.
Uzyjemy do tego cPickle
Warto zauwazyc, ze mozna w pythonie za pomoca dekoratorow latwo modyfikowac funkcje tak aby cachowaly rezultaty,
bez potrzeby tworzenia osobnej wersji cached dla kazdej funkcji

Jak zostanie czas pokaze jak to zrobic generycznie (dla chetnych, kod w utils.py : dekorator @cached)
"""
MNIST_training = None
MNIST_testing = None
def get_MNIST_training_normalized():
    """
        get_dataset("testing"), cached for performance
        @returns X,Y
    """
    X, Y = get_dataset("training")
    X=X/255.0
    X-=0.5
    return X,Y

    global MNIST_training
    if MNIST_training is None:
        if os.path.exists("./data/MNIST_training.pkl"):
            MNIST_training = cPickle.load(open("./data/MNIST_training.pkl", "r"))
        else:
            MNIST_training = get_dataset("training")
            MNIST_training = normalize_data(MNIST_training[0], MNIST_training[1])
            cPickle.dump(MNIST_training, open("./data/MNIST_training.pkl","w"))
    return MNIST_training

def get_MNIST_testing_normalized():
    """
        get_dataset("training"), cached for performance
        @returns X,Y
    """
    X, Y = get_dataset("testing")
    X/=255.0
    X-=0.5
    return X,Y

    global MNIST_testing
    if MNIST_testing is None:
        if os.path.exists("./data/MNIST_testing.pkl"):
            MNIST_testing = cPickle.load(open("./data/MNIST_testing.pkl", "r"))
        else:
            MNIST_testing = get_dataset("testing")
            MNIST_testing = normalize_data(MNIST_testing[0], MNIST_testing[1])
            cPickle.dump(MNIST_training, open("./data/MNIST_testing.pkl","w"))
    return MNIST_testing


@timed
def test_dataset():
    """ Should be working after implementing get_dataset() """
    X,Y = get_MNIST_training_normalized()
    digits_test_truth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 632, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 166, 0, 0, 0, 0, 0]
    digits_test = []
    for example in itertools.islice(X,30):
        digits_test.append(sum(example[1:100]))
        assert(example.shape == (28*28,))

    show_as_image(X[0,:], 28, 28)
    print digits_test
    print digits_test_truth
    assert(digits_test_truth == digits_test)
    assert(X.shape == (60000, 28*28))
    assert(Y.shape == (60000,))
    return "Dziala :)"



def show_as_image(X, rows=28, columns=28):
    pl.imshow(X.reshape((rows,columns)), cmap = pl.cm.gray)
    pl.show()


if __name__ == "__main__":
    #Kolejnosc dzialan:

    #1.
    #2.
    #3.
    #4.

    #test wszystkiego:
    print test_dataset()
    print test_dataset()
