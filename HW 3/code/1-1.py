from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_svmlight_file
from libsvm.python.commonutil import *
from libsvm.python.svmutil import *
import numpy as np
import csv


def load_SVM(number=1):
    path_train = 'data/DataA/a%da' % number
    path_test = 'data/DataA/a%da.t' % number
    return path_train, path_test


# SVM
def SVM(path_train, path_test, svm_type=0, kernel_type=0, degree=3, gamma=1/123, cost=1):
    y_train, X_train = svm_read_problem(path_train)
    y_test, X_test = svm_read_problem(path_test)
    param1 = '-s %d -t %d -d %d -g %d -c %f -h 0' % (svm_type, kernel_type, degree, gamma, cost)
    model = svm_train(y_train, X_train, param1)
    p_label, p_acc, p_val = svm_predict(y_test, X_test, model)
    accuracy = "%.4f%%" % p_acc[0]
    return accuracy


# MLP
def MLP(path_train, path_test):
    X_train, y_train = load_svmlight_file(path_train)
    X_test, y_test = load_svmlight_file(path_test)
    X_train = X_train.todense()
    X_train = np.c_[X_train, np.zeros((len(y_train), 123 - np.size(X_train[0, :])))]
    X_test = X_test.todense()
    X_test = np.c_[X_test, np.zeros((len(y_test), 123 - np.size(X_test[0, :])))]
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    accuracy = 1 - np.sum(np.abs(predict - y_test)) / np.size(predict)
    accuracy = "%.4f%%" % float(accuracy * 100)
    return accuracy


def SVMMLP():
    accuracy = []
    for number in [1, 2, 3, 4, 5, 6, 7, 9]:
        path_train, path_test = load_SVM(number=number)
        accuracy_SVM = SVM(path_train=path_train, path_test=path_test)
        accuracy_MLP = MLP(path_train=path_train, path_test=path_test)
        print(str(number) + '\t' + accuracy_SVM + '\t' + accuracy_MLP)
        accuracy.append([accuracy_SVM, accuracy_MLP])

    print(accuracy)

    with open('result/1-1-SVM-MLP.csv', 'w', newline='') as file:
        write = csv.writer(file)
        for line in accuracy:
            write.writerow(line)


def kernel_SVM():
    accuracy = []
    for number in [1, 2, 3, 4, 5, 6, 7, 9]:
        path_train, path_test = load_SVM(number=number)
        current_accuracy = []
        for kernel_type in range(4):
            accuracy_SVM = SVM(path_train=path_train, path_test=path_test, kernel_type=kernel_type)
            current_accuracy.append(accuracy_SVM)
        accuracy.append(current_accuracy)

    print(accuracy)

    with open('result/1-1-SVM-kernelType.csv', 'w', newline='') as file:
        write = csv.writer(file)
        for line in accuracy:
            write.writerow(line)


def cost_SVM():
    accuracy = []
    for number in [1, 2, 3, 4, 5, 6, 7, 9]:
        path_train, path_test = load_SVM(number=number)
        current_accuracy = []
        for cost in [0.001, 0.01, 0.1, 1, 10]:
            accuracy_SVM = SVM(path_train=path_train, path_test=path_test, cost=cost)
            current_accuracy.append(accuracy_SVM)
        accuracy.append(current_accuracy)

    print(accuracy)

    with open('result/1-1-SVM-cost.csv', 'w', newline='') as file:
        write = csv.writer(file)
        for line in accuracy:
            write.writerow(line)


# SVMMLP()
# kernel_SVM()
# cost_SVM()
