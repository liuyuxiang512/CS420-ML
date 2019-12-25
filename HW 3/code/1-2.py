from sklearn import svm
import numpy as np
import struct


def load(type='train'):
    path_y = 'data/DataB/%s-labels-idx1-ubyte' % type
    path_X = 'data/DataB/%s-images-idx3-ubyte' % type

    with open(path_y, 'rb') as file:
        magic, n = struct.unpack('>II', file.read(8))
        labels = np.fromfile(file, dtype=np.uint8)

    with open(path_X, 'rb') as file:
        magic, num, rows, cols = struct.unpack('>IIII', file.read(16))
        images = np.fromfile(file, dtype=np.uint8).reshape(len(labels), 28 * 28)

    return images, labels


X_train, y_train = load('train')
X_test, y_test = load('t10k')
clf = svm.SVC(gamma='scale')
clf.fit(X_train, y_train)
predict = clf.predict(X_test)

index = np.arange(0, 10000)
index = index[y_test == predict]

accuracy = np.size(index) / np.size(y_test)
accuracy = "%.4f%%" % float(accuracy * 100)

print(accuracy)
