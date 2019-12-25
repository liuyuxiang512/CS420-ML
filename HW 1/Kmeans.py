import matplotlib.pyplot as plt
from data import Dataset
import numpy as np


class Kmeans(object):
    def __init__(self, cluster_num=6):
        self.cluster_num = cluster_num
        dataset = Dataset()
        dataset.generate()
        self.data = dataset.data
        self.centers = np.zeros((self.cluster_num, 2))
        self.r = np.zeros((np.shape(self.data)[0], self.cluster_num))

    def classify(self):
        # Initialization
        for i in range(self.cluster_num):
            mu_x = np.random.normal(2, 2)
            mu_y = np.random.normal(2, 2)
            self.centers[i][0] = mu_x
            self.centers[i][1] = mu_y

        # Iteration
        change = True
        while change:
            # E step
            tmp = self.r.copy()
            self.r = np.zeros(np.shape(self.r))
            for n in range(np.shape(self.data)[0]):
                distance = np.sqrt(np.sum(np.square(np.tile(self.data[n], (self.cluster_num, 1)) - self.centers), axis=1))
                index_min = distance.argmin()
                self.r[n][index_min] = 1

            if (self.r == tmp).all():
                change = False

            # M step
            for k in range(self.cluster_num):
                if np.sum(self.r[:, k]) != 0:
                    self.centers[k] = np.sum(self.data * np.tile(self.r[:, k], (2, 1)).T, axis=0).T / np.sum(self.r[:, k])

    def show(self):
        for k in range(self.cluster_num):
            x = []
            y = []
            for n in range(np.shape(self.data)[0]):
                if self.r[n][k] == 1:
                    x.append(self.data[n][0])
                    y.append(self.data[n][1])

            plt.scatter(x, y)

        plt.scatter(self.centers[:, 0], self.centers[:, 1])
        plt.title('Clusters of Data Points')
        plt.show()


if __name__ == '__main__':
    kmeans = Kmeans()
    kmeans.show()
    kmeans.classify()
    kmeans.show()

