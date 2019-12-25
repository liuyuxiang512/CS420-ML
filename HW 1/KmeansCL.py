import matplotlib.pyplot as plt
from data import Dataset
import numpy as np


class KmeansCL(object):
    def __init__(self, cluster_num=6):
        self.cluster_num = cluster_num
        self.rate = 0.01
        dataset = Dataset()
        dataset.generate()
        self.data = dataset.data
        self.centers = np.zeros((self.cluster_num, 2))
        self.r = np.zeros((np.shape(self.data)[0], self.cluster_num))

    def classify(self):
        # Initialization
        for i in range(self.cluster_num):
            mu_x = np.random.normal(2, 1)
            mu_y = np.random.normal(2, 1)
            self.centers[i][0] = mu_x
            self.centers[i][1] = mu_y

        self.show(init=1)

        # Iteration
        index_min = np.zeros((np.shape(self.data)[0], 1), dtype=np.int64)
        index_rival = np.zeros((np.shape(self.data)[0], 1), dtype=np.int64)
        for i in range(500):
            self.r = np.zeros(np.shape(self.r))
            for n in range(np.shape(self.data)[0]):
                distance = np.sqrt(np.sum(np.square(np.tile(self.data[n], (self.cluster_num, 1)) - self.centers), axis=1))
                index_min[n] = distance.argmin()
                distance[index_min[n]] += 100
                index_rival[n] = distance.argmin()
                self.r[n][index_min[n]] = 1
                self.r[n][index_rival[n]] = - 0.1

                self.centers[index_min[n]] += self.rate * self.r[n][index_min[n]] * (self.data[n].T - self.centers[index_min[n]])
                self.centers[index_rival[n]] += self.rate * self.r[n][index_rival[n]] * (self.data[n].T - self.centers[index_rival[n]])

    def show(self, init=0, note=None):
        plt.xlim((-2.5, 6.5))
        plt.ylim((-2.5, 6.5))
        x = []
        y = []
        if init:
            note = "Initialization"
            x.append(self.data[:, 0])
            y.append(self.data[:, 1])
            plt.scatter(x, y)
        else:
            for k in range(self.cluster_num):
                x = []
                y = []
                for n in range(np.shape(self.data)[0]):
                    if self.r[n][k] == 1:
                        x.append(self.data[n][0])
                        y.append(self.data[n][1])
                plt.scatter(x, y)

        plt.scatter(self.centers[:, 0], self.centers[:, 1])
        plt.title('Clusters of Data Points - Yuxiang Liu')
        if note:
            plt.savefig('report/figures/%s' % note)
        plt.show()


if __name__ == '__main__':
    kmeans = KmeansCL()
    kmeans.classify()
    kmeans.show(note="Final")
