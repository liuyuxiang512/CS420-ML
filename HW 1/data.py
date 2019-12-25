import matplotlib.pyplot as plt
import numpy as np


class Dataset(object):
    def __init__(self, cluster_num=3, data_num=np.array([100, 150, 200, 250, 300]), dimension=2):
        self.cluster_num = cluster_num
        self.data_num = data_num
        self.dimension = dimension
        self.data = np.zeros((np.sum(data_num[:cluster_num]), dimension))

        means = []
        for num in range(cluster_num):
            mean = []
            for dim in range(dimension):
                mean_x = np.random.normal(2 * num, 0.4)
                mean.append(mean_x)
            means.append(mean)
        self.cluster_mean = np.array(means)
        # print(means)
        # self.cluster_cov

    def generate(self):
        num = 0
        for i in range(self.cluster_num):
            sigma = np.random.uniform(0.5, 0.6)
            for j in range(self.data_num[i]):
                for dim in range(self.dimension):
                    data_x = np.random.normal(self.cluster_mean[i][dim], sigma)
                    self.data[num + j][dim] = data_x
            num += self.data_num[i]

    def show(self, divide=True):
        if self.dimension == 2:
            plt.xlim((-2.5, 2 * self.cluster_num + 0.5))
            plt.ylim((-2.5, 2 * self.cluster_num + 0.5))
            if divide:
                num = 0
                for i in range(self.cluster_num):
                    x = self.data[num: num + self.data_num[i], 0]
                    y = self.data[num: num + self.data_num[i], 1]
                    num += self.data_num[i]
                    plt.scatter(x, y)
            else:
                plt.scatter(self.data[:, 0], self.data[:, 1])
            plt.title('Distribution of Data Points - Yuxiang Liu')
            plt.show()


if __name__ == '__main__':
    dataset = Dataset()
    dataset.generate()
    dataset.show()
    dataset.show(divide=False)
