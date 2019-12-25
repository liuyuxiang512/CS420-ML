from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from data import Dataset
import matplotlib.pyplot as plt
import numpy as np


class EM(object):
    def __init__(self, n_components=5, dataset=None):
        self.model = GaussianMixture(n_components=n_components, max_iter=10000)
        self.n_components = n_components
        self.cluster_num = dataset.cluster_num
        self.data = dataset.data
        self.aic_bic = None
        self.aic = []
        self.bic = []
        self.best_k_aic = 0
        self.best_k_bic = 0
        self.model.fit(self.data)

    def aic_select(self):
        self.aic_bic = "aic"
        for k in range(self.n_components):
            gmm = GaussianMixture(n_components=k + 1, max_iter=10000)
            gmm.fit(self.data)
            self.aic.append(gmm.aic(self.data))
            self.best_k_aic = np.argmin(self.aic) + 1
        print("The optimal k by AIC is " + str(self.best_k_aic))
        self.model = GaussianMixture(n_components=self.best_k_aic)
        self.model.fit(self.data)

    def bic_select(self):
        self.aic_bic = "bic"
        for k in range(self.n_components):
            gmm = GaussianMixture(n_components=k + 1, max_iter=10000)
            gmm.fit(self.data)
            self.bic.append(gmm.bic(self.data))
            self.best_k_bic = np.argmin(self.bic) + 1
        print("The optimal k by BIC is " + str(self.best_k_bic))
        self.model = GaussianMixture(n_components=self.best_k_bic)
        self.model.fit(self.data)

    def show(self, note=None, n=False):
        plt.figure()
        plt.xlim((-2.5, 0.5 + 2 * self.cluster_num))
        plt.ylim((-2.5, 0.5 + 2 * self.cluster_num))
        labels = self.model.predict(self.data)
        plt.scatter(self.data[:, 0], self.data[:, 1], c=labels, s=15)
        if n and self.aic_bic == "aic":
            plt.savefig('report/figures/AIC_%s_%d' % (note, n))
        if n and self.aic_bic == "bic":
            plt.savefig('report/figures/BIC_%s_%d' % (note, n))


class VBEM(object):
    def __init__(self, n_components=5, dataset=None):
        self.model = BayesianGaussianMixture(n_components=n_components, max_iter=10000)
        self.n_components = n_components
        self.cluster_num = dataset.cluster_num
        self.data = dataset.data
        self.model.fit(self.data)

    def show(self, note=None, n=False):
        plt.figure()
        plt.xlim((-2.5, 0.5 + 2 * self.cluster_num))
        plt.ylim((-2.5, 0.5 + 2 * self.cluster_num))
        labels = self.model.predict(self.data)
        plt.scatter(self.data[:, 0], self.data[:, 1], c=labels, s=15)
        if n:
            plt.savefig('report/figures/VBEM_%s_%d' % (note, n))


def evaluation_sample_size():
    note = "sample_size"
    for n in np.array([30, 50, 80, 120, 180]):
        dataset = Dataset(data_num=[n, n, n])
        dataset.generate()
        gmm = EM(n_components=5, dataset=dataset)
        gmm.aic_select()
        gmm.show(note=note, n=n)
        gmm.bic_select()
        gmm.show(note=note, n=n)
        gmm_VB = VBEM(n_components=5, dataset=dataset)
        gmm_VB.show(note=note, n=n)


def evaluation_cluster_num():
    note = "cluster_num"
    for num in np.array([1, 2, 3, 4, 5]):
        dataset = Dataset(cluster_num=num)
        dataset.generate()
        gmm = EM(n_components=5, dataset=dataset)
        gmm.aic_select()
        gmm.show(note=note, n=num)
        gmm.bic_select()
        gmm.show(note=note, n=num)
        gmm_VB = VBEM(n_components=5, dataset=dataset)
        gmm_VB.show(note=note, n=num)


if __name__ == '__main__':
    # evaluation_sample_size()
    evaluation_cluster_num()
