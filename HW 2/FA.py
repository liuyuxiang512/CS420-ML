from sklearn.decomposition import FactorAnalysis
import numpy as np
import pandas as pd

# parameters
N = 100  # sample size
n = 7  # observed variable dimension
m = 3  # latent variable dimension
square_sigma = 0.1
mu = 0  # mean

head = ['N', 'n', 'm', 'variance', 'mean', 'AIC', 'BIC', 'AIC value', 'BIC value']

array = []
# for N in [10, 20, 50, 80, 100, 200, 300, 500]:
# for n in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]:
# for m in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
# for square_sigma in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]:
for mu in [-1, -0.8, -0.5, -0.2, 0, 0.2, 0.5, 0.8, 1]:
    # data set generation
    arrayA = np.random.rand(n, m)
    arrayX = np.zeros((N, n))
    for i in range(N):
        y_t = np.random.randn(m)
        e_t = np.random.normal(mu, np.sqrt(square_sigma), n)
        x_t = np.dot(arrayA, y_t) + mu + e_t
        arrayX[i] = x_t

    # two-step model selection
    aic = []
    bic = []
    best_k_aic = 0
    best_k_bic = 0
    for k in range(2 * m - 1):
        transformer = FactorAnalysis(n_components=k+1)
        arrayX_transformed = transformer.fit_transform(arrayX)
        aic.append(transformer.score(arrayX) * N - (k + 1))
        bic.append(transformer.score(arrayX) * N - (k + 1) * np.log(N) * 0.5)
    best_k_aic = np.argmax(aic) + 1
    best_k_bic = np.argmax(bic) + 1

    line = [N, n, m, square_sigma, mu, best_k_aic, best_k_bic, np.max(aic), np.max(bic)]
    array.append(line)

for line in array:
    print(line)

test = pd.DataFrame(columns=head, data=array)
# test.to_csv('results/sample_size.csv', index=None, encoding='gbk')
# test.to_csv('results/dimension_n.csv', index=None, encoding='gbk')
# test.to_csv('results/dimension_m.csv', index=None, encoding='gbk')
# test.to_csv('results/variance.csv', index=None, encoding='gbk')
test.to_csv('results/mean.csv', index=None, encoding='gbk')
