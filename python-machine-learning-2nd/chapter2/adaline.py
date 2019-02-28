import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from iris import load_iris_data, plot_decision_regions


class AdalineGD(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.weights = None
        self.costs = []

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.weights = rng.normal(loc=0.0, scale=0.01, size=X.shape[1] + 1)
        for i in range(self.n_iter):
            output = self.net_input(X)
            active = self.activation(output)
            errors = y - active
            # update by all samples
            self.weights[1:] += self.eta * X.T.dot(errors)
            self.weights[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.costs.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def activation(self, z):
        return z

    def predict(self, X):
        res = self.activation(self.net_input(X))
        return np.where(res > 0.0, 1, -1)


class AdalineSGD(object):
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=123):
        self.eta = eta
        self.n_iter = n_iter
        self.weight_initizalized = False
        self.shuffle = shuffle
        self.random_state = random_state
        self.rng = np.random.RandomState(self.random_state)
        self.weights = None
        self.costs = []

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def activation(self, z):
        return z

    def predict(self, X):
        res = self.activation(self.net_input(X))
        return np.where(res > 0.0, 1, -1)

    def _init_weights(self, n):
        self.weights = self.rng.normal(loc=0.0, scale=0.01, size=n + 1)
        self.weight_initizalized = True

    def _shuffle(self, X, y):
        r = self.rng.permutation(len(y))
        return X[r], y[r]

    def _update_weights(self, xi, target):
        """
        SGD: update weights by one sample
        """
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.weights[1:] += self.eta * xi.dot(error)
        self.weights[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def partial_fit(self, X, y):
        if not self.weight_initizalized:
            self._init_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def fit(self, X, y):
        self._init_weights(X.shape[1])
        self.costs.clear()
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.costs.append(avg_cost)
        return self


def standardization(X):
    X_std = np.copy(X)
    for i in range(X.shape[1]):
        X_std[:, i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()
    return X_std


def lr_compare(X, y):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    ada1 = AdalineGD(n_iter=10, eta=0.01)
    ada1.fit(X, y)
    ax[0].plot(range(1, len(ada1.costs) + 1), np.log10(ada1.costs), marker='o')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('log(sum-squared-error')
    ax[0].set_title('Adaline-lr 0.01')

    ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
    ax[1].plot(range(1, len(ada2.costs) + 1), ada2.costs, marker='o')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('log(sum-squared-error')
    ax[1].set_title('Adaline-lr 0.0001')
    plt.show()


def train_AdalineBGD(X, y):
    ada = AdalineGD(n_iter=15, eta=0.01)
    ada.fit(X, y)
    plot_decision_regions(
        X,
        y,
        ada,
        xlabel='sepal length [standardized]',
        ylabel='petal length [standardized]',
        title='Adaline - Batch Gradient Descent')

    plt.plot(range(1, len(ada.costs) + 1), ada.costs, marker='o')
    plt.xlabel('epoch')
    plt.ylabel('Sum-squared-error')
    plt.show()


def train_AdalineSGD(X, y):
    ada = AdalineSGD(n_iter=15, eta=0.01, random_state=123)
    ada.fit(X, y)
    plot_decision_regions(
        X,
        y,
        ada,
        xlabel='sepal length [standardized]',
        ylabel='petal length [standardized]',
        title='Adaline - Stochastic Gradient Descent')
    plt.plot(range(1, len(ada.costs) + 1), ada.costs, marker='o')
    plt.xlabel('epoch')
    plt.ylabel('avg-costs')
    plt.show()


if __name__ == "__main__":
    X, y = load_iris_data()
    X_std = standardization(X)
    # lr_compare(X_std, y)
    train_AdalineBGD(X_std, y)
    # train_AdalineSGD(X_std, y)
