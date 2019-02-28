import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from iris import load_iris_data, plot_decision_regions


class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=123):
        """
        eta: learning rate
        """
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.errors = []
        self.weights = rng.normal(
            loc=0.0, scale=0.01, size=1 + X.shape[1])  # 1 for bias
        for _ in range(self.n_iter):
            error = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                error += int(update != 0.0)
            self.errors.append(error)
        return self

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


def train_perceptron(X, y, verbose=True):
    ppn = Perceptron(eta=0.1, n_iter=20)
    ppn.fit(X, y)
    if verbose:
        plt.plot(range(1, len(ppn.errors) + 1), ppn.errors, marker='o')
        plt.xlabel('epoch')
        plt.ylabel('error')
        plt.show()
    return ppn


if __name__ == "__main__":
    X, y = load_iris_data()
    ppn = train_perceptron(X, y, verbose=False)
    plot_decision_regions(
        X,
        y,
        ppn,
        xlabel="sepal length [cm]",
        ylabel="petal length [cm]",
        title="perceptron")
