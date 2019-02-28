import numpy as np
from iris import load_iris_data, plot_decision_regions
import matplotlib.pyplot as plt


class LogisticBGD(object):
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
            output = self.activation(self.net_input(X))
            errors = y - output
            # update by all samples
            self.weights[1:] += self.eta * X.T.dot(errors)
            self.weights[0] += self.eta * errors.sum()
            cost = self.cost_function(y, output)
            self.costs.append(cost)
        return self

    def cost_function(self, y, output):
        """
        logistic cost function
        """
        cost = -y.dot(np.log(output)) - (1 - y).dot(np.log(1 - output))
        return cost

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def activation(self, z):
        """
        sigmoid function
        """
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        output = self.activation(self.net_input(X))
        return np.where(output > 0.5, 1, -1)


def train_logisticBGD(X, y):
    lrgd = LogisticBGD(eta=0.05, n_iter=20, random_state=1)
    lrgd.fit(X, y)
    plot_decision_regions(
        X=X,
        y=y,
        classifier=lrgd,
        test_idx=None,
        xlabel='petal length [standardized]',
        ylabel='petal width [standardized]',
        title='LogisticBGD')

    plt.plot(range(1, len(lrgd.costs) + 1), lrgd.costs, marker='o')
    plt.xlabel('epoch')
    plt.xlabel('cost')
    plt.xlabel('LogisticBGD-costs')
    plt.show()


if __name__ == "__main__":

    X_train_std, y_train, X_test_std, y_test = load_iris_data()
    # logistic works for binary classifiction
    X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
    y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]
    train_logisticBGD(X_train_01_subset, y_train_01_subset)
