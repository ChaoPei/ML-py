import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from iris import plot_decision_regions


def generate_nolinear_data(verbose=False):
    np.random.seed(1)
    X_xor = np.random.randn(200, 2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, -1)
    if verbose:
        plt.scatter(
            X_xor[y_xor == 1, 0],
            X_xor[y_xor == 1, 1],
            c='b',
            marker='x',
            label='1')
        plt.scatter(
            X_xor[y_xor == -1, 0],
            X_xor[y_xor == -1, 1],
            c='r',
            marker='o',
            label='2')
        plt.show()
    return X_xor, y_xor


def train_rbf_svm(X, y):
    svc = SVC(kernel='rbf', gamma=1.0, C=1.0, random_state=1)
    svc.fit(X, y)
    plot_decision_regions(
        X=X,
        y=y,
        classifier=svc,
        test_idx=None,
        xlabel='x',
        ylabel='y',
        title='SVC')
    return svc


if __name__ == "__main__":
    X, y = generate_nolinear_data()
    train_rbf_svm(X, y)