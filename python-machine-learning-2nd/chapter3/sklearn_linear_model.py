from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from iris import load_iris_data, plot_decision_regions
import numpy as np


def train_perceptron(X_train_std, y_train, X_test_std, y_test):
    ppn = Perceptron(
        max_iter=50,
        tol=None,
        eta0=0.1,
        shuffle=True,
        random_state=1,
        verbose=1)
    ppn.fit(X_train_std, y_train)
    y_pred = ppn.predict(X_test_std)
    print("Actual iter: %d" % ppn.n_iter_)
    print("Misclassified samples: %d of %d" % (
        (y_test != y_pred).sum(), y_test.sum()))
    print("Accuracy: %.2f" % (accuracy_score(y_test, y_pred)))
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))  # y have only 1 dim
    plot_decision_regions(
        X=X_combined_std,
        y=y_combined,
        classifier=ppn,
        test_idx=range(105, 150),
        xlabel='petal length [standardized]',
        ylabel='petal width [standardized]',
        title='Perceptron')
    return ppn


def train_logistic(X_train_std, y_train, X_test_std, y_test):
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    lgr = LogisticRegression(C=100.0, random_state=1)
    lgr.fit(X_train_std, y_train)
    y_pred = lgr.predict(X_test_std)
    print("Accuracy: %.2f" % (accuracy_score(y_test, y_pred)))

    plot_decision_regions(
        X=X_combined_std,
        y=y_combined,
        classifier=lgr,
        test_idx=range(105, 150),
        xlabel='petal length [standardized]',
        ylabel='petal width [standardized]',
        title='Logistic')
    return lgr


def train_svc(X_train_std, y_train, X_test_std, y_test):
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    svc = SVC(kernel='linear', C=100.0, random_state=1)
    svc.fit(X_train_std, y_train)
    y_pred = svc.predict(X_test_std)
    print("Accuracy: %.2f" % (accuracy_score(y_test, y_pred)))

    plot_decision_regions(
        X=X_combined_std,
        y=y_combined,
        classifier=svc,
        test_idx=range(105, 150),
        xlabel='petal length [standardized]',
        ylabel='petal width [standardized]',
        title='SVC')
    return svc


if __name__ == "__main__":
    X_train_std, y_train, X_test_std, y_test = load_iris_data()
    train_svc(X_train_std, y_train, X_test_std, y_test)
