#! -*- coding: utf-8 -*-

"""simple perceptron implementation"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os


class Perceptron(object):
  
  def __init__(self, lr=0.01, num_iter=50, random_state=17):
    self._lr =  lr
    self._num_iter = num_iter
    self._random_state = random_state
  
  def fit(self, X, y):
    """
    Fit training data
    """
    rgen = np.random.RandomState(self._random_state)
    self._weights = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
    self._errors = []

    for _ in range(self._num_iter):
      err = 0
      for xi, target in zip(X, y):
        update = self._lr * (target - self.predict(xi))
        self._weights[1:] += update * xi  
        self._weights[0] += update  #  bias
        err += int(update != 0.0)
      self._errors.append(err)
    return self

  def net_input(self, X):
    return np.dot(X, self._weights[1:] + self._weights[0])
  
  def predict(self, X):
    return np.where(self.net_input(X) >= 0.0, 1, -1)


def draw_data(X, y):
  plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
  plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

  plt.xlabel('sepal length [cm]')
  plt.ylabel('petal length [cm]')
  plt.legend(loc='upper left')
  plt.show()


def draw_decision_regions(X, y, classifer, resolution=0.02):
  markers = ('s', 'x', 'o', '^', 'v')
  colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
  cmap = ListedColormap(colors[:len(np.unique(y))])

  x0_min, x0_max = X[:, 0].min()-1, X[:, 0].max()+1
  x1_min, x1_max = X[:, 1].min()-1, X[:, 1].max()+1

  xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, resolution),
                         np.arange(x1_min, x1_max, resolution))
  z = classifer.predict(np.array([xx0.ravel(), xx1.ravel()]).T)
  z = z.reshape(xx1.shape)
  


if __name__ == "__main__":

  project_dir = os.path.dirname(os.path.realpath(__file__))
  df = pd.read_csv(os.path.join(project_dir, "data.csv"), header=None)
  y = df.iloc[0:100, 4].values
  y = np.where(y == 'Iris-setosa', -1, 1)
  X = df.iloc[0:100, [0,2]].values
  # draw_data(X, y)

  perceptron = Perceptron(lr=0.1, num_iter=20)
  perceptron.fit(X, y)
  plt.plot(range(1, len(perceptron._errors) + 1), perceptron._errors, marker='o')
  plt.xlabel('epochs')
  plt.ylabel('errors')
  plt.show()
