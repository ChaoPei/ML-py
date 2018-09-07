# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datetime
import sys

from kmeans import KMeansClassifier


def load_data(path):
    df = pd.read_csv(path, sep="\t", header=0, dtype=str, na_filter=False)
    return np.array(df).astype(np.float)

if __name__ == "__main__":

    data = load_data("./data/test.txt")
    k = 3
    classifier = KMeansClassifier(k)
    classifier.fit(data)
    centers = classifier._centroids
    labels = classifier._labels
    sse = classifier._sse
    print(labels)
    print(sse)
    colors = [
        'b',
        'g',
        'r',
        'k',
        'c',
        'm',
        'y',
        '#e24fff',
        '#524C90',
        '#845868']
    for i in range(k):
        index = np.nonzero(labels == i)[0]
        x = data[index, 0]
        y = data[index, 1]
        for j in range(len(x)):
            plt.text(x[j], y[j], str(i), color=colors[i], fontdict={'weight': 'bold', 'size': 6})

        plt.scatter(centers[i, 0], centers[i, 1], marker='x', color=colors[i], linewidths=7)
        plt.title("SSE={:.2f}".format(sse))

    plt.axis([-7, 7, -7, 7])
    outname = "./result/{}_clusters_".format(k) + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".png"
    plt.savefig(outname)
    plt.show()
