# -*- coding: utf-8 -*-


'''
k-means implement
'''

import numpy as np


class KMeansClassifier():

    ''' this is a k-means classifier '''

    # @k: class number
    # @initCent: init center node
    # @max_iter: max iterator number
    def __init__(self, k=3, init_cent='random', max_iter=500):
        self._k = k
        self._init_cent = init_cent
        self._max_iter = max_iter

        self._cluster_assment = None    # store cluster info
        self._centroids = None          # store center info
        self._labels = None             # store sample label
        self._sse = None                # store sum of distance

    # calc Eulerian distance between two vector(array)
    def _calc_eulerian_dist(self, arr1, arr2):
        return np.math.sqrt(sum(np.power(arr1 - arr2, 2)))

    # calc Manhattan distance between two vector(array)
    def _calc_manhattan_dist(self, arr1, arr2):
        return sum(np.abs(arr1 - arr2))

    # randomly generate k center (not choose for all data)
    # @data: feature vector
    def _rand_cent(self, data, k):

        n = data.shape[1]  # get feature vector dims
        centrodis = np.empty((k, n))  # store k center with n dims feature
        for j in range(n):
            min_j = min(data[:, j])   # min in all data feature  
            range_j = float(max(data[:, j] - min_j))   # all data feature range

            # generate k random data for the jth dimension feature
            centrodis[:, j] = (min_j + range_j * np.random.rand(k, 1)).flatten()  # flattern return 1 dimension vector with k elements

        return centrodis

    # train to fit
    def fit(self, data):

        if not isinstance(data, np.ndarray) or \
          isinstance(data, np.matrixlib.defmatrix.matrix):
            try:
                data = np.asarray(data)
            except:
                raise TypeError("numpy.ndarray required for data")

        m = data.shape[0]   # get sample number

        # create a empty cluster info matrix, the first column save the sample index and the second column save the dist.
        self._cluster_assment = np.zeros((m, 2))

        if self._init_cent == "random":
            self._centroids = self._rand_cent(data, self._k)
        
        # start train iter
        cluster_changed = True
        for n in range(self._max_iter):
            print("start {}th iter train...".format(n))
            cluster_changed = False

            # classify every sample to their cloest cluster center and save the center index and distance
            for i in range(data.shape[0]):
                min_dist = np.inf
                min_center_idx = -1
                for j in range(self._centroids.shape[0]):
                    sample_feature = data[i, 0]
                    center_feature = self._centroids[j, :]

                    dist = self._calc_eulerian_dist(sample_feature, center_feature)

                    if dist < min_dist:
                        min_dist = dist
                        min_center_idx = j
                
                # update the cluster info
                if self._cluster_assment[i, 0] != min_center_idx:
                    cluster_changed = True
                    self._cluster_assment[i, :] = min_center_idx, min_dist**2

            # if fit, exit loop
            if not cluster_changed:
                break
            
            # find the center for every cluster
            for i in range(self._k):
                index_all = self._cluster_assment[:, 0]
                value = np.nonzero(index_all == i)  # find all ith cluster sample
                ith_cluster_samples = data[value[0]]
                self._centroids[i, :] = np.mean(ith_cluster_samples, axis=0)
        
        self._labels = self._cluster_assment[:, 0]
        self._sse = sum(self._cluster_assment[:, 1])
    
    def predict(self, test_sample):
        if not isinstance(test_sample, np.ndarray):
            try:
                test_sample = np.asarray(test_sample)
            except:
                raise TypeError("numpy.ndarray required for test_sample")
        
        m = test_sample.shape[0]
        preds = np.empty((m, ))
        for i in range(m):
            min_dist = np.inf
            for j in range(self._k):
                dist = self._calc_eulerian_dist(self._centroids[j, :], test_sample[i, :])
                if dist < min_dist:
                    min_dist = dist
                    preds[i] = j
        return preds
            


                
