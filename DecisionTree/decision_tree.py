# -*- coding: utf-8 -*-

import numpy as np

'''
Using a single decision tree to classify
'''


class DecisionTreeClassifier(object):
    def __init__(self, max_iter=100, num_steps=10, num_iter=50):
        self._max_iter = max_iter
        self._num_steps = num_steps
        self._num_iter = num_iter
    

    '''
    A single level decision tree: stump, classifying a sample
    by it's one dimension feature compare to the threshold.

    class: 0 or 1.

    @x: feature of a sample
    @axis: feature dimension
    @thresh: threshold for classifying
    @comp_rule: compare rule for classifying
    '''

    def _stump_classify(self, x, axis, threshold, comp_rule):
        
        ret_array = np.ones((np.shape(x)[0], 1))  # size: m * 1
        if comp_rule == 'lt':  # 'lt' rule: less than threshold will be classified as 1
            ret_array[x[:, axis] <= threshold] = -1.0
        else:
            ret_array[x[:, axis] > threshold] = -1.0
        return ret_array

    '''
    find the best single level decision tree

    @init_weights: the weights of every sample
    '''

    def _build_stump(self, x, y, init_weights):

        x_mat = np.mat(x)
        y_mat = np.mat(y)
        labels_mat = y_mat.T

        m, n = np.shape(x)  # m is sample number, n is feature number
        best_stump = {}
        min_error = np.inf

        best_classify_result = np.mat(np.zeros((m, 1)))

        for axis in range(n):  # every feature
            min_val = x_mat[:, axis].min()
            max_val = x_mat[:, axis].max()
            step_size = (max_val - min_val) / self._num_steps

            # choose different threshold for classifying
            for j in range(-1, int(self._num_steps) + 1):
                for comp_rule in ['lt', 'gt']:
                    threshold = (min_val + float(j) * step_size)
                    pred_vals = self._stump_classify(x_mat, axis, threshold,
                                                     comp_rule)

                    # record the errors using this threshold
                    errs = np.mat(np.ones((m, 1)))
                    errs[pred_vals == labels_mat] = 0

                    # calculate weighted errors
                    weighted_errs = init_weights.T * errs
                    if weighted_errs < min_error:
                        min_error = weighted_errs
                        best_classify_result = pred_vals
                        best_stump['dim'] = axis
                        best_stump['threshold'] = threshold
                        best_stump['comp_rule'] = comp_rule

        return best_stump, best_classify_result, min_error

    '''
    train a few weak classifiers
    '''

    def fit(self, x, y):
        weak_classifier_array = []
        m = np.shape(x)[0]
        weights = np.mat(np.ones((m, 1)) / m)
        all_classifier_pred = np.mat(np.zeros((m, 1)))
        for i in range(self._num_iter):
            best_stump, best_classify_result, min_err = self._build_stump(
                x, y, weights)

            # calculate this weak classifier weight
            alpha = float(0.5 * np.log((1.0 - min_err)) / max(
                min_err, 1e-16))  # alpha is depends on it's definition
            best_stump['alpha'] = alpha
            weak_classifier_array.append(best_stump)
            expon = np.multiply(-1 * alpha * np.mat(y).T, best_classify_result)
            weights = np.multiply(weights, np.exp(expon))
            weights = weights / weights.sum()

            # calculate all classifier errors, if errors == 0, break train
            all_classifier_pred += alpha * best_classify_result
            pred_errors = np.multiply(
                np.sign(all_classifier_pred) != np.mat(y).T, np.ones((m, 1)))

            error_rate = pred_errors.sum() / m
            if error_rate == 0.0:
                break

        return weak_classifier_array

    def predict(self, test_x, classifiers):
        test_mat = np.mat(test_x)
        m = np.shape(test_mat)[0]
        all_classifier_pred = np.mat(np.zeros((m, 1)))
        for i in range(len(classifiers)):
            classifier_pred = self._stump_classify(
                test_mat, classifiers[i]['dim'], classifiers[i]['threshold'],
                classifiers[i]['comp_rule'])

            all_classifier_pred += classifier_pred * classifiers[i]['alpha']

        return np.sign(all_classifier_pred)
