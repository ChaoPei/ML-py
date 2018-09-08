#! -*- conding: utf-8 -*-
'''
softmax classifier implementation
'''

import math
import numpy as np
import time

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize


class SoftmaxClassifier():
    def __init__(self,
                 learning_rate=0.45,
                 max_iters=1000,
                 weights_decay_lambda=0.0):
        self.learning_rate = learning_rate  # 学习速率
        self.max_iters = max_iters          # 最大迭代次数
        self.weights_decay_lambda = weights_decay_lambda  # 权重衰减

    def calc_feature_result(self, x, i):
        '''计算第i类特征和权重的指数结果'''

        return math.exp(np.dot(x, self.weights[i]))

    def calc_probability(self, x, i):
        '''计算第i类的分类概率'''

        i_feas_res = self.calc_feature_result(x, i)
        all_feas_res = sum(
            [self.calc_feature_result(x, j) for j in range(self.k)])

        return i_feas_res / all_feas_res

    def calc_loss_gradient(self, X, labels):
        '''计算批量代价函数的损失和梯度，函数公式：http://deeplearning.stanford.edu/wiki/index.php/Softmax%E5%9B%9E%E5%BD%92'''
        
        k = len(set(labels))
        m, n = X.shape
        # 标签二值化，将所属的类别设置为1，其余类别设置为0，便于计算softmax的损失
        bin_labels = label_binarize(labels, classes=np.unique(labels).tolist()).reshape((m, k))
        # print("bin_labels shape: (%d, %d)" % (bin_labels.shape[0], bin_labels.shape[1]))

        # features shape: (n+1, m)
        features = np.concatenate((X, np.ones((m, 1))), axis=1).reshape((m, n + 1)).T
        # print("features shape: (%d, %d)" % (features.shape[0], features.shape[1]))
        # print("weights shape: (%d, %d)" % (self.weights.shape[0], self.weights.shape[1]))
        # self.weights shape: (k, n+1)
        results = self.weights.dot(features)
        # probilities shape: (k, m)
        probilities = np.exp(results) / np.sum(np.exp(results), axis=0)
        # print("probilities shape: (%d, %d)" % (probilities.shape[0], probilities.shape[1]))

        # 计算损失值
        loss = (-1 / m) * np.sum(np.multiply(bin_labels, np.log(probilities).T)) \
            + self.weights_decay_lambda * np.sum(np.square(self.weights))
        # 计算梯度
        grad = (-1 / m) * (features.dot(bin_labels - probilities.T)).T  \
            + self.weights_decay_lambda * self.weights

        return loss, grad

    def train(self, X_train, y_train):

        k = len(set(y_train))
        m, n = X_train.shape
        self.weights = np.zeros((k, n+1))

        for i in range(self.max_iters):
            loss, grad = self.calc_loss_gradient(X_train, y_train)
            self.weights -= self.learning_rate * grad

            if i % 100 == 0:
                print("train loop iters: %d, loss: %f" % (i, loss))
        print(self.weights)

    def predict(self, X_test):
        X_test = np.c_[X_test, np.ones((X_test.shape[0], 1))]
        prods = self.weights.dot(X_test.T)
        results = np.exp(prods) / np.sum(np.exp(prods), axis=0)
        preds = results.argmax(axis=0)
        return preds


# 绘制结果图 
def plot_res(preds, labels_test, features_test, learning_rate):
    s = set(preds)
    col = ['r', 'b', 'g', 'y', 'm']
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(0, len(s)):
        index1 = (preds == i)
        index2 = (labels_test == i)
        x1 = features_test[index1, :]
        x2 = features_test[index2, :]
        ax.scatter(
            x1[:, 0], x1[:, 1], color=col[i], marker='v', linewidths=0.5)
        ax.scatter(x2[:, 0], x2[:, 1], color=col[i], marker='.', linewidths=12)

    plt.title('learning rating=' + str(learning_rate))
    plt.legend(
        ('c1:predict', 'c1:true', 'c2:predict', 'c2:true', 'c3:predict',
         'c3:true', 'c4:predict', 'c4:true', 'c5:predict', 'c5:true'),
        shadow=True,
        loc=(0.01, 0.4))
    plt.show()


if __name__ == "__main__":

    # 生成数据: 特征是二维坐标，根据二维坐标位置分为五种类别
    features, labels = [], []
    for idx in range(1000):
        label = idx % 5
        x = np.random.random(2) + label
        y = label
        features.append(x)
        labels.append(y)

    features = np.array(features)
    labels = np.array(labels)

    X_train, X_test, Y_train, Y_test = train_test_split(
        features, labels, test_size=0.3, random_state=2333)

    print("Start trian...")
    time1 = time.time()
    softmax_classifier = SoftmaxClassifier()
    softmax_classifier.train(X_train, Y_train)
    time2 = time.time()
    print("train cost time: %d" % (time2 - time1))

    print("Start predict...")
    time3 = time.time()
    preds = softmax_classifier.predict(X_test)
    time4 = time.time()
    print("predict cost time: %d" % (time4 - time3))

    score = accuracy_score(Y_test, preds)
    print("The accruacy socre is " + str(score))

    #plot_res(preds, Y_test, X_test, softmax_classifier.learning_rate)

