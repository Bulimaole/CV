# -*- coding: utf-8 -*-
"""
陈小虎

PCA detail
"""
import numpy as np

class CPCA(object):
    def __init__(self, X, K):
        self.X = X
        self.K = K
        self.centrX = []
        self.C = []
        self.U = []
        self.Z = []

        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()

    def _centralized(self):
        print('样本矩阵X:\n', self.X)
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])
        print('样本集的均值特征：\n', mean)
        centrX = self.X - mean
        print('样本矩阵X的中心化centrX:\n', centrX)
        return centrX

    def _cov(self):
        ns = np.shape(self.centrX)[0]
        C = np.dot(self.centrX.T, self.centrX)/(ns - 1)
        print('样本矩阵X的协方差矩阵C：\n', C)
        return C

    def  _U(self):
        a,b = np.linalg.eig(self.C)
        print('样本集的协方差矩阵C的特征值：\n', a)
        print('样本集的协方差矩阵C的特征向量：\n', b)
        ind = np.argsort(-1*a)
        UT = [b[:,ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print('%d降阶维转换矩阵U:\n'%self.K, U)
        return U

    def _Z(self):
        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z：\n', Z)
        return Z

if __name__=='__main__':
    X = np.array([[11, 12, 15],
                  [11, 22, 55],
                  [33, 34, 52],
                  [34, 34,11],
                  [23, 12, 23],
                  [56, 76, 89],
                  [34, 22, 22],
                  [23, 25, 91],
                  [11, 61, 72],
                  [83, 22, 6]])
    K = np.shape(X)[1] - 1
    print('样本集（10行3列， 10个样例， 每个样例3个特征）:\n', X)
    pca = CPCA(X,K)