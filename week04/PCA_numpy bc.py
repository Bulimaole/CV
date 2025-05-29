# -*- coding= utf-8 -*-
"""
陈小虎

PAC_numpy
"""

import numpy as np
class PCA():
    def __init__(self,n_components):
        self.n_components = n_components

    def fit_transform(self,X):
        self.n_features = X.shape[1]
        X = X - X.mean(axis=0)
        self.covariance = np.dot(X.T,X)/X.shape[0]
        eig_vals,eig_vectors = np.linalg.eig(self.covariance)
        idx = np.argsort(-eig_vals)
        self.components_ = eig_vectors[:,idx[:self.n_components]]
        return np.dot(X,self.components_)

pca = PCA(n_components=2)
X = np.array([[2,3,34,-4], [3,-5,64,23], [2,34,12,23], [43,21,-6,23], [92,-33,34,12], [12,34,-12,32]])
newX=pca.fit_transform(X)
print(newX)