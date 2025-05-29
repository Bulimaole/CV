#coding=utf-8

"""
陈小虎
PCA_sklearn
"""

import numpy as np
from sklearn.decomposition import PCA
X = np.array([[3,6,23,-1], [4,5,-7,9], [3,5,23,-4], [43,32,-6,-7], [23,5,-7,9], [5,-3,45,75]])
pca = PCA(n_components=2)
pca.fit(X)
newX=pca.fit_transform(X)
print(newX)