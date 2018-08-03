#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/8/2
"""

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()  # for plot styling

from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs


def test_of_k_means():
    # 创建测试点，X是数据，y是标签，X:(300,2), y:(300,)
    X, y_true = make_blobs(n_samples=300, centers=9, cluster_std=0.60, random_state=0)
    kmeans = KMeans(n_clusters=9)  # 将数据聚类
    kmeans.fit(X)  # 数据X
    y_kmeans = kmeans.predict(X)  # 预测

    # 颜色范围viridis: https://matplotlib.org/examples/color/colormaps_reference.html
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=20, cmap='viridis')  # c是颜色，s是大小

    centers = kmeans.cluster_centers_  # 聚类的中心
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=40, alpha=0.5)  # 中心点为黑色

    plt.show()  # 展示


if __name__ == '__main__':
    test_of_k_means()
