# D:\Anaconda3\python.exe
# -*- coding = utf-8 -*-
# @Time : 2022/12/9 14:49
# @Author : L_PenguinQAQ
# @File : Kmeans.py
# @Software: PyCharm
# @function: K-means算法模型模块文件


import math
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
import random

np.seterr(divide='ignore',invalid='ignore')


class Kmeans:
    def __init__(self, k=2, tolerance=0.01, max_iter=3000, type=1, init='kmeans'):
        self.k = k
        self.tol = tolerance
        self.max_iter = max_iter
        self.type = type
        self.features_count = -1
        self.classifications = None
        self.centroids = None
        self.init = init

    def RWS(self, P, r):
        """利用轮盘法选择下一个聚类中心"""
        q = 0  # 累计概率
        for i in range(len(P)):
            q += P[i]  # P[i]表示第i个个体被选中的概率
            if i == (len(P) - 1):  # 对于由于概率计算导致累计概率和小于1的，设置为1
                q = 1
            if r <= q:  # 产生的随机数在m~m+P[i]间则认为选中了i
                return i

    def initCentroids(self, data):
        if self.init == 'kmeans':
            # 初始化聚类中心（维度：k个 * features种数）
            self.centroids = np.zeros([self.k, data.shape[1]])
            # 从数据集中随机选取k个数据
            randomPoints = random.sample(range(0, data.shape[0]+1), self.k)
            for i in range(self.k):
                self.centroids[i] = data[randomPoints[i]]
                print(f'选取初始点{i+1}: {self.centroids[i]}')
        elif self.init == 'kmeans++':
            self.centroids = []
            init = random.randint(0, data.shape[0])
            self.centroids.append(data[init])
            print(f'选取初始点1: {self.centroids[0]}')
            for i in range(1, self.k):
                weights = []
                for d in data:
                    weight = self.calWeight(self.centroids, d)
                    weights.append(weight)
                # 避免产生inf造成问题
                weights = np.array(weights) / 10

                precents = np.square(weights) / np.square(weights).sum()
                b = np.square(weights).sum()
                a = precents.sum()
                r = np.random.random()  # r为0至1的随机数
                choiced_index = self.RWS(precents, r)  # 利用轮盘法选择下一个聚类中心
                print(f'选取初始点{i+1}: {data[choiced_index]}')
                self.centroids.append(data[choiced_index])

            self.centroids = np.array(self.centroids)

    def calWeight(self, c, point):
        if self.type == 1:
            weights = np.linalg.norm(point - c, axis=1)
            return np.min(weights)
        elif self.type == 2:
            temp = []
            for p in c:
                min_w = np.minimum(point[0], p[0])
                min_h = np.minimum(point[1], p[1])
                inter_area = np.multiply(min_w, min_h)  # inter_area表示重叠面积
                box_area = np.multiply(p[0], p[1])
                cluster_area = np.multiply(point[0], point[1])
                iou = inter_area / (box_area + cluster_area - inter_area)
                temp.append(1 - iou)
            return min(temp)


    def fit(self, data):
        """
        :param data: numpy数组，约定shape为：(数据数量，数据维度)
        :type data: numpy.ndarray
        """
        self.features_count = data.shape[1]
        self.initCentroids(data)

        for i in range(self.max_iter):
            # 清空聚类列表
            self.classifications = [[] for i in range(self.k)]
            # 对每个点与聚类中心进行距离计算
            for d in data:
                # 预测分类
                classification = self.predict(d)
                # 加入类别
                self.classifications[classification].append(d)

            # 记录前一次的结果
            prev_centroids = np.ndarray.copy(self.centroids)

            # 更新中心
            for classification in range(self.k):
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            # 检测相邻两次中心的变化情况
            for c in range(self.k):
                if np.linalg.norm(prev_centroids[c] - self.centroids[c]) > self.tol:
                    break
                # 如果都满足条件（上面循环没break），则返回
                else:
                    return

    def predict(self, data):
        if self.type == 1:
            # 距离计算
            distances = np.linalg.norm(data - self.centroids, axis=1)   # 求范数
            # 最小距离索引
            return distances.argmin()
        elif self.type == 2:
            temp = []
            for p in self.centroids:
                min_w = np.minimum(data[0], p[0])
                min_h = np.minimum(data[1], p[1])
                inter_area = np.multiply(min_w, min_h)  # inter_area表示重叠面积
                box_area = np.multiply(p[0], p[1])
                cluster_area = np.multiply(data[0], data[1])
                iou = inter_area / (box_area + cluster_area - inter_area)
                temp.append(1 - iou)
            return np.array(temp).argmin()

# 将聚类模型结果可视化函数
def kmeans_plot(kmeans_model, title=""):
    """
    :param kmeans_model: 训练的kmeans模型
    :type kmeans_model: Kmeans | FastKmeans

    :param title: 展示图标题
    :type title: string
    """
    style.use('ggplot')
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    # 2D
    if kmeans_model.features_count == 2:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot()

        for i in range(kmeans_model.k):
            color = colors[i % len(colors)]

            for feature_set in kmeans_model.classifications[i]:
                ax.scatter(feature_set[0], feature_set[1], marker="x", color=color, s=50, linewidths=1)

        for centroid in kmeans_model.centroids:
            ax.scatter(centroid[0], centroid[1], marker="o", color="k", s=50, linewidths=3)
    # 3D
    elif kmeans_model.features_count == 3:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')

        for i in range(kmeans_model.k):
            color = colors[i%len(colors)]

            for feature_set in kmeans_model.classifications[i]:
                ax.scatter(feature_set[0], feature_set[1], feature_set[2], marker="x", color=color, s=50, linewidths=1)

        for centroid in kmeans_model.centroids:
            ax.scatter(centroid[0], centroid[1], centroid[2], marker="o", color="k", s=50, linewidths=3)

    # 解决中文标题问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 设置标题
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    data = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [1, 1.5], [2, 2.5], [3, 3.5], [4, 4.5]], dtype="float16")
    model = Kmeans(k=3, type=2)
    model.fit(data)
    kmeans_plot(model)