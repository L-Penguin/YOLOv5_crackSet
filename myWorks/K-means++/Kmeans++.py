# D:\Anaconda3\python.exe
# -*- coding = utf-8 -*-
# @Time : 2022/12/27 20:30
# @Author : L_PenguinQAQ
# @File : Kmeans++.py
# @Software: PyCharm
# @function: 简单介绍该文件功能


import pandas as pd
import numpy as np
import random
import math
from matplotlib import pyplot as plt


# 无监督算法，学习过程就是训练质心的位置，进行聚类
class Kmeans:
    # 添加init参数，默认init = 'random'就是标准Kmeans，init = 'Kmeans++'则为Kmeans++
    def __init__(self, k, n_init, init='random'):
        self.k = k
        self.n_init = n_init
        self.init = init

    def calc_distance(self, x1, x2):
        diff = x1 - x2
        distances = np.sqrt(np.square(diff).sum(axis=1))
        return distances

    def fit(self, x):
        m, n = x.shape
        if self.init == 'random':
            # 随机选定k个数据作为初始质心，不重复选取
            self.original_ = np.random.choice(m, self.k, replace=False)
            # 默认类别是从0到k-1
            self.original_center = x[self.original_]
        elif self.init == 'Kmeans++':
            first = np.random.choice(m)
            # 储存在一个列表中
            index_select = [first]
            # 继续选取k-1个点
            for i in range(1, self.k):
                all_distances = np.empty((m, 0))
                for j in index_select:
                    # 计算每个数据点到已选择的质心的距离
                    distances = self.calc_distance(x, x[j]).reshape(-1, 1)
                    # 把每个数据点到已选择的质心的距离储存在数组中，每个质心一列
                    all_distances = np.c_[all_distances, distances]
                # 找到每个点到已选择质心的最小距离
                min_distances = all_distances.min(axis=1).reshape(-1, 1)
                # 在min_distances里面选取距离较大的点作为下一个质心，我们就选最大的点
                index = np.argmax(min_distances)
                index_select.append(index)
            # 生成Kmeans++方法的初始质心，默认类别是从0到k-1
            self.original_center = x[index_select]

        while True:
            # 初始化一个字典，以类别作为key，赋值一个空数组
            dict_y = {}
            for j in range(self.k):
                dict_y[j] = np.empty((0, n))
            for i in range(m):
                distances = self.calc_distance(x[i], self.original_center)
                # 把第i个数据分配到距离最近的质心，存放在字典中
                label = np.argsort(distances)[0]
                dict_y[label] = np.r_[dict_y[label], x[i].reshape(1, -1)]
            centers = np.empty((0, n))
            # 对每个类别的样本重新求质心
            for i in range(self.k):
                center = np.mean(dict_y[i], axis=0).reshape(1, -1)
                centers = np.r_[centers, center]
            # 与上一次迭代的质心比较，如果没有发生变化，则停止迭代（也可考虑收敛时停止）
            result = np.all(centers == self.original_center)
            if result == True:
                return dict_y, centers
                break
            else:
                # 继续更新质心
                self.original_center = centers

    def select_optimal(self, x):
        # 储存每次的J值
        result = []
        # 储存每次的聚类质心
        center = []
        for i in range(self.n_init):
            dict_y_i, center_i = self.fit(x)
            # 计算J值
            for j in range(self.k):
                result_i = 0
                # 计算第j个类别的样本到类别质心的距离之和
                distance_j = np.sum(self.calc_distance(dict_y_i[j], center_i[j]))
                result_i += distance_j
            result.append(result_i)
            center.append(center_i)
        # 找到最小J值，对应的聚类质心
        index = np.argmin(result)
        self.original_center = center[index]

    def predict(self, x):
        y_preds = []
        m, n = x.shape
        for i in range(m):
            distances = self.calc_distance(x[i], self.original_center)
            y_pred = np.argsort(distances)[0]
            y_preds.append(y_pred)
        return y_preds


if __name__ == '__main__':
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    myKmeansPP = MyKmeansPlusPlus(2)
    myKmeansPP.fit(data)
