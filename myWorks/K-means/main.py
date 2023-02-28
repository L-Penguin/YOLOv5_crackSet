# D:\Anaconda3\python.exe
# -*- coding = utf-8 -*-
# @Time : 2022/12/9 15:20
# @Author : L_PenguinQAQ
# @File : main.py
# @Software: PyCharm
# @function: K-means算法测试文件

import os
import numpy as np
from Kmeans import Kmeans, kmeans_plot


def loadPoints(path):
    files = os.listdir(path)
    points = []
    for f in files:
        if os.path.splitext(f)[-1] == '.txt':
            txt = open(os.path.join(path, f))
            for line in txt:
                line = line.strip()
                w = float(line.split(' ')[-2]) * 640
                h = float(line.split(' ')[-1]) * 640
                points.append([w, h])

    return np.array(points, dtype="float32")


# model = Kmeans(k=5)
# model.fit(blobs(centers=5, random_state=1, n_features=2))
# kmeans_plot(model)
# model.fit(blobs(centers=3, random_state=3, n_features=3))
# kmeans_plot(model)
path = r'D:\Data\YOLO\dataSets\concreteCrackSet\labels'
points = loadPoints(path)

# type为1 范式距离；type为2 IOU。
model = Kmeans(k=9, type=2, init="kmeans")
model.fit(points)
for i, p in enumerate(model.centroids):
    print(f'聚类中心点{i+1}: {round(p[0])} * {round(p[1])} = {round(p[0]) * round(p[1])}')
kmeans_plot(model, "Kmeans++_IOU")
