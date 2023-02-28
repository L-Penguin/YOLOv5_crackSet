## K-means算法

> 一种无监督的学习。
>
> `K-means`算法是一种聚类分析（`cluster analysis`）的算法，其主要是来计算数据聚类的算法，主要通过不断地取离种子点最近均值的算法。
>
> 算法思路：对给定的样本集，事先确定聚类簇数K，让簇内的样本尽可能密集分布在一起，使簇间的距离尽可能大。该算法试图使集群数据分为n组独立数据样本，使n组集群间的方差相等，数学描述为最小化惯性或集群内的平方和。

### 算法步骤

1. 选择初始化的`k`个样本作为初始化聚类中心$a={a_1,a_2,...,a_k}$；
2. 针对数据集中每个样本$x_i$计算它到`k`个聚类中心的距离并将其分到距离组西奥的聚类中心所对应的类中；
3. 针对每个类别$S_j$，重新计算它的聚类中心$a_j=\frac{1}{|S_i|}\sum_{X\in{S_i}}X$；
5. 重复上面2、3两步操作，直到达到某个中止条件（迭代次数、最小误差变化等）。

<img src="./K-means_structure.png" style="margin: 0 20px"/>

```python
class Kmeans:
    def __init__(self, k=2, tolerance=0.01, max_iter=300):
        self.k = k
        self.tol = tolerance
        self.max_iter = max_iter
        self.features_count = -1
        self.classifications = None
        self.centroids = None

    def fit(self, data):
        """
        :param data: numpy数组，约定shape为：(数据数量，数据维度)
        :type data: numpy.ndarray
        """
        self.features_count = data.shape[1]
        # 初始化聚类中心（维度：k个 * features种数）
        self.centroids = np.zeros([self.k, data.shape[1]])
        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            # 清空聚类列表
            self.classifications = [[] for i in range(self.k)]
            # 对每个点与聚类中心进行距离计算
            for feature_set in data:
                # 预测分类
                classification = self.predict(feature_set)
                # 加入类别
                self.classifications[classification].append(feature_set)

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
        # 距离
        distances = np.linalg.norm(data - self.centroids, axis=1)
        # 最小距离索引
        return distances.argmin()

```

### 总结

- `Kmeans`算法在迭代的过程中<strong style="color:red;">使用所有点的均值</strong>作为新的质点(中心点)，如果簇中存在异常点，将导致均值偏差比较严重；
- `Kmeans`算法是初始质点敏感的，选择不同的初始质点可能导致不同的簇划分规则；
- `Kmeans`算法缺点：
  - K值是用户定的，在进行数据处理前，K值是未知的，不同的K值得到的结果也不一样；
  - 对初始簇质心是敏感的；
  - 不适合发现非凸形状的簇或者大小差别较大的簇；
  - 特殊值(离群值)对模型的影响比较大。
- `Kmeans`算法优点：
  - 理解容易，聚类效果不错；
  - 处理大数据集的时候，该算法可以保证较好的伸缩性和高效率；
  - 当簇近似高斯分布的时候，效果非常不错。

