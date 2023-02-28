## K-means++算法

> 针对`Kmeans`算法的第一步选取初始簇类中心做改进，可以直观地将这改进理解为这K个初始聚类中心相互之间应该分得越开越好。

### 算法步骤

- 从数据集中随机选取一个样本作为初始聚类中心$c_1$；
- 首先计算每个样本与当前已有聚类中心之间得最短距离(即与最近得一个聚类中心的距离)，用$D(x)$表示；接着计算每个样本被选为下一个聚类中心的概率$\frac{D(x)^2}{\sum_{x\in{X}}D(x)^2}$。最后，按照轮盘法选择出下一个聚类中心；
- 重复第2步直到选择出共K个聚类中心；
- 之后的过程与经典`K-means`算法中第2步至第4步相同。