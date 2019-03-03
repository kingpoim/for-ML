# for-ML
## 决策树
### 信息论基础
参考：https://www.cnblogs.com/xmeo/p/6543054.html
#### 熵
+ 熵是对平均不确定性的度量.
![img](https://github.com/kingpoim/img_for_ml/blob/master/%E7%86%B5.png)
#### 联合熵
+ 两个随机变量X，Y的联合分布，可以形成联合熵Joint Entropy，用H(X,Y)表示。
#### 条件熵
+ 在随机变量X发生的前提下，随机变量Y发生所新带来的熵定义为Y的条件熵，用H(Y|X)表示，用来衡量在已知随机变量X的条件下随机变量Y的不确定性,  用H(X|Y)表示
#### 信息增益
+ 在一个条件下，信息不确定性减少的程度！
#### 基尼不纯度
+ 将来自集合中的某种结果随机应用于集合中某一数据项的预期误差率。
![img](https://github.com/kingpoim/img_for_ml/blob/master/%E5%9F%BA%E5%B0%BC%E4%B8%8D%E7%BA%AF%E5%BA%A6.png)
### 决策树的不同分类算法的原理及应用场景
#### ID3算法
参考：https://www.cnblogs.com/wxquare/p/5379970.html
+ 选择熵减少程度最大的特征来划分数据（贪心），也就是“最大信息熵增益”原则
![img](https://github.com/kingpoim/img_for_ml/blob/master/ID3.png)
#### C4.5算法
+ 解决了ID3因分支数量不同造成的不公平
![img](https://github.com/kingpoim/img_for_ml/blob/master/c4.5.png)
#### CART分类树
+ 使用基尼指数（Gini）来选择最好的数据分割的特征
![img](https://github.com/kingpoim/img_for_ml/blob/master/cart.png)
### 回归树原理
参考：https://blog.csdn.net/weixin_36586536/article/details/80468426
+ 与分类树不同的是，回归树用平方误差最小化准则
+ 回归树是可以用于回归的决策树模型，一个回归树对应着输入空间（即特征空间）的一个划分以及在划分单元上的输出值.与分类树不同的是，回归树对输入空间的划分采用一种启发式的方法，会遍历所有输入变量，找到最优的切分变量jj和最优的切分点ss，即选择第jj个特征xjxj和它的取值ss将输入空间划分为两部分，然后重复这个操作
而如何找到最优的jj和ss是通过比较不同的划分的误差来得到的。一个输入空间的划分的误差是用真实值和划分区域的预测值的最小二乘来衡量的

### 决策树防止过拟合手段
#### 造成过拟合主要原因：
1. 样本问题
2. 方法问题
#### 解决办法
1. 合理、有效地抽样，用相对能够反映业务逻辑的训练集去产生决策树
2. 剪枝：提前停止树的增长或者对已经生成的树按照一定的规则进行后剪枝。（先剪枝，后剪枝）

### 模型评估
参考：https://www.cnblogs.com/fushengweixie/p/8039991.html

### sklearn参数详解，Python绘制决策树
#### sklearn参数详解
参考：https://cloud.tencent.com/developer/article/1146079
sklearn.tree.DecisionTreeClassifier
        (criterion='gini', splitter='best', max_depth=None, min_samples_split=2, 
        min_samples_leaf=1,min_weight_fraction_leaf=0.0, max_features=None, 
        random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
        min_impurity_split=None, class_weight=None, presort=False)
criterion:特征选择的标准，有信息增益和基尼系数两种，使用信息增益的是ID3和C4.5算法（使用信息增益比），使用基尼系数的CART算法，默认是gini系数。

splitter:特征切分点选择标准，决策树是递归地选择最优切分点，spliter是用来指明在哪个集合上来递归，有“best”和“random”两种参数可以选择，best表示在所有特征上递归，适用于数据集较小的时候，random表示随机选择一部分特征进行递归，适用于数据集较大的时候。

max_depth:决策树最大深度，决策树模型先对所有数据集进行切分，再在子数据集上继续循环这个切分过程，max_depth可以理解成用来限制这个循环次数。

min_samples_split:子数据集再切分需要的最小样本量，默认是2，如果子数据样本量小于2时，则不再进行下一步切分。如果数据量较小，使用默认值就可，如果数据量较大，为降低计算量，应该把这个值增大，即限制子数据集的切分次数。

min_samples_leaf:叶节点（子数据集）最小样本数，如果子数据集中的样本数小于这个值，那么该叶节点和其兄弟节点都会被剪枝（去掉），该值默认为1。

min_weight_fraction_leaf:在叶节点处的所有输入样本权重总和的最小加权分数，如果不输入则表示所有的叶节点的权重是一致的。

max_features:特征切分时考虑的最大特征数量，默认是对所有特征进行切分，也可以传入int类型的值，表示具体的特征个数；也可以是浮点数，则表示特征个数的百分比；还可以是sqrt,表示总特征数的平方根；也可以是log2，表示总特征数的log个特征。

random_state:随机种子的设置，与LR中参数一致。

max_leaf_nodes:最大叶节点个数，即数据集切分成子数据集的最大个数。

min_impurity_decrease:切分点不纯度最小减少程度，如果某个结点的不纯度减少小于这个值，那么该切分点就会被移除。

min_impurity_split:切分点最小不纯度，用来限制数据集的继续切分（决策树的生成），如果某个节点的不纯度（可以理解为分类错误率）小于这个阈值，那么该点的数据将不再进行切分。

class_weight:权重设置，主要是用于处理不平衡样本，与LR模型中的参数一致，可以自定义类别权重，也可以直接使用balanced参数值进行不平衡样本处理。

presort:是否进行预排序，默认是False，所谓预排序就是提前对特征进行排序，我们知道，决策树分割数据集的依据是，优先按照信息增益/基尼系数大的特征来进行分割的，涉及的大小就需要比较，如果不进行预排序，则会在每次分割的时候需要重新把所有特征进行计算比较一次，如果进行了预排序以后，则每次分割的时候，只需要拿排名靠前的特征就可以了。
#### python绘制决策树
参考：https://blog.csdn.net/sinat_29957455/article/details/76553987

