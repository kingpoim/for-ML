# for-ML
## 逻辑回归
### 逻辑回归与线性回归的联系与区别
#### 联系
线性回归其实可以说是逻辑回归的输入，将线性回归加入sigmoid函数，使得样本能压缩在（0，1）之间，加入阈值，从而判断样本分类。
#### 区别
+ 线性回归主要用于回归预测，逻辑回归主要用于解决分类问题。
+ 线性回归的参数计算方法是最小二乘法，逻辑回归的参数计算方法是梯度下降
+ 线性回归的输出是线性且是连续实数，逻辑回归输出是非线性，是[0,1]连续数
### 逻辑回归原理
在线性回归基础上加入sigmoid函数，通过极大似然（对数）求损失函数，再通过梯度下降求损失函数最大值。
### 逻辑回归损失函数推导及优化
![img](https://github.com/kingpoim/img_for_ml/blob/master/LR%E6%8E%A8%E5%AF%BC.jpeg)

### 正则化与模型评估指标
#### 正则化
+ 为了解决过拟合问题，在逻辑回归中，正则化需要自定义代价函数
![img](https://github.com/kingpoim/img_for_ml/blob/master/LR%E6%AD%A3%E5%88%99%E5%8C%96.png)
#### 模型评估指标
+ ROC-AUC
+ ROC是越接近（0，1）点越好
+ AUC值是越接近1越好
### 逻辑回归的优缺点
+ 优点：
1. 适合二分类问题
2. 计算代价不高，容易理解实现。可用于分布式计算
3. LR对于数据中小噪声的鲁棒性很好，并且不会受到轻微的多重共线性的特别影响。（L2正则化可帮助解决）

+ 缺点：
1. 容易欠拟合，分类精度不高。
2. 数据特征有缺失或者特征空间很大时表现效果并不好。
3. 只能解决二分类问题（softmax除外）
### 样本不均衡问题解决办法
1. 根据样本量给类型添加权重，即sklearn的class_weight参数
参考：https://blog.csdn.net/Candy_GL/article/details/82858471
2.调用fit函数时，通过sample_weight来自己调节每个样本权重
### sklearn参数
参考：https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
参考：https://blog.csdn.net/ustbclearwang/article/details/81235892
Sklearn.linear_model.LogisticRegression
(penalty=’l2’, dual=False, tol=0.0001, C=1.0, fit_intercept=True,intercept_scaling=1, class_weight=None, random_state=None, solver=’liblinear’, max_iter=100, multi_class=’ovr’,verbose=0, warm_start=False, n_jobs=1)

penalty:’l1’ or ‘l2’ ,默认’l2’ #惩罚

dual:bool 默认False ‘双配方仅用于利用liblinear解算器的l2惩罚。’

tol: float, 默认: 1e-4 ‘公差停止标准’

C:float 默认:1.0 正则化强度， 与支持向量机一样，较小的值指定更强的正则化。

fit_intercept: bool 默认:True 指定是否应将常量（a.k.a. bias或intercept）添加到决策函数中。

intercept_scaling:float ,默认:1 仅在使用求解器“liblinear”且self.fit_intercept设置为True时有用。 在这种情况下，x变为[x，self.intercept_scaling]，即具有等于intercept_scaling的常数值的“合成”特征被附加到实例矢量。 截距变为intercept_scaling * synthetic_feature_weight

class_weight: dict or ‘balanced’ 默认:None

 与{class_label：weight}形式的类相关联的权重。 如果没有给出，所有类都应该有一个权重。“平衡”模式使用y的值自动调整与输入数据中的类频率成反比的权重，如n_samples /（n_classes * np.bincount（y））。请注意，如果指定了sample_weight，这些权重将与sample_weight（通过fit方法传递）相乘。

random_state:int,RandomState实例或None，可选，默认值：None

在随机数据混洗时使用的伪随机数生成器的种子。 如果是int，则random_state是随机数生成器使用的种子; 如果是RandomState实例，则random_state是随机数生成器; 如果为None，则随机数生成器是np.random使用的RandomState实例。 在求解器=='sag'或'liblinear'时使用。

solver:{‘newton-cg’,’lbfgs’,’liblinear’,’sag’,’saga’}

默认: ‘liblinear’ 在优化问题中使用的算法。

对于小型数据集，'liblinear'是一个不错的选择，而'sag'和'saga'对于大型的更快。

对于多类问题，只有'newton-cg'，'sag'，'saga'和'lbfgs'处理多项损失; 'liblinear'仅限于’ovr’方案。'newton-cg'，'lbfgs'和'sag'只处理L2惩罚，而'liblinear'和'saga'处理L1惩罚。请注意，“sag”和“saga”快速收敛仅在具有大致相同比例的要素上得到保证。 您可以使用sklearn.preprocessing中的缩放器预处理数据。

max_iter: int 默认:100 仅适用于newton-cg，sag和lbfgs求解器。 求解器收敛的最大迭代次数。

muti_class:str,{‘ovr’:’multinomial’},默认:’ovr’

多类选项可以是'ovr'或'multinomial'。 如果选择的选项是'ovr'，那么二元问题适合每个标签。 另外，最小化损失是整个概率分布中的多项式损失拟合。 不适用于liblinear解算器。

verbose: int,默认:0 对于liblinear和lbfgs求解器，将verbose设置为任何正数以表示详细程度。

warm_start:bool 默认:False

设置为True时，重用上一次调用的解决方案以适合初始化，否则，只需擦除以前的解决方案。 对于liblinear解算器没用。

版本0.17中的新功能：warm_start支持lbfgs，newton-cg，sag，saga求解器。

n_jobs: int,默认:1

如果multi_class ='ovr'“，则在对类进行并行化时使用的CPU核心数。 无论是否指定'multi_class'，当``solver``设置为'liblinear'时，都会忽略此参数。 如果给定值-1，则使用所有核心。
-




