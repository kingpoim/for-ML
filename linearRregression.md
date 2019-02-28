# for-ML
## 线性回归
### 概念
+ 有监督学习：常用于分类，有label的学习。
+ 无监督学习：常用于聚类，无label的学习。
+ 泛化能力：指对新样本的适应能力（方法学习到的模型对未知数据的预测能力）。以泛化误差评价，也就是损失函数的期望（期望误差）。

+ 欠拟合：模型复杂度低，没有很好地学习到特征。
+ 解决欠拟合：提高模型复杂度。
+ 过拟合：模型复杂度过高，模型学习得过于贴合测试集了。
+ 防止过拟合：
1. Early Stopping提前终止
2. 数据集扩增
3. 正则化：L1,L2范式
4. 加入dropout层，用于神经网络，随机切断神经元的联系。
+ 交叉验证：k-fold cross validation，将数据分成k份，循环选取1份作为验证集，其余为测试集，循环k次。
### 线性回归原理
+ 学习一个线性模型来最好的拟合数据。
![img](https://github.com/kingpoim/img_for_ml/blob/master/linearregression.png)
### 线性回归损失函数、代价函数、目标函数
+ 损失函数 =（真实值-预测值）^2，损失函数越小，证明预测值越接近真实值，就是函数模拟得越好。
+ 代价函数 代价函数是损失函数的平均值
+ 目标函数 参考：https://blog.csdn.net/huobumingbai1234/article/details/80348369
用最小二乘给出目标函数
前提：
1. 各个样本之间是独立的
2. 误差服从均值是0，方差是σ² 的高斯分布(中心极限定理)
![img](https://github.com/kingpoim/img_for_ml/blob/master/%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E7%9B%AE%E6%A0%87%E5%87%BD%E6%95%B0.jpeg)
### 优化方法
+ 梯度下降法
 参考：http://www.cnblogs.com/ooon/p/4947688.html
 ![img](https://github.com/kingpoim/img_for_ml/blob/master/%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E5%9B%BE.png)
 ![img](https://github.com/kingpoim/img_for_ml/blob/master/%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E5%9B%BE2.png)
+ 牛顿法
收敛速度快
参考：https://blog.csdn.net/qq_30986521/article/details/83515186
![img](https://github.com/kingpoim/img_for_ml/blob/master/%E7%89%9B%E9%A1%BF%E6%B3%95.png)
+ 拟牛顿法
参考：https://www.cnblogs.com/shixiangwan/p/7532830.html#top
修改了牛顿法每次需要用hessian矩阵的逆矩阵的劣势，使用正定矩阵来近似Hessian矩阵的逆。
![img](https://github.com/kingpoim/img_for_ml/blob/master/%E6%8B%9F%E7%89%9B%E9%A1%BF%E6%B3%95.png)
### 线性回归的评估指标
参考：https://blog.csdn.net/qq_21840201/article/details/80428526
SSE(和方差、误差平方和)：The sum of squares dueto error
MSE(均方差、方差)：Meansquared error
RMSE(均方根、标准差)：Root mean squared error
R-square(确定系数)：Coefficientof determination
Adjusted R-square：Degree-of-freedomadjusted coefficient of determination
![img](https://github.com/kingpoim/img_for_ml/blob/master/%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E8%AF%84%E4%BC%B0%E6%8C%87%E6%A0%87.png)
### sklearn参数详解
+ LinearRegression(fit_intercept=True,normalize=False,copy_X=True,n_jobs=1)
+ fit_intercept:是否有截据，如果没有则直线过原点。
+ normalize:是否将数据归一化
+ copy_X:默认为True，当为True时，X会被copied,否则X将会被覆写。
+ n_jobs:默认值为1。计算时使用的核数。




