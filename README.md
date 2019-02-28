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

