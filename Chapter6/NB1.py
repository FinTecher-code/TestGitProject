import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df = pd.read_csv('../data/6_credit.csv')
print(df.head())
#数据可视化
#把credit为1的数据筛选出来形成单独的数据集
a1 = df[df['credit']==1]
a2 = df[df['credit']==2]
a3 = df[df['credit']==3]
#创建子图，大小为8*5
fig,ax = plt.subplots(figsize = (8,5))
#构建a1的散点图，设置散点形状为o
ax.scatter(a1['income'],a1['points'],s=30,c='b',marker='o',label='credit=1')
#构建a2的散点图，设置散点形状为*
ax.scatter(a2['income'],a2['points'],s=30,c='r',marker='x',label='credit=2')
#构建a3的散点图，设置散点形状为^
ax.scatter(a3['income'],a3['points'],s=30,c='g',marker='^',label='credit=3')
ax.legend()
ax.set_xlabel('income')
ax.set_ylabel('points')
plt.show()
#展示不同等级下的其他特征变量之间的散点图
fig,ax = plt.subplots(figsize= (8,5))
#构建a1的散点图，设置散点形状为o
ax.scatter(a1['house'],a1['numbers'],s=30,c='b',marker='o',label='credit=1')
#构建a2的散点图，设置散点形状为*
ax.scatter(a2['house'],a2['numbers'],s=30,c='r',marker='x',label='credit=2')
#构建a3的散点图，设置散点形状为^
ax.scatter(a3['house'],a3['numbers'],s=30,c='g',marker='^',label='credit=3')
ax.legend(loc='upper left')
ax.set_xlabel('house')
ax.set_ylabel('numbers')
plt.show()
#选取特征变量与响应变量，并进行数据划分
X = df.iloc[:,1:6]  #取df的前5列为X变量
y = df['credit']
#把X、y转化为数组形式，以便于计算
X = np.array(X.values)
y = np.array(y.values)
from sklearn.model_selection import train_test_split
#以25%的数据构建测试样本，剩余作为训练样本
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=1)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
#进行高斯朴素贝叶斯估计
from sklearn.naive_bayes import GaussianNB
#设定模型为高斯朴素贝叶斯
GNB = GaussianNB()
GNB.fit(X_train,y_train)
#获取各个类标记对应的先验概率
print(GNB.class_prior_)
#获取各个类标记对应的样本数
print(GNB.class_count_)
#获取各个类标记在各个特征值上的均值
print(GNB.theta_)
#获取各个类标记在各个特征值上的方差
print(GNB.sigma_)
#根据求出参数对测试集进行预测
y_pred = GNB.predict(X_test)
print(y_pred)
#模型比较和选择包
from sklearn import model_selection
#计算混淆矩阵，主要来评估分类的准确性
from sklearn.metrics import confusion_matrix
#计算精度得分
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred)) #计算准确率
print(confusion_matrix(y_true=y_test,y_pred = y_pred))  #计算混淆矩阵
#与逻辑回归分类算法评价效果的比较
from sklearn.linear_model.logistic import LogisticRegression
clf = LogisticRegression(solver='liblinear',multi_class='ovr')
clf.fit(X_train,y_train)
#预测测试集
y_pred_classifier = clf.predict(X_test)
print(accuracy_score(y_test,y_pred_classifier)) #计算准确率
print(confusion_matrix(y_true=y_test,y_pred = y_pred_classifier))  #计算混淆矩阵
