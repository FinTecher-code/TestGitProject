#生成线性可分二分类数据并进行可视化
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#固定随机数种子
np.random.seed(3)
#分别生成两个20*2维抛物线式正态数组，其中第一个以[-1,-1]为中心，第二个以[1,1]为中心
#np.r_表示按列连接两个矩阵
X_sq = np.r_[np.random.randn(20,2)**2-[1,1],np.random.randn(20,2)**2+[1,1]]
#生成类别变量y，前20个为0，后20个为1
y_sq = [0] *20 +[1]*20
fig,ax = plt.subplots(figsize=(8,6)) #创建子图，大小为8*6
#构建y = 0 的散点图，设置散点形状为o
ax.scatter(X_sq[0:20,1],X_sq[0:20,0],s=30,c='b',marker='o',label='y=0')
#构建y = 1 的散点图，设置散点形状为x
ax.scatter(X_sq[20:40,1],X_sq[20:40,0],s=30,c='r',marker='x',label='y=1')
ax.legend()
plt.show()
#非线性不可分SVM算法的学习，设置核函数为多项式
from sklearn.svm import SVC
#设定模型为线性核函数的SVM
clf_sq = SVC(kernel='poly',degree=2)
print(clf_sq.fit(X_sq,y_sq))
#获取训练结果并预测
#查看支持向量
print(clf_sq.support_vectors_)
#预测y
print(clf_sq.predict(X_sq))
#查看SVM预测精度
print(clf_sq.score(X_sq,y_sq))
#分线性可分SVM算法的学习，设置核函数为径向基函数
clf_rbf = SVC(kernel='rbf',gamma=1)
print(clf_rbf.fit(X_sq,y_sq))
#查看支持向量
print(clf_rbf.support_vectors_)
#预测y
print(clf_rbf.predict(X_sq))
#查看SVM预测精度
print(clf_rbf.score(X_sq,y_sq))
#非线性可分SVM算法的学习，设置核函数为sigmoid函数
clf_sig = SVC(kernel='sigmoid',gamma=1)
print(clf_sig.fit(X_sq,y_sq))
print(clf_sig.support_vectors_)
print(clf_sig.predict(X_sq))
print(clf_sig.score(X_sq,y_sq))