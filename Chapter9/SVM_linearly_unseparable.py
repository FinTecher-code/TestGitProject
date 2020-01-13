#生成线性可分二分类数据并进行可视化
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#固定随机数种子
np.random.seed(2)
#分别生成两个20*2维正态数组，其中第一个以[-1,-1]为中心，第二个以[1,1]为中心
#np.r_表示按列连接两个矩阵
X_n = np.r_[np.random.randn(20,2)-[1,1],np.random.randn(20,2)+[1,1]]
#生成类别变量y，前20个为0，后20个为1
y_n = [0] *20 +[1]*20
fig,ax = plt.subplots(figsize=(8,6)) #创建子图，大小为8*6
#构建y = 0 的散点图，设置散点形状为o
ax.scatter(X_n[0:20,1],X_n[0:20,0],s=30,c='b',marker='o',label='y=0')
#构建y = 1 的散点图，设置散点形状为x
ax.scatter(X_n[20:40,1],X_n[20:40,0],s=30,c='r',marker='x',label='y=1')
ax.legend()
plt.show()
#线性不可分SVM算法的学习
from sklearn.svm import SVC
#设定模型为线性核函数的SVM
clf_n = SVC(kernel='linear')
print(clf_n.fit(X_n,y_n))
#获取训练结果并预测
#查看拟合模型的w
print(clf_n.coef_)
#查看支持向量
print(clf_n.support_vectors_)
#预测y
print(clf_n.predict(X_n))
#查看SVM预测精度
print(clf_n.score(X_n,y_n))

#重新设置松弛变量，实现SVM算法的学习
#设定模型为线性核函数的SVM
clf_nSV = SVC(kernel='linear',C=0.2)
print(clf_nSV.fit(X_n,y_n))
#训练结果并预测
print(clf_nSV.coef_)    #查看拟合模型的w
print(clf_nSV.support_vectors_) #查看支持向量
print(clf_nSV.predict(X_n)) #预测y
print(clf_nSV.score(X_n,y_n)) #查看SVM预测精度