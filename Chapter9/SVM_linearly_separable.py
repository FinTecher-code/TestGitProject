#生成线性可分二分类数据并进行可视化
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#固定随机数种子
np.random.seed(0)
#分别生成两个20*2维正态数组，其中第一个以[-2,-2]为中心，第二个以[2,2]为中心
#np.r_表示按列连接两个矩阵
X = np.r_[np.random.randn(20,2)-[2,2],np.random.randn(20,2)+[2,2]]
#生成类别变量y，前20个为0，后20个为1
y = [0] *20 +[1]*20
fig,ax = plt.subplots(figsize=(8,6)) #创建子图，大小为8*6
#构建y = 0 的散点图，设置散点形状为o
ax.scatter(X[0:20,1],X[0:20,0],s=30,c='b',marker='o',label='y=0')
#构建y = 1 的散点图，设置散点形状为x
ax.scatter(X[20:40,1],X[20:40,0],s=30,c='r',marker='x',label='y=1')
ax.legend()
plt.show()
#SVM算法的学习
from sklearn.svm import SVC
#设定模型为线性核函数的SVM
clf = SVC(kernel='linear')
print(clf.fit(X,y))
#获取训练结果并预测
#查看拟合模型的w
print(clf.coef_)
#查看支持向量
print(clf.support_vectors_)
#预测y
print(clf.predict(X))
#查看SVM预测精度
print(clf.score(X,y))
#绘制超平面与支持向量
w = clf.coef_[0]    #获取参数w
a = -w[0]/w[1]  #获取斜率
#生成xx为-5到5之间步长为1的数组
xx = np.linspace(-5,5)
#生产超平面yy
yy = a*xx-(clf.intercept_[0])/w[1]
#获取支持向量第一列
b = clf.support_vectors_[0]
#生成下方的yy
yy_down = a*xx+(b[1] -a*b[0])
#获取支持向量的第二列
b = clf.support_vectors_[-1]
#生成上方的yy
yy_up = a*xx+(b[1]-a*b[0])
#绘制超平面
plt.plot(xx,yy)
#绘制超平面下方的直线
plt.plot(xx,yy_down,'--')
#绘制超平面上方的直线
plt.plot(xx,yy_up,'--')
#绘制支持向量的散点
plt.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],
            c='black',s=30,facecolors='none')
#构建y=0的散点图，设置散点形状为o,cmap=plt.cm.Paired表示绘图样式选择Paired主题
plt.scatter(X[0:20,1],X[0:20,0],s=30,c='b',
            marker='o',label='y=0',cmap=plt.cm.Paired)
#构建y=1的散点图，设置散点形状为*
plt.scatter(X[20:40,1],X[20:40,0],s=30,c='r',
            marker='x',label='y=1',cmap=plt.cm.Paired)
plt.legend()
plt.show()