#生成连续性数据二分类数据并进行可视化
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#固定随机数种子
np.random.seed(4)
#分别生成40*2维正态数组
X_1 = np.random.randn(40,2)
#生成连续型变量y,前20个为0,中间20个为1,后20个为2
y_1 = X_1[:,0]+2*X_1[:,1]+np.random.randn(40)
fig,ax = plt.subplots(figsize=(8,6))
#构建y_1 与 X_1[:,0] 的散点图，设置散点形状为o
ax.scatter(y_1,X_1[:,0],s=30,c='b',marker='o')
#构建y_1 与 X_1[:,1] 的散点图，设置散点形状为x
ax.scatter(y_1,X_1[:,1],s=30,c='r',marker='x')
plt.show()
#SVR算法的学习
from sklearn.svm import SVR
clf_1 = SVR(gamma='auto')
print(clf_1.fit(X_1,y_1))
#训练结果并预测
print(clf_1.support_vectors_)
print(clf_1.score(X_1,y_1))
#预测值与实际值可视化比较
y_hat = clf_1.predict(X_1)
#设置图片尺寸
plt.figure(figsize=(10,6))
#设置t变量
t = np.arange(len(X_1))
#绘制原始变量y_1曲线
plt.plot(t,y_1,'r',linewidth=2,label='y_1')
#绘制y_test曲线
plt.plot(t,y_hat,'g',linewidth=2,label='y_hat')
plt.legend()
plt.show()
#获取训练结果并预测
from sklearn import metrics
#用Scikit-learn计算MAE
print("MAE:",metrics.mean_absolute_error(y_1,y_hat))
#用Scikit-learn计算MSE
print("MSE:",metrics.mean_squared_error(y_1,y_hat))
#用Scikit-learn计算RMSE
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_1,y_hat)))
