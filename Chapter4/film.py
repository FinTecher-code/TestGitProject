import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('../data/3_film.csv')
print(df.head())
#调整直方图尺寸
df.hist(xlabelsize=12,ylabelsize=12,figsize=(12,7))
plt.show()
#绘制密度图 density是密度图意思
df.plot(kind='density',subplots=True,layout=(2,2),sharex=False,
        fontsize=8,figsize=(12,7))
plt.show()
#绘制箱线图
df.plot(kind = 'box',subplots = True,layout = (2,2),sharex = False,
        sharey = False,fontsize = 8,figsize = (12,7))
plt.show()
#多变量的数据可视化之相关系数热力图
#设置变量名
names = ['filmnum','filmsize','ratio','quality']
#计算变量之间的相关系数矩阵
correlations = df.corr()
#绘制相关系数热力图
fig = plt.figure() #调用figure创建一个绘图对象
#调用画板绘制第一个子图
ax = fig.add_subplot(1,1,1)
#绘制热力图，从0.3 到 1
cax = ax.matshow(correlations,vmin = 0.3,vmax = 1)
#将matshow生成热力图设置为颜色渐变条
fig.colorbar(cax)
#生成0~4,步长为1
ticks = np.arange(0,4,1)
ax.set_xticks(ticks) #生成刻度
ax.set_yticks(ticks) #生成刻度
ax.set_xticklabels(names) #生成x轴标签
ax.set_yticklabels(names) #生成y轴标签
plt.show()
#散点图矩阵
from pandas.plotting import scatter_matrix
scatter_matrix(df,figsize=(8,8),c='b')
plt.show()
#选取特征变量与响应变量，进行数据划分
X = df.iloc[:,1:4] #选取data中的X变量
y = df.filmnum #设定target为y
from sklearn.model_selection import train_test_split
#把X,y转化为数组形式，以便于计算
X = np.array(X.values)
y = np.array(y.values)
#将25%的数据构建测试样本，剩余作为训练样本
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=1)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
#进行线性回归操作
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
print('求解截距项为：',lr.intercept_)
print('求解系数为：',lr.coef_)
#对测试集的预测
y_hat = lr.predict(X_test)
print(y_hat[0:9]) #打印前10个数据
#对测试集相应变量实际值与预测值的比较
plt.figure(figsize=(10,6)) #设置图片尺寸
t = np.arange(len(X_test)) #创建t变量
#绘制y_test曲线
plt.plot(t,y_test,'r',linewidth = 2,label ='y_test')
#绘制y_hat曲线
plt.plot(t,y_hat,'g',linewidth = 2,label='y_train')
plt.legend() #设置图例
plt.show()
#对预测结果进行评估
from sklearn import metrics
from sklearn.metrics import r2_score
#拟合优度R^2的方法一
print("r2:",lr.score(X_test,y_test))
#拟合优度R^2的方法二
print("r2_score:",r2_score(y_test,y_hat))
#用Scikit-learn计算MAE
print("MAE:",metrics.mean_absolute_error(y_test,y_hat))
#用Scikit-learn计算MSE
print("MSE:",metrics.mean_squared_error(y_test,y_hat))
#用Scikit-learn计算RMSE
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test,y_hat)))