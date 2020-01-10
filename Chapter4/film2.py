import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df = pd.read_csv('../data/3_film.csv')
print(df.head())
#插入一列全为1的数组
df.insert(1,'Ones',1) #在df第1列和第2列之间插入一列全是1的数组
print(df.head())
#选取特征变量与响应变量，并划分数据
cols = df.shape[1] #计算df的列数
#取数据df的第2列之后的数据作为X变量
X = df.iloc[:,1:cols]
y = df.filmnum
from sklearn.model_selection import train_test_split
#把X、y转化为数组形式，以便于计算
X = np.array(X.values)
y = np.array(y.values)
#以25%的数据构建测试样本，剩余作为训练样本
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25
                                                 ,random_state=1)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
#构建计算成本函数的函数
def computeCost(X,y,theta):
    inner = np.power(((X*theta.T)-y),2)
    return np.sum(inner)/(2*len(X))
#构建梯度下降法求解函数
#梯度下降算法函数，X、y是输入变量，theta是参数，alpha是学习率，iters是梯度下降迭代次数
def gradientDescent(X,y,theta,alpha,iters):
    #构建零值矩阵
    temp = np.matrix(np.zeros(theta.shape))
    #计算需要求解的参数个数
    parameters = int(theta.ravel().shape[1])
    #构建iters个0的数组
    cost = np.zeros(iters)
    for i in range(iters):
        error = (X * theta.T) - y
        # 对于theta中的每一个元素依次计算
        for j in range(parameters):
            #计算两矩阵相乘
            term = np.multiply(error,X[:,j])
            #更新法则
            temp[0,j] = theta[0,j] - ((alpha/len(X))*np.sum(term))
            theta = temp
            #基于求出来的theta求解成本函数
            cost[i] = computeCost(X,y,theta)
    return theta,cost
#设定相关参数的初始值，并带入gradientDescent()函数中求解
alpha = 0.000001 #设定学习流率
iters = 100 #设定迭代次数
theta = np.matrix(np.array([0,0,0,0]))
#采用gradientDescent()函数来优化求解
g,cost = gradientDescent(X,y,theta,alpha,iters)
#代入初始值并求解
print(g)
#对测试集X_test进行预测
y_hat = X_test * g.T #求出预测集y_test的预测值
print("y_hat:",y_hat)
#设置图片尺寸
plt.figure(figsize=(10,6))
#创建t变量
t = np.arange(len(X_test))
#绘制y_test曲线
plt.plot(t,y_test,'r',linewidth = 2,label = 'y_test')
#绘制y_hat曲线
plt.plot(t,y_hat,'g',linewidth = 2,label = 'y_hat')
plt.legend()
plt.show()
#对预测结果进行评价
from sklearn import metrics
from sklearn.metrics import r2_score
#拟合优度R^2的方法二
print("r2_score:",r2_score(y_test,y_hat))
#用Scikit-learn计算MAE
print("MAE:",metrics.mean_absolute_error(y_test,y_hat))
#用Scikit-learn计算MSE
print("MSE:",metrics.mean_squared_error(y_test,y_hat))
#用Scikit-learn计算RMSE
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test,y_hat)))
