import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
boston = load_boston()
#打印boston包含的内容
print(boston.keys())
#打印data的变量名
print(boston.feature_names)\
#将data转换为DataFrame格式以方便展示
bos = pd.DataFrame(boston.data)
#选取的单变量为RM(每个房屋的房间数量)
print(bos[5].head())
#将target转换为DataFrame格式以方便展示
bos_target = pd.DataFrame(boston.target)
print(bos_target.head())
#绘制房价，每个房屋的房间数量的散点图
import matplotlib.font_manager as fm
#选取data中的RM变量
X = bos.iloc[:,5:6]
#选定target为y
y = bos_target
#定义自定义字体，文件名是系统中文字体
myfont = fm.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.scatter(X,y)
#将x、y轴标签设定文字为中文msyh格式
plt.xlabel(u'住宅平均房间数',fontproperties=myfont)
plt.ylabel(u'房屋价格',fontproperties=myfont)
#标题
plt.title(u'RM与MEDV的关系',fontproperties=myfont)
plt.show()
#数据划分
from sklearn.model_selection import train_test_split
#把X、y转化为数组形式，以便于计算
X = np.array(X.values)
y = np.array(y.values)
#以25%的数据构建测试样本，剩余作为训练样本
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
#线性回归预测
from sklearn.linear_model import LinearRegression
#设定回归算法
lr = LinearRegression()
#使用训练数据进行参数求解
lr.fit(X_train,y_train)
print('求解的截距项为：',lr.intercept_)
print('求解系数为：',lr.coef_)
#对测试集进行预测
y_hat = lr.predict(X_test)
print(y_hat[0:9])
#模型评价
#y_test 与 y_hat 的可视化
plt.figure(figsize=(10,6)) #设置图片尺寸
#创建t变量
t = np.arange(len(X_test))
#绘制y_test曲线
plt.plot(t,y_test,'r',linewidth = 2,label = 'y_test')
#绘制y_hat曲线
plt.plot(t,y_hat,'g',linewidth = 2,label = 'y_train')
#设置图例
plt.legend()
plt.show()
#采用评估指标拟合优度R^2、MAE、MSE、RMSE
from sklearn import metrics
from sklearn.metrics import r2_score
#拟合优度R2的输出方法一
print("r2:",lr.score(X_test,y_test))
#拟合优度R2的输出方法二
print("r2_score:",r2_score(y_test,y_hat))
#用Scikit-learn计算MAE
print("MAE:",metrics.mean_absolute_error(y_test,y_hat))
#用Scikit-learn计算MSE
print("MSE:",metrics.mean_squared_error(y_test,y_hat))
#用Scikit-learn计算RMSE
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test,y_hat)))
#采用最小二乘法比较MAE、MSE、RMSE
import math
#构建最小二乘回归函数
def linefit(x,y):
    #计算样本值
    N = len(x)
    #设置初始值
    sx,sy,sxx,syy,sxy = 0,0,0,0,0
    for i in range(0,N):
        sx += x[i] #计算xi的总和
        sy += y[i] #计算yi的总和
        sxx += x[i] * x[i] #计算xi平方的总和
        syy += y[i] * y[i] #计算yi平方的总和
        sxy += x[i] * y[i] #计算xi * yi的总和
    #求解系数a
    a = (sy * sx/N - sxy)/(sx * sx/N -sxx)
    b = (sy - a * sx) / N
    return a,b
#求解参数a、b
a,b = linefit(X_train,y_train)
#对测试集的y值进行预测
y_hat1 = a*X_test + b
#用Scikit-learn计算MAE
print("MAE:",metrics.mean_absolute_error(y_test,y_hat1))
#用Scikit-learn计算MSE
print("MSE:",metrics.mean_squared_error(y_test,y_hat1))
#用Scikit-learn计算RMSE
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test,y_hat1)))
