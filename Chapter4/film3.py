import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df = pd.read_csv('../data/3_film.csv')
print(df.head())  #展示前5行数据
#选中data中的X变量
X = df.iloc[:,1:4] #先行，后列
#设定target为y
y = df.filmnum
from sklearn.model_selection import train_test_split
#把X、y转化为数组形式，以便于计算
X = np.array(X.values)
y = np.array(y.values)
#以25%的数据构建测试样本，剩余作为训练样本
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=1)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
#岭回归估计
from sklearn import linear_model
#设置lambda值
ridge = linear_model.Ridge(alpha=0.1)
ridge.fit(X_train,y_train)
#使用训练集数据进行参数求解
print('求解截距项：',ridge.intercept_)
print('求解系数为：',ridge.coef_)
#根据求出参数对测试集进行预测
y_hat = ridge.predict(X_test)
print(y_hat[0:9])
#对测试集相应变量实际值与预测值的比较
plt.figure(figsize=(10,6))
t = np.arange(len(X_test))
#绘制y_test曲线
plt.plot(t,y_test,'r',linewidth = 2,label = 'y_test')
#绘制y_hat曲线
plt.plot(t,y_hat,'g',linewidth = 2 ,label = 'y_train')
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


