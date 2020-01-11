import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('../data/3_film.csv')
print(df.head())  # 展示前5行数据
# 选中data中的X变量
X = df.iloc[:, 1:4]  # 先行，后列
# 设定target为y
y = df.filmnum
from sklearn.model_selection import train_test_split
# 把X、y转化为数组形式，以便于计算
X = np.array(X.values)
y = np.array(y.values)
# 以25%的数据构建测试样本，剩余作为训练样本
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
from sklearn import linear_model
lasso = linear_model.Lasso(alpha=0.1) #设置lambda值
#使用训练数据进行参数求解
lasso.fit(X_train,y_train)
print('求解截距项为：',lasso.intercept_)
print('求解系数为：',lasso.coef_)
#对测试集的预测
y_hat_lasso = lasso.predict(X_test)
#打印前10个预测值
print(y_hat_lasso[0:9])
#对测试集相应变量实际值与预测值进行比较
plt.figure(figsize=(10,6))
t = np.arange(len(X_test))
#绘制y_test曲线
plt.plot(t,y_test,'r',linewidth = 2, label = 'y_test')
#绘制y_hat曲线
plt.plot(t,y_hat_lasso,'g',linewidth = 2 ,label ='y_hat_lasso')
plt.legend()
plt.show()
#对预测结果进行评价
from sklearn import metrics
from sklearn.metrics import r2_score
#拟合优度R^2的方法二
print("r2_score:",r2_score(y_test,y_hat_lasso))
#用Scikit-learn计算MAE
print("MAE:",metrics.mean_absolute_error(y_test,y_hat_lasso))
#用Scikit-learn计算MSE
print("MSE:",metrics.mean_squared_error(y_test,y_hat_lasso))
#用Scikit-learn计算RMSE
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test,y_hat_lasso)))
