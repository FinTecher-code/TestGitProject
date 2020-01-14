import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
###一、数据预处理
##1、数据加载与预览
df = pd.read_csv('../data/13_house_train.csv')
print(df.head())
#快速查看数据的描述
print(df.info())
#数据统计信息预览
print(df.describe())    #数据的描述性统计信息
##2、处理缺失值
print(df[df.isnull().values == True])   #显示存在的缺失值的行列
#填充缺失值
df = df.fillna(df.mean()) #用该列平均值填充缺失值
#对其中的一个缺失值进行定位，查看填充后的结果
print(df.loc[95])   #定位第96行数据
##3、数据转换
df['built_date'] = pd.to_datetime(df['built_date']) #采用to_datetime转换为日期标准格式
print(df.head())
#日期数据转换成建筑年龄age并被替换
import datetime as dt
#当前的年份
now_year = dt.datetime.today().year
age = now_year-df.built_date.dt.year
#删除built_date列，方便后面计算
df.pop('built_date')
#将age列移动到第4列
df.insert(2,'age',age)
print(df.head())
#文本属性数据转换成数值型
print(df['floor'].unique()) #提取floor的取值
#将floor中的Low转换为0
df.loc[df['floor'] == 'Low','floor'] = 0
#将floor中的Medium转换为1
df.loc[df['floor'] == 'Medium','floor'] = 1
#将floor中的High转换为2
df.loc[df['floor'] == 'High','floor'] = 2
print(df.info())
###二、特征提取
##1、变量特征图表
#变量直方图
df.hist(xlabelsize=8,ylabelsize=8,layout=(3,5),figsize=(20,12)) #绘制直方图
plt.show()
#变量箱线图
df.plot(kind='box',subplots=True,layout=(3,5),
        sharex=False,sharey=False,fontsize=12,figsize=(20,12))  #绘制箱线图
plt.show()
##2、变量关联性分析
#生成相关系数矩阵
corr_matrix = df.corr()
#打印price与其他变量的相关系数
print(corr_matrix['price'].sort_values(ascending=False))
#目标变量与特征变量的散点图
plt.figure(figsize=(8,3))
#1行2列，第1个图
plt.subplot(121)
#绘制area与price的散点图
plt.scatter(df['price'],df['area'])
#1行2列，第2个图
plt.subplot(122)
#绘制pm25与price的散点图
plt.scatter(df['price'],df['pm25'])
plt.show()
#目标变量与特征变量的散点图
plt.figure(figsize=(8,3))
#1行2列，第1个图
plt.subplot(121)
#绘制age与price的散点图
plt.scatter(df['price'],df['age'])
#1行2列，第2个图
plt.subplot(122)
#绘制green_rate与price的散点图
plt.scatter(df['price'],df['green_rate'])
plt.show()
###三、数据建模
##1、对训练数据集的划分
#选取特征变量并划分数据集
col_n = ['area','crime_rate','pm25','traffic','shockproof',
         'school','age','floor'] #设置变量列表
#选取特征变量
X = df[col_n]
#设定price为y
y = df.price
from sklearn.model_selection import train_test_split
#把X、y转化为数组形式，以便于计算
X = np.array(X.values)
y = np.array(y.values)
#以25%的数据构建测试样本，剩余作为训练样本
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,
                                                 random_state=1)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
##2、采用不同算法的建模训练
#建模之岭回归
from sklearn import linear_model
#设置alpha值
ridge = linear_model.Ridge(alpha=0.1)
#使用训练数据进行参数求解
print(ridge.fit(X_train,y_train))
#对验证集的预测
y_hat = ridge.predict(X_test)
from sklearn import metrics
#用scikit-learn计算RMSE
print('RMSE_Ridge:',np.sqrt(metrics.mean_squared_error(y_test,y_hat)))
#建模之Lasso回归
from sklearn import linear_model
lasso = linear_model.Lasso(alpha=0.1)
#使用训练数据进行参数求解
print(lasso.fit(X_train,y_train))
#对验证集的预测
y_hat_lasso = lasso.predict(X_test)
#用scikit-learn计算RMSE
print('RMSE_losso:',np.sqrt(metrics.mean_squared_error(y_test,y_hat_lasso)))
#建模之支持向量机回归
from sklearn.svm import SVR
#使用线性核函数配置的支持向量机进行回归训练并预测
linear_svr = SVR(kernel='linear')
#训练模型
print(linear_svr.fit(X_train,y_train))
#对验证集的预测
y_hat_svr = linear_svr.predict(X_test)
#用scikit-learn计算RMSE
print('RMSE_svr:',np.sqrt(metrics.mean_squared_error(y_test,y_hat_svr)))
#建模之随机森林回归
from sklearn.ensemble import RandomForestRegressor
#使用随机森林进行回归训练并预测
rf = RandomForestRegressor(random_state=200,max_features=0.3,n_estimators=100)
#训练模型
print(rf.fit(X_train,y_train))
#对验证集的预测
y_hat_rf = rf.predict(X_test)
#用scikit-learn计算RMSE
print('RMSE_rf:',np.sqrt(metrics.mean_squared_error(y_test,y_hat_rf)))
##3、参数调优
#SVR参数调优
#设置惩罚项C的等差参数序列
alphas_svr = np.linspace(0.1,1.2,20)
#设置RMSE列表
rmse_svr = []
for c in alphas_svr:
    #设定模型为SVR
    model = SVR(kernel='linear',C=c)
    #使用训练数据进行参数求解
    model.fit(X_train,y_train)
    #预测划分训练集
    y_hat = model.predict(X_test)
    #将得到的均方误差结果加入RMSE列表中
    rmse_svr.append(np.sqrt(metrics.mean_squared_error(y_test,y_hat)))
#绘制不同C取值的RMSE结果
plt.plot(alphas_svr,rmse_svr)
#添加标题
plt.title('Cross Validation Score with Model SVR')
#添加x轴标签
plt.xlabel('alpha')
#添加y轴标签
plt.ylabel('rmse')
plt.show()
#Lasso参数调优
#设置惩罚项alpha的参数序列
alphas_lasso = np.linspace(-0.1,0.1,20)
#设置RMSE列表
rmse_lasso =[]
for alpha in alphas_lasso:
    #设定模型为Lasso
    model = linear_model.Lasso(alpha)
    #使用训练数据进行参数求解
    model.fit(X_train,y_train)
    #预测划分训练集
    y_hat = model.predict(X_test)
    #将得到的均方误差结果加入RMSE列表中
    rmse_lasso.append(np.sqrt(metrics.mean_squared_error(y_test,y_hat)))
#绘制不同c取值的RMSE结果
plt.plot(alphas_lasso,rmse_lasso)
#添加标题
plt.title('Cross Validation Score with Model Lasso')
plt.xlabel('alpha')
plt.ylabel('rmse')
plt.show()
##结论c=1.0的SVR所得的RMSE最小，所以该模型为最好的模型
###四、预测与提交结果
#预测house_test数据并提交结果
df_test = pd.read_csv('../data/13_house_test.csv')
print(df_test.head())
#预览house_test数据信息
print(df_test.info())   #快速查看数据的描述
#数据转换
df_test['built_date'] = pd.to_datetime(df_test['built_date'])
age = now_year - df_test.built_date.dt.year
#删除built_date列，方便后面计算
df_test.pop('built_date')
#将age列移动到第4列
df_test.insert(2,'age',age)
#将floor中的Low转换为0
df_test.loc[df_test['floor'] == 'Low','floor'] = 0
#将floor中的Medium转换为1
df_test.loc[df_test['floor'] == 'Medium','floor'] = 1
#将floor中的High转换为2
df_test.loc[df_test['floor'] == 'High','floor'] = 2
print(df_test.head())
#house_test数据的预测与结果的提交
#选取特征变量
testX = df_test[col_n]
#使用线性核函数配置的支持向量机进行回归训练并预测
svr_test = SVR(kernel='linear',C = 1.0)
svr_test.fit(X,y)
#对测试集进行预测
testy_svr = svr_test.predict(testX)
#将预测的数据和之前的合并
submit = pd.read_csv('../data/13_pred_test.csv')
submit['price'] = testy_svr
print(submit.head())