import numpy as np
#导入读取mat文件的模块
from scipy.io import loadmat
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
data = loadmat('../data/10_digital.mat')
print(data)
#把X、y转化为数组形式，以便于计算
X = data['X']   #提取X变量
y = data['y']   #提取y变量
print(X.shape,y.shape)
#矩阵X的第一行第101列至第119列的数据
print(X[0,100:120])
#数据预处理-----对X进行标准化转换
from sklearn.neural_network import MLPClassifier #导入MLP分类器程序库
from sklearn.preprocessing import StandardScaler #导入标准化库
#标准化转换
scaler = StandardScaler()
#训练标准化对象
scaler.fit(X)
#转换数据集
X = scaler.transform(X)
#将数据集划分为训练集和测试集
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=2)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
#构建多层感知机来实现神经网络的训练
mlp = MLPClassifier(solver='adam',activation='tanh',alpha=1e-5,
                    hidden_layer_sizes=(50,),learning_rate_init=0.001,
                    max_iter=2000) #设置MLP算法
print(mlp.fit(X_train,y_train))
#展示训练结果，并对测试集进行预测
print('每层网络层系数矩阵维度：\n',[coef.shape for coef in mlp.coefs_])
y_pred = mlp.predict(X_test)
print('预测结果：',y_pred)
#列表中第i个元素代表i+1层的偏差向量
print(mlp.intercepts_)
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_true=y_test,y_pred=y_pred))