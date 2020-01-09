from sklearn import datasets
import numpy as np
iris = datasets.load_iris() #加载鸢（yuan）尾花数据集
print('iris.data:',iris.data) #打印分类样本的特征
print('iris.target:',iris.target) #打印数据集的目标值
iris_X = iris.data #将样本特征值设置为X
iris_y = iris.target #将目标值设置为Y
#数据预处理
from sklearn.model_selection import train_test_split #导入数据集分离模块
#划分为训练集和测试集数据
X_train,X_test,y_train,y_test = train_test_split(iris_X,iris_y,test_size=0.3)
print(y_test)
#打印X,Y数组形状
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
#逻辑回归分析
#线性模型中的逻辑回归
from sklearn.linear_model import LogisticRegression
#引入逻辑回归算法
lr = LogisticRegression(solver='lbfgs',multi_class='ovr')
#用逻辑回归算法拟合
lr.fit(X_train,y_train)
#打印逻辑回归求解参数
print(lr.coef_)
print("LogisticRegression:")
#对测试集的数据进行预测
y_pred = lr.predict(X_test)
#导入性能指标库
from sklearn import metrics
#用Scikit-learn 计算 MSE(均方差)
print("MSE:",metrics.mean_squared_error(y_test,y_pred))
#用Scikit-learn 计算 RMSE(均方根差)
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
#决策树算法
from sklearn.tree import DecisionTreeClassifier
#设定决策树算法
tree_clf = DecisionTreeClassifier(max_depth=2)
#拟合
tree_clf.fit(X_train,y_train)
#预测
y_pred_tree = tree_clf.predict(X_test)
print("DecisionTree:")
#用Scikit-learn计算MSE
print("MSE:",metrics.mean_squared_error(y_test,y_pred_tree))
#用Scikit-learn计算RMSE
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test,y_pred_tree)))
#K近邻算法
from sklearn.neighbors import KNeighborsClassifier
#设定K近邻算法
knn = KNeighborsClassifier()
#拟合
knn.fit(X_train,y_train)
#预测
y_pred_knn = tree_clf.predict(X_test)
print("KNN:")
#用Scikit-learn计算MSE
print("MSE:",metrics.mean_squared_error(y_test,y_pred_knn))
#用Scikit-learn计算RMSE
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test,y_pred_knn)))
#支持向量机算法
from sklearn import svm
#设定svm算法
svm = svm.SVC(gamma='auto')
#拟合
svm.fit(X_train,y_train)
y_pred_svm = tree_clf.predict(X_test)
print("SVM:")
#用Scikit-learn计算MSE
print("MSE:",metrics.mean_squared_error(y_test,y_pred_svm))
#用Scikit-learn计算RMSE
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test,y_pred_svm)))



