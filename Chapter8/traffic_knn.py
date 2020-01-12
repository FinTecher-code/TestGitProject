import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
df = pd.read_csv('../data/7_traffic.csv')
print(df.head())
X = df.iloc[:,0:6]
y = df['traffic']
#将数据集划分为训练集与测试集
X = np.array(X.values)
y = np.array(y.values)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=2)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
#K近邻算法训练，并输出结果
from sklearn import neighbors
#定义一个knn算法分类器
knn = neighbors.KNeighborsClassifier(n_neighbors=5,weights='distance')
print(knn.fit(X_train,y_train))
y_pred_knn = knn.predict(X_test)
print(y_pred_knn)
print(accuracy_score(y_test,y_pred_knn))
print(confusion_matrix(y_true=y_test,y_pred=y_pred_knn))
#定义一个knn1算法分类器
knn1 = neighbors.KNeighborsClassifier(n_neighbors=15,weights='distance')
print(knn1.fit(X_train,y_train))
y_pred_knn1 = knn1.predict(X_test)
print(y_pred_knn1)
print(accuracy_score(y_test,y_pred_knn1))
print(confusion_matrix(y_true=y_test,y_pred=y_pred_knn1))
#以余弦相似性为相似度衡量标准进行K近邻训练
knn2 = neighbors.KNeighborsClassifier(n_neighbors=15,
                                      metric='cosine',weights='distance')
print(knn2.fit(X_train,y_train))
y_pred_knn2 = knn2.predict(X_test)
print(y_pred_knn2)
print(accuracy_score(y_test,y_pred_knn2))
print(confusion_matrix(y_true=y_test,y_pred=y_pred_knn2))