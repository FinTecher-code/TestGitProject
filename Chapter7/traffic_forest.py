import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
df = pd.read_csv('../data/7_traffic.csv')
print(df.head())
#数据初步可视化
X = df.iloc[:,0:6]
y = df['traffic']
#调增直方图尺寸
X.hist(xlabelsize=12,ylabelsize=12,figsize=(18,12))
plt.show()
#目标变量的直方图
y.hist(xlabelsize=12,ylabelsize=12,figsize=(8,5))
plt.show()
#将数据集划分为训练集与测试集
X = np.array(X.values)
y = np.array(y.values)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=2)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
#进行ID3决策树算法训练，并进行预测
from sklearn import tree
tree_ID3 = tree.DecisionTreeClassifier(criterion='entropy')
tree_ID3 = tree_ID3.fit(X_train,y_train)
y_pred_ID3 = tree_ID3.predict(X_test)
print(accuracy_score(y_test,y_pred_ID3))
print(confusion_matrix(y_true=y_test,y_pred=y_pred_ID3))
#然后进行随机森林训练，并输出结果
from sklearn.ensemble import RandomForestClassifier
#定义一个随机森林分类器
clf = RandomForestClassifier(n_estimators=10,max_depth=None,
                             min_samples_split=2,oob_score=True,random_state=0)
clf.fit(X_train,y_train)
print(clf.oob_score_)
#预测测试集
y_pred_rf = clf.predict(X_test)
print(accuracy_score(y_test,y_pred_rf))
print(confusion_matrix(y_true=y_test,y_pred=y_pred_rf))
#用ExtraTreesClassifier包进行训练
from sklearn.ensemble import ExtraTreesClassifier
#定义一个极端森林分类器
clf_extra = ExtraTreesClassifier(n_estimators = 10,max_depth = None,
                                  min_samples_split = 2,random_state = 0)
clf_extra.fit(X_train,y_train)
#用Extra_Trees算法对测试集进行预测
y_pred_extra = clf_extra.predict(X_test)
print(accuracy_score(y_test,y_pred_extra))
print(confusion_matrix(y_true=y_test,y_pred=y_pred_extra))
