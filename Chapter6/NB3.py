from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df = pd.read_csv('../data/6_credit.csv')
print(df.head())
#选取特征变量与响应变量，并进行数据划分
X = df.iloc[:,1:6]  #取df的前5列为X变量
y = df['credit']
#把X、y转化为数组形式，以便于计算
X = np.array(X.values)
y = np.array(y.values)
from sklearn.model_selection import train_test_split
#以25%的数据构建测试样本，剩余作为训练样本
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=1)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
BNB = BernoulliNB(alpha=1.0,binarize=2.0,fit_prior=True)
BNB.fit(X_train,y_train)
#类先验概率对数值，类先验概率等于各类的个数/类的总个数
print(BNB.class_log_prior_)
#指定类的各类特征概率(条件概率)对数值，返回形状为(n_classes,n_features)数组
print(BNB.feature_log_prob_)
#根据求出的参数对测试集进行预测
y_pred_BNB = BNB.predict(X_test)
print(accuracy_score(y_test,y_pred_BNB))
print(confusion_matrix(y_true=y_test,y_pred = y_pred_BNB))  #计算混淆矩阵

