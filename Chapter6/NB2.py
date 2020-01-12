from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
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
MNB = MultinomialNB(alpha=1.0)
MNB.fit(X_train,y_train)
#打印各类标记的平滑先验概率
print(MNB.class_log_prior_)
#将多项式朴素贝叶斯解释的class_log_prior_映射为线性模型，其值和class_log_prior_相同
print(MNB.intercept_)
#根据求出的参数对测试集进行预测
y_pred_MNB = MNB.predict(X_test)
print(accuracy_score(y_test,y_pred_MNB))
print(confusion_matrix(y_true=y_test,y_pred = y_pred_MNB))  #计算混淆矩阵

