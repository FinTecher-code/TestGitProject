from sklearn.linear_model.logistic import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('../data/5_logisitic_admit.csv')
#在df插入全为1的一列
df.insert(1,'Ones',1)
#展示df列表前10行
print(df.head(10))
#把admit为1的数据筛选出来形成单独的数据集
positive = df[df['admit']==1]
#把admit为0的数据筛选出来形成单独的数据集
negative = df[df['admit']==0]
#创建子图，大小为8*5
fig,ax = plt.subplots(figsize = (8,5))
#构建positive的散点图，设置散点形状为o
ax.scatter(positive['gre'],positive['gpa'],s=30,
           c='b',marker='o',label = 'admit')
#构建negative的散点图，设置散点形状为x
ax.scatter(negative['gre'],negative['gpa'],s=30,
           c='b',marker='x',label = 'not admit')
#设置图例
ax.legend()
#设置x轴标签
ax.set_xlabel('gre')
#设置y轴标签
ax.set_ylabel('gpa')
plt.show()
#梯度下降法求解参数
X = df.iloc[:,1:4]  #取df的后3列为X变量
y = df['admit']
#把X、y转化为数组形式，以便于计算
X = np.array(X.values)
y = np.array(y.values)
m,n = np.shape(X)
#导入sklearn库中的逻辑回归包
#设置算法为逻辑回归
lf = LogisticRegression(solver='liblinear',max_iter=1000)
#用逻辑回归拟合
print(lf.fit(X,y))
#基于sklearn得到参数值
print(lf.coef_)
#预测
pred_sk = lf.predict(X)
correct = [1 if ((a==1 and b==1) or (a==0 and b==0)) else
           0 for (a,b) in zip(pred_sk,y)]
accuracy = (sum(map(int,correct)) % len(correct))
print('accuracy={:.2f}%'.format(100*accuracy/m))
print(pred_sk)
lf.score(X,y)