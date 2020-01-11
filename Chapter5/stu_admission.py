#构建sigmoid()函数与predict()函数
#方法一：构建梯度下降法求解
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#构建sigmoid函数
def sigmoid(z):
    return 1/(1+np.exp(-z)) #sigmoid函数形式
def predict(theta,X):
    #根据sigmoid函数预测admit的概率
    prob = sigmoid(X*theta.T)
    #根据admit的概率设定阈值，大于0.5计为1，否则为0
    return [1 if a >= 0.5 else 0 for a in prob]
#构建梯度下降gradientDescent()函数
def gradientDescent(X,y,theta,alpha,m,numIter):
    #矩阵转置
    XTrans = X.transpose()
    #在1-numIterations之间for循环
    for i in range(0,numIter):
        #将theta转化为矩阵
        '''numpy并不推荐使用matrix类型。
        主要是因为array才是numpy的标准类型，并且基本上各种函数都有队array类型的处理，
        而matrix只是一部分支持而已。'''
        theta = np.matrix(theta)
        #将预测值转化为数组
        pred = np.array(predict(theta,X))
        #预测值减去实际值
        loss = pred - y
        #计算梯度
        gradient = np.dot(XTrans,loss)/m
        #参数theta的计算，即更新法则
        theta = theta - alpha*gradient
    return theta
#预测
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
#设置训练样本值m,变量个数n
m,n = np.shape(X)
#初始化
theta = np.ones(n)
#检查X与y的行列数，是否一致
print(X.shape,theta.shape,y.shape)
#迭代次数
numIter = 1000
#学习率
alpha = 0.00001
#采用构造的gradientDescent求解theta
theta = gradientDescent(X,y,theta,alpha,m,numIter)
print(theta)
#预测并计算准确率
pred = predict(theta,X) #使用predict()函数来预测y
#将预测为1实际也为1，预测为0实际也为0的均标记为1
correct = [1 if ((a==1 and b==1) or (a==0 and b==0)) else
           0 for (a,b) in zip(pred,y)]
#采用加总correct值来计算预测对的个数
accuracy = (sum(map(int,correct)) % len(correct))
#打印预测准确率
print('accuracy={:.2f}%'.format(100*accuracy/m))
