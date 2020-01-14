import matplotlib.pyplot as plt
import numpy as np
#创建x变量
x = [0.69,-1.31,0.39,0.09,1.29,0.49,0.19,-0.81,-0.31,-0.71]
#创建y变量
y = [0.49,-1.21,0.99,0.29,1.09,0.79,-0.31,-0.81,-0.31,-1.01]
#创建z变量
z = [0,0,0,0,0,1,1,1,1,1]
#两变量列合并，形成10*2维的变量X
X = np.c_[x,y]
print(X)
#进行LDA训练，并输出结果
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#设置LDA降维参数，并将降维后的维度设为1
lda = LinearDiscriminantAnalysis(n_components=1)
#进行降维训练
print(lda.fit(X,z))
#输出相关的LDA训练结果
X_new = lda.transform(X)
print('降维后变量：',X_new)
print('权重向量：',lda.coef_)
#输出其他结果
print('每个类别的均值向量：',lda.means_)
print('整体样本的均值向量：',lda.xbar_)
plt.xlim(-2,2) #设置横坐标的范围
plt.ylim(-1.5,2) #设置纵坐标的范围
Z = np.c_[x,y,z]
# print(Z)
# print(len(Z))
# print(Z[0][2],Z[7][2])
for i in range(len(Z)):
    if Z[i][2] == 0.0:
        plt.scatter(Z[i][0],Z[i][1],marker='o',c='black',s=50)
    else:
        plt.scatter(Z[i][0],Z[i][1],marker='^',c='b',s=50)
#获取坐标轴的位置
ax = plt.gca()
#将左边框放到x=0位置，将下边框放到y=0位置
ax.spines['left'].set_position(("data",0))
ax.spines['bottom'].set_position(("data",0))
plt.show()