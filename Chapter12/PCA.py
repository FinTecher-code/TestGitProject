import matplotlib.pyplot as plt
import numpy as np
#创建x变量
x = [0.69,-1.31,0.39,0.09,1.29,0.49,0.19,-0.81,-0.31,-0.71]
#创建y变量
y = [0.49,-1.21,0.99,0.29,1.09,0.79,-0.31,-0.81,-0.31,-1.01]
#两变量列合并，形成10*2维的变量X
X = np.c_[x,y]
print(X)
#进行PCA训练，并输出结果
from sklearn.decomposition import PCA
#创建一个PCA对象，设定保留的特征数2
pca = PCA(n_components=2)
#进行PCA降维
print(pca.fit(X))
#输出相关PCA训练结果
print('特征值：',pca.explained_variance_)
print('特征值的贡献率：',pca.explained_variance_ratio_)
#保留主成分为1进行PCA训练，并输出结果
#创建一个PCA对象，设定保留的特征数1
pca_one = PCA(n_components=1)
print(pca_one.fit(X))   #进行PCA降维
print('特征值：',pca_one.explained_variance_)
print('特征值贡献率：',pca_one.explained_variance_ratio_)
#生成降维后的数据
X_new = pca_one.transform(X)
print(X_new)