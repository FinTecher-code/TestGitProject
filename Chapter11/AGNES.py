import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('../data/11_beverage.csv')
#取df的2列为X变量
X = df.iloc[:,0:2]
X = np.array(X.values)
#进行AGENS算法的训练，并输出结果
from sklearn.cluster import AgglomerativeClustering
#设置聚类结果的类簇
n_clusters = 4
#设定算法为AGNES算法，距离度量为最小距离
ward = AgglomerativeClustering(n_clusters,linkage='ward')
#进行聚类算法训练
print(ward.fit(X))
#输出相关聚类结果，并评估聚类效果
#输出每一样本的聚类的类簇标签
labels = ward.labels_
print('各类簇标签值：',labels)
from sklearn import metrics
#根据聚类结果预测每个X所对应的类簇
y_pred = ward.fit_predict(X)
#采用CH指标评估聚类结果
print(metrics.calinski_harabasz_score(X,y_pred))
#聚类结果及其各类簇中心点的可视化
markers = ['o','^','*','s']
colors = ['r','b','g','peru']
plt.figure(figsize=(7,5))
#画每个类簇的样本点
for c in range(n_clusters):
    #根据不同分类值c筛选X
    cluster = X[labels == c]
    #按照c的不同取值选取相应样本点、标记、颜色、画散点图
    plt.scatter(cluster[:,0],cluster[:,1],
                marker=markers[c],s=20,c=colors[c])
#设置坐标轴的label
plt.xlabel('juice')
plt.ylabel('sweet')
plt.show()
#进行最大距离度量方式的AGNES算法的训练，并输出结果
#设定算法为AGNES算法，距离度量为最大距离
complete = AgglomerativeClustering(n_clusters,linkage='complete')
print(complete.fit(X))
#输出每一样本的聚类的类簇标签
labels_com = complete.labels_
print('各类簇标签值：',labels_com)
from sklearn import metrics
#根据聚类结果预测每个X所对应的类簇
y_pred_com = complete.fit_predict(X)
#采用CH指标评估聚类结果
print(metrics.calinski_harabasz_score(X,y_pred_com))
#聚类结果及其各类簇中心点的可视化
markers = ['o','^','*','s']
colors = ['r','b','g','peru']
plt.figure(figsize=(7,5))
#画每个类簇的样本点
for c in range(n_clusters):
    #根据不同分类值c筛选X
    cluster = X[labels == c]
    #按照c的不同取值选取相应样本点、标记、颜色、画散点图
    plt.scatter(cluster[:,0],cluster[:,1],
                marker=markers[c],s=20,c=colors[c])
#设置坐标轴的label
plt.xlabel('juice')
plt.ylabel('sweet')
plt.show()