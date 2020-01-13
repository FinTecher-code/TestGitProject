import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('../data/11_beverage.csv')
print(df.head())
#样本数据转化并且进行可视化
X = df.iloc[:,0:2]  #取df的2列为X变量
X = np.array(X.values) #把X化为数组形式
plt.scatter(X[:,0],X[:,1],s=20,marker='o',c='b')
#设置坐标轴的label
plt.xlabel('juice')
plt.ylabel('sweet')
plt.show()
#进行K均值算法的训练
from sklearn.cluster import KMeans
#设置聚类结果的类簇
n_clusters = 3
kmean = KMeans(n_clusters)
print(kmean.fit(X))
#输出相关聚类结果，并评估
from sklearn import metrics
#根据聚类结果预测每个X所对应的类簇
y_pred = kmean.predict(X)
#采用CH指标评估聚类结果
print(metrics.calinski_harabasz_score(X,y_pred))
#输出每一样本的聚类的类簇标签
labels = kmean.labels_
#输出聚类的类簇中心点
centers = kmean.cluster_centers_
print('各类簇标签值：',labels)
print('各类簇中心：',centers)
#聚类结果及其各类簇中心点的可视化
#设置散点图标记列表
markers = ['o','^','*']
#设置散点图颜色列表
colors = ['r','b','g']
#设置图形大小
plt.figure(figsize=(7,5))
#画每个类簇的样本点
for c in range(n_clusters):
    cluster = X[labels == c]    #根据不同分类值c筛选X
    #按照c的不同取值选取相应样本点、标记、颜色，画散点图
    plt.scatter(cluster[:,0],cluster[:,1],marker=markers[c],s=20,c=colors[c])
#画出每个类簇中心点
plt.scatter(centers[:,0],centers[:,1],marker='o',c='black',alpha=0.9,s=50)
#设置坐标轴的label
plt.xlabel('juice')
plt.ylabel('sweet')
plt.show()

#设定k=4进行K均值算法的训练
n_clusters_four = 4
#设定算法为KMeans算法
kmean_four = KMeans(n_clusters_four)
print(kmean_four.fit(X))
#根据聚类结果预测每个X所对应的类簇
y_pred_four = kmean_four.predict(X)
#采用CH指标评估聚类结果
print(metrics.calinski_harabasz_score(X,y_pred_four))
#输出聚类中心点
labels_four = kmean_four.labels_  #输出每一样本的聚类的类簇标签
#输出聚类的类簇中心点
centers_four = kmean_four.cluster_centers_
print('各类簇中心：',centers_four)
#k = 4时聚类结果及其各类簇中心点的可视化
#设置散点图标记列表
markers = ['o','^','*','s']
#设置散点图颜色列表
colors = ['r','b','g','peru']
#设置图形大小
plt.figure(figsize=(7,5))
#画每个类簇的样本点
for c in range(n_clusters_four):
    cluster = X[labels_four== c]    #根据不同分类值c筛选X
    #按照c的不同取值选取相应样本点、标记、颜色，画散点图
    plt.scatter(cluster[:,0],cluster[:,1],marker=markers[c],s=20,c=colors[c])
#画出每个类簇中心点
plt.scatter(centers_four[:,0],centers_four[:,1],marker='o',c='black',alpha=0.9,s=50)
#设置坐标轴的label
plt.xlabel('juice')
plt.ylabel('sweet')
plt.show()
#设置k一定的取值范围，进行聚类并评价不同的聚类结果
from scipy.spatial.distance import cdist
#类簇的数量2~9
clusters = range(2,10)
#距离函数
distances_sum = []
for k in clusters:
    #对不同取值k进行训练
    kmeans_model = KMeans(n_clusters = k).fit(X)
    #计算各对象离各类簇中心的欧式距离，生成距离表
    distances_point = cdist(X,kmeans_model.cluster_centers_,'euclidean')
    #提取每个对象到其类簇中心的距离(该距离最短，所以用min函数),并相加
    distances_cluster = sum(np.min(distances_point,axis=1))
    #依次存入类簇数从2到9的距离结果
    distances_sum.append(distances_cluster)
#画出不同聚类结果下的距离总和
plt.plot(clusters,distances_sum,'bx-')
plt.xlabel('k')
plt.ylabel('distances')
plt.show()