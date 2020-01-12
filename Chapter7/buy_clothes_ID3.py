import os
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df = pd.read_csv('../data/7_buy.csv')
print(df)
#进行ID3决策树算法的分类
X = df.iloc[:,0:4]
y = df['buy']
#把X、y转化为数组形式，以便于计算
X = np.array(X.values)
y = np.array(y.values)
from sklearn import tree
#默认采用的是gini，即是cart算法，在这里通过entropy设置ID3算法
tree_ID3 = tree.DecisionTreeClassifier(criterion='entropy')
tree_ID3 = tree_ID3.fit(X,y)
print(tree_ID3)
#预测并评估效果
y_pred = tree_ID3.predict(X)
print(y_pred)
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
print(accuracy_score(y,y_pred))
print(confusion_matrix(y_true=y,y_pred=y_pred))
#生成决策树结构图
#得到所有列列标签并转化为list形式
feature_names = list(df.columns[:-1])
#创建目标变量的类别名称
target_names = ['0','1']
#导入pydotplus,即python写dot语言的接口
import pydotplus
from IPython.display import Image
#将对象写入内存中
dot_data = StringIO()
#生成决策树结构
tree.export_graphviz(tree_ID3,out_file=dot_data,feature_names=feature_names,
                     class_names=target_names,filled=True,rounded=True,
                     special_characters=True)
#生成决策树图形
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
os.environ["PATH"] += os.pathsep + 'E:/graphviz/bin'
#生成图形并展示出来
graph.write_pdf("tree_ID3.pdf")
