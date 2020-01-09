import pandas as pd
# ./表示 当前目录路径
#../表示 上一级目录路径
df1 = pd.read_csv('../../data/2_apple.csv')
#head只对前5行进行查看
print(df1.head())
print(df1.tail(3))
print(df1.index)  #显示索引
print(df1.columns)  #显示列
print(df1.describe())
#对2_apple.csv 按某一列的值进行排序
df1 = df1.sort_values(by = 'apple')
print(df1.head())
#对df1进行转置
df1 = df1.T
print(df1.head())