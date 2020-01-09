import pandas as pd
df = pd.DataFrame({'A':[1,2,3,4],'B':[5,6,7,8]})
print(df['A'])
print(df[0:3])
print(df.iloc[1:3,0:2]) #选择第2、3行，第1、2列数据
df['C'] = pd.Series([1,2])
print(df)  #缺失值NaN
#去掉缺失值
print(df.dropna(how='any'))