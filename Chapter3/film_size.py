import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('../data/3_film.csv')
x = df['filmsize']
y = df['filmnum']
#scatter绘制散点图函数,c='b'是颜色为蓝色
plt.scatter(x,y,c='b')
plt.xlabel("x")
plt.ylabel("y")
plt.show()