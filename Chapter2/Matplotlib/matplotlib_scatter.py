import matplotlib.pyplot as plt
import numpy as np
#(-pi,pi)之间，间隔为0.1的数组
x = np.arange(-np.pi,np.pi,0.1)
y = np.sin(x)
plt.scatter(x,y,color = 'r',marker= 'o') #绘制散点图
plt.show()