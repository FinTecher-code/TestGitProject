import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-np.pi,np.pi,0.1) #创建数组
y = np.sin(x)
plt.plot(x,y)
plt.show()
#设置绿色、线条宽度、线条样式
plt.plot(x,y,color = 'green',linewidth = 2.0,linestyle = '-.')
plt.show()