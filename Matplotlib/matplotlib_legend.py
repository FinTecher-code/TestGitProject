import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-np.pi,np.pi,0.1)
y = np.sin(x)
sin,cos = np.sin(x),np.cos(x)
plt.plot(x,sin,label='sin')
plt.plot(x,cos,color = 'red',linewidth = 2.0,linestyle = ':',label= 'cos')
plt.grid(True)
plt.legend()
plt.show()
plt.xlim(-4,4) #设置横坐标的范围
plt.ylim(-1.5,1.5) #设置纵坐标的范围
plt.xlabel("x") #横轴标识
plt.ylabel("y") #纵轴标识
plt.title("sinx plot") #设定图形的标题
plt.plot(x,y)
plt.show()
