import matplotlib.pyplot as plt
import numpy as np
data = np.random.randint(1,11,4)
print(data)
labels = ['one','two','three','four'] #定义饼状图的标签，标签是列表
plt.pie(data,labels = labels) #绘制饼图
plt.axis('equal') #设置x、y轴刻度一致，这样饼图才能是圆的
plt.legend()
plt.show()