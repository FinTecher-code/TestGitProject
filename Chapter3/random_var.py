import matplotlib.pyplot as plt
import numpy as np
#生成-10 ~ 10 之间的100个数据
x1 = np.random.randint(-10,10,100)
#生成y1
y1 = x1 ** 2 - x1 + np.random.randint(1,100,100)
plt.scatter(x1,y1,c='b')
plt.show()