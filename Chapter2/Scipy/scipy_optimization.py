from scipy import optimize
import numpy as np
#优化--->找最小值或者等式数值的解
def f(x):
    return x**2+10*np.sin(x)
x = np.arange(-5.5,0.2)
from matplotlib import pyplot as plt
plt.plot(x,f(x))
plt.show()
from scipy.optimize import fmin
print(fmin(f,0))