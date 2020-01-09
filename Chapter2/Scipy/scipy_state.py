import scipy.stats as stats
import numpy as np
#产生一些具有特定分布的随机数
x = stats.uniform.rvs(size = 20)
print(x)
#创建正态分布与泊松分布的数组
y = stats.norm.rvs(size = 20,loc = 0,scale = 1)  #产生了20个服从[0,1]正态分布的随机数
z = stats.poisson.rvs(0.6,loc = 0,size = 20)  #产生poisson分布
print(y)
print(z)
#T检验
a = np.random.normal(0,1,size=100) #生成均值为0，标准差为1的100个正态分布随机数
b = np.random.normal(1,1,size=10) #生成均值为1，标准差为1的10个正态分布随机数
print(stats.ttest_ind(a,b)) #T检验

