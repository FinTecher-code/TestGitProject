import numpy as np
#数组的创建
array = np.array([1,2,3,4])
array2 = np.array([[1,2,3,4],[4,5,6,7],[7,8,9,10]])
print(array)
print(array2)
print('数组维度为：',array2.shape)
print('数组的数据类型为：',array2.dtype)
print('数组元素个数为：',array2.size)
print('使用arange函数创建的数组为：',np.arange(10))
print('使用linespace函数创建的数组为：',np.linspace(0,1,5))
print('使用zeros函数创建的数组为：',np.zeros((2,3)))
print('使用eys函数创建的数组为：',np.eye(3))
print('生成随机数组为：',np.random.random(10))

