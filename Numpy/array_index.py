import numpy as np
#数组的索引
arr = np.arange(10)
print('索引结果为：',arr[5])
print('索引结果为：',arr[3:5])
arr[2:4] = 100,101
print('索引结果为：',arr)
arr1 = np.array([[1,2,3,4,5],[4,5,6,7,8],[7,8,9,10,11]])
print('创建的二维数组为：',arr1)
print('索引结果为：',arr1[0,3:5])