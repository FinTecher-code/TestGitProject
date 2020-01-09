from scipy import linalg
import numpy as np
arr1 = np.array([[1,2],[3,4]])
arr2 = np.array([[3,2],[6,4]])
#计算方阵的行列式
print(linalg.det(arr1))
print(linalg.det(arr2))
#计算方阵的逆
iarr = linalg.inv(arr1)
print(iarr)