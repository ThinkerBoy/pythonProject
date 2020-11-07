#机器学习线性代数基础书中代码

import numpy as np

a = np.array([1,2,3,4])
#一维数组无法直接转置
print(a)
print(a.T)
print(a.transpose())

A_t = a[:,np.newaxis]
print(A_t)
print(A_t.shape)

A = np.array([[1,2,3,4]])
print(A.T)

#加法

u = np.array([[1,2,3]]).T
v = np.array([[4,5,6]]).T

print(u+v)

u = np.array([[1,2,3]]).T
print(3*u)
#此时不能用U、v做向量的内积运算，矩阵不能应用于内积计算

u = np.array([1,2,3])
v = np.array([5,6,7])
print(np.dot(u,v))

print(np.cross(u,v))