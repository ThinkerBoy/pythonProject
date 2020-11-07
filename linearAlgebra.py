import numpy as np

a = np.array([1,2,3,4])
print(a)
#一维转置无效
print(a.T)
#转置
a_t= a[:,np.newaxis]

print(a_t)