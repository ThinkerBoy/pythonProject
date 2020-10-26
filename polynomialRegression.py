import numpy as np
import matplotlib.pyplot as plt


#加载数据
train = np.loadtxt('click.csv',delimiter=',',skiprows=1)

train_x = train[:,0]
train_y = train[:,1]

#损失函数
def E(x,y):
    return 1/2*np.sum((f(x)-y)**2)

#归一化

mu = train_x.mean()
sigma = train_x.std()

def standardize(x):
    return (x-mu)/sigma

train_z = standardize(train_x)
#初始化参数

theta = np.random.rand(3)

# 创建训练数据的矩阵

def to_matrix(x):

    return np.vstack([np.ones(x.shape[0]),x,x**2]).T

X = to_matrix(train_z)

#预测函数

def f(x):
    return np.dot(x,theta)

# 误差的差值

ETA = 1e-3
diff = 1

#重复学习

error = E(X,train_y)
count = 1
while diff > 1e-2:
    #更新参数
    theta = theta - ETA*np.dot(f(X)-train_y,X)

    current_error = E(X,train_y)
    diff = error -current_error
    error = current_error

    count += 1
    #log = '第{}次: theat0 = {:3f},theat1 = {:3f},差值 = {:4f}'
    #print(log.format(count, theat0, theat1, diff))

x = np.linspace(-3,3,100)
plt.plot(train_z,train_y,'o')
plt.plot(x,f(to_matrix(x)))
plt.show()


def MSE(x,y):
    return (1/x.shape[0])*np.sum((y-f(x))**2)

# 用随机值初始化参数

theta = np.random.rand(3)

# 均方误差的历史记录

errors = []

# 误差的差值

diff = 1

# 重复学习

errors.append(MSE(X,train_y))

while diff > 1e-2:
    theta = theta - ETA*np.dot((f(X)-train_y),X)
    errors.append(MSE(X,train_y))
    diff = errors[-2] - errors[-1]

#绘制误差变化图

x = np.arange(len(errors))

print(x)
plt.plot(x,errors)
plt.show()

#随机梯度下降算法

while diff > 1e-2:
    #为了调整训练数据的顺序，准备随机的序列
    p = np.random.permutation(X.shape[0])
    #随机取出训练数据，使用随机梯度下降更新参数
    for x, y  in zip(X[p,:],train_y[p]):
        theta = theta - ETA*(f(x) - y) * x
    errors.append(MSE(X,train_y))
    errors.append(MSE(X, train_y))
    diff = errors[-2] - errors[-1]

# 绘制误差变化图

x = np.arange(len(errors))

print(x)
plt.plot(x, errors)
plt.show()

x = np.linspace(-3,3,100)
plt.plot(train_z,train_y,'o')
plt.plot(x,f(to_matrix(x)))
plt.show()