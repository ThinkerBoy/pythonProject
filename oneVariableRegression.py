import numpy as np
import matplotlib.pyplot as plt



#加载数据
train = np.loadtxt('click.csv',delimiter=',',skiprows=1)

train_x = train[:,0]
train_y = train[:,1]



#plt.plot(train_x,train_y,'o')
#plt.show()


#函数定义
theat0 = np.random.rand()
theat1 = np.random.rand()

def f(x):
    return theat0+theat1*x
#损失函数
def E(x,y):
    return 1/2*np.sum((f(x)-y)**2)

#归一化

mu = train_x.mean()
sigma = train_x.std()

def standardize(x):
    return (x-mu)/sigma

train_z = standardize(train_x)

plt.plot(train_z,train_y,'o')
plt.show()

#学习率

ETA =1e-3
#误差的差值

diff = 1

#更新次数
count = 0
# 重复学习

error = E(train_z,train_y)

while diff > 1e-2:
    #更新结果保存到临时变量
    tmp0 = theat0 - ETA*np.sum(f(train_z)-train_y)
    tmp1 = theat1 - ETA*np.sum((f(train_z)-train_y)*train_z)


    #更新参数
    theat0 = tmp0
    theat1 = tmp1

    #计算与上一次的误差值

    current_error = E(train_z,train_y)
    diff =  error - current_error
    error = current_error
    #输出日志

    count += 1
    log = '第{}次: theat0 = {:3f},theat1 = {:3f},差值 = {:4f}'
    print(log.format(count,theat0,theat1,diff))

x = np.linspace(-3,3,100)

plt.plot(train_z,train_y,'o')
plt.plot(x,f(x))
plt.show()