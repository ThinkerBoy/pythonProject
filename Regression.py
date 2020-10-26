import numpy as np
import matplotlib.pyplot as plt


train = np.loadtxt('housePrices.txt',delimiter=',',skiprows=1)

train_x = train[:,0]
train_y = train[:,1]

train_x_mean =train_x.mean()

train_x_std = train_x.std()

train_x = (train_x - train_x_mean) / train_x_std
plt.plot(train_x,train_y,'o')
plt.show()