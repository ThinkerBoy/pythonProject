# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt



def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

sigmoid_inputs = np.arange(-10, 10, 0.1)
sigmoid_outputs = sigmoid(sigmoid_inputs)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Sigmoid Function Input :: {}".format(sigmoid_inputs))
    #print("Sigmoid Function Output :: {}".format(sigmoid_outputs))

    plt.plot(sigmoid_inputs, sigmoid_outputs)
    plt.xlabel("Sigmoid Inputs")
    plt.ylabel("Sigmoid Outputs")
    plt.show()
