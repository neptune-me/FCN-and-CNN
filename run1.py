import fc_net
import solver

import numpy as np
import matplotlib.pyplot as plt
import math
import random

def load_data():
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    i = -math.pi/2
    while i <= math.pi/2:
        X_train.append(i)
        y_train.append(math.sin(i))
        i += 0.03
    for i in range(2, 6):
        x = -math.pi/i
        X_test.append(x)
        y_test.append(math.sin(x))
    X_test.append(0)
    y_test.append(0)
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)



