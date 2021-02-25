import numpy as np

def one_hot(n, l):
    arr = np.zeros(l)
    arr[n] = 1
    return arr