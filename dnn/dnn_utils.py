import numpy as np


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def relu(Z):
    A = max(0, Z)
    cache = Z
    return A, cache


def sigmoid_backward(dA, cache):
    Z = cache
    a, _ = sigmoid(Z)
    dZ = dA * a * (1 - a)
    return dZ


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ
