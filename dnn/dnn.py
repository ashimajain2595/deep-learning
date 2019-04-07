import numpy as np
import matplotlib.pyplot as plt

from dnn_utils import *


def initialise_parameters(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


def linear_forward_activation(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)
    elif activation == 'relu':
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X, parameters):
    caches = []
    L = len(parameters) // 2
    A = X
    for l in range(1, L):
        A_prev = A
        A, cache = linear_forward_activation(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],
                                             activation='relu')
        caches.append(cache)
    AL, cache = linear_forward_activation(A, parameters['W' + str(L)], parameters['b' + str(L)], activation='sigmoid')
    caches.append(cache)
    return AL, caches


def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL))) / m
    cost = np.squeeze(cost)
    return cost


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ) / m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def linear_backward_activation(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
    elif activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = Y.shape[1]
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    cache = caches[L - 1]
    grads['dA' + str(L - 1)], grads['dW' + str(L)], grads['db' + str(L)] = linear_backward_activation(dAL, cache,
                                                                                                      activation='sigmoid')
    for l in reversed(range(1, L)):
        cache = caches[l - 1]
        grads['dA' + str(l - 1)], grads['dW' + str(l)], grads['db' + str(l)] = linear_backward_activation(
            grads['dA' + str(l)], cache, activation='relu')
    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)]
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]
    return parameters


def L_layer_model(X, Y, layer_dims, learning_rate=0.0075, num_iterations=2500, print_cost=False):
    costs = []
    parameters = initialise_parameters(layer_dims)
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print('Cost after iteration %i : %f' % (i, cost))
            costs.append(cost)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title('Learning rate = ' + str(learning_rate))
    plt.show()
    return cost, parameters
