import numpy as np
import code_for_hw3_part1 as lib

data = np.array([[2, 3,  4,  5]])
labels = np.array([[1, 1, -1, -1]])

theta, theta_0 = lib.perceptron(data, labels)
print(theta, theta_0)

print(np.sign(theta.T @ [[1, 6]] + theta_0))

def one_hot(x, k):
    result = np.zeros([k, 1])
    result[x - 1] = 1
    return result

print(one_hot(3, 7))

def one_hot2(x, k):
    result = np.zeros([k])
    result[x - 1] = 1
    return result

def encode(data):
    return np.array([one_hot2(x, 6) for x in data[0]]).T

data_one_hot = encode(data)
print(data_one_hot)
theta, theta_0 = lib.perceptron(data_one_hot, labels)
print(theta, theta_0)
print((theta.T @ encode([[1, 6]]) + theta_0) > 0)

s = encode([[1]])
n = encode([[6]])
print(s.T @ theta / np.linalg.norm(theta))
print(n.T @ theta / np.linalg.norm(theta))

data = np.array([[1, 2, 3, 4, 5, 6]])
labels = np.array([[1, 1, -1, -1, 1, 1]])
data_one_hot = encode(data)
print(lib.perceptron(data_one_hot, labels))