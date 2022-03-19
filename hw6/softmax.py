import numpy as np
from math import exp

def soft_max(z):
    return np.array([exp(z_j)/sum([exp(z_k) for z_k in z[:,0]]) for z_j in z[:,0]])

def d_nll(w, x, y):
    a = soft_max(w.T @ x)
    return np.array([[x[k] @ (a[j] - y[j]) for j in range(w[k].shape[0])] for k in range(w.shape[0])])

w = np.array([[1, -1, -2], [-1, 2, 1]])
x = np.array([[1, 1]]).T
y = np.array([[0, 1, 0]]).T

print(d_nll(w, x, y))
