import numpy as np

# Given
data = np.array([[1, 2, 1, 2, 10, 10.3, 10.5, 10.7],
                 [1, 1, 2, 2,  2,  2,  2, 2]])
labels = np.array([[-1, -1, 1, 1, 1, 1, 1, 1]])
blue_th = np.array([[0, 1]]).T
blue_th0 = -1.5
red_th = np.array([[1, 0]]).T
red_th0 = -2.5

def margin(x, y, theta, theta_0):
    return y * (theta.T @ x + theta_0) / np.linalg.norm(theta)

def margin_sum(data, labels, theta, theta_0):
    return np.sum(margin(data, labels, theta, theta_0), axis=1)

print(margin_sum(data, labels, red_th, red_th0))
print(margin_sum(data, labels, blue_th, blue_th0))

# More
data = np.array([[1.1, 1, 4],[3.1, 1, 2]])
labels = np.array([[1, -1, -1]])
th = np.array([[1, 1]]).T
th0 = -4

def hinge_loss(gamma, gamma_ref):
    if gamma < gamma_ref:
        return 1 - gamma / gamma_ref
    else:
        return 0

for x in margin(data, labels, th, th0)[0]:
    print(hinge_loss(x, 2**0.5/2))