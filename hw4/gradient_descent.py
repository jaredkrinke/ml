import numpy as np

# Gradient descent
def gd(f, df, x0, step_size_fn, max_iter):
    x = x0.copy()
    xs = []
    fs = []
    for i in range(0, max_iter):
        xs.append(x.copy())
        fs.append(f(x))
        x -= df(x) * step_size_fn(i)
    return x, fs, xs

# Numerical gradient
def num_grad(f, delta=0.001):
    def df(x):
        gradient = np.zeros(x.shape)
        for i in range(x.shape[0]):
            d = np.zeros(x.shape)
            d[i, 0] = delta
            gradient[i, 0] = (f(x + d) - f(x - d)) / (2*delta)
        return gradient
    return df

def minimize(f, x0, step_size_fn, max_iter):
    return gd(f, num_grad(f), x0, step_size_fn, max_iter)

# Support vector machine objective
def hinge(v):
    return np.where(v < 1, 1 - v, 0)

def hinge_loss(x, y, th, th0):
    return hinge(y * (th.T @ x + th0))

def svm_obj(x, y, th, th0, lam):
    return 1 / x.shape[1] * np.sum(hinge_loss(x, y, th, th0)) + lam * np.linalg.norm(th) ** 2

# SVM gradient
def d_hinge(v):
    return np.where(v < 1, -1, 0)

def d_hinge_loss_th(x, y, th, th0):
    return d_hinge(y * (th.T @ x + th0)) * x * y

def d_hinge_loss_th0(x, y, th, th0):
    return d_hinge(y * (th.T @ x + th0)) * y

def d_svm_obj_th(x, y, th, th0, lam):
    return np.mean(d_hinge_loss_th(x, y, th, th0), axis=1, keepdims=True) + 2 * lam * th

def d_svm_obj_th0(x, y, th, th0, lam):
    return np.mean(d_hinge_loss_th0(x, y, th, th0), axis=1, keepdims=True)

def svm_obj_grad(x, y, th, th0, lam):
    return np.vstack((d_svm_obj_th(x, y, th, th0, lam), d_svm_obj_th0(x, y, th, th0, lam)))

# SVM optimization (analytical)
def batch_svm_min(data, labels, lam):
    d = data.shape[0]
    theta = np.zeros([d, 1])
    theta_0 = 0
    x0 = np.vstack((theta, [[theta_0]]))

    extract_theta = lambda x: x[:-1]
    extract_theta_0 = lambda x: x[-1:]

    def f(x): return svm_obj(data, labels, extract_theta(x), extract_theta_0(x), lam)
    def df(x): return svm_obj_grad(data, labels, extract_theta(x), extract_theta_0(x), lam)
    def svm_min_step_size_fn(i): return 2/(i+1)**0.5
    
    return gd(f, df, x0, svm_min_step_size_fn, 10)

# SVG optimization (numerical)
def batch_svm_min_numerical(data, labels, lam):
    def f(x): return svm_obj(data, labels, x[:-1], x[-1:], lam)
    def svm_min_step_size_fn(i): return 2/(i+1)**0.5
    return minimize(f, np.zeros([data.shape[0] + 1, 1]), svm_min_step_size_fn, 10)
