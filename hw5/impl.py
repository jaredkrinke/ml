import numpy as np

def lin_reg(x, th, th0):
    return np.dot(th.T, x) + th0

def d_lin_reg_th(x, th, th0):
    return x

def d_square_loss_th(x, y, th, th0):
    return -2 * (y - lin_reg(x, th, th0)) * d_lin_reg_th(x, th, th0)

def d_mean_square_loss_th(x, y, th, th0):
    return np.mean(d_square_loss_th(x, y, th, th0), axis = 1, keepdims = True)

def d_lin_reg_th0(x, th, th0):
    return np.ones([1, x.shape[1]])

def d_square_loss_th0(x, y, th, th0):
    return -2 * (y - lin_reg(x, th, th0)) * d_lin_reg_th0(x, th, th0)

def d_mean_square_loss_th0(x, y, th, th0):
    return np.mean(d_square_loss_th0(x, y, th, th0), axis = 1, keepdims = True)

def d_ridge_obj_th(x, y, th, th0, lam):
    return d_mean_square_loss_th(x, y, th, th0) + 2 * lam * th

def d_ridge_obj_th0(x, y, th, th0, lam):
    return d_mean_square_loss_th0(x, y, th, th0)

def sgd(X, y, J, dJ, w0, step_size_fn, max_iter):
    w = w0
    fs = []
    ws = []
    for i in range(0, max_iter):
        index = np.random.randint(0, X.shape[1])
        sample = X[:,index:index + 1]
        label = y[:,index]
        ws.append(w.copy())
        fs.append(J(sample, label, w))
        if i < max_iter - 1: w -= step_size_fn(i) * dJ(sample, label, w)
    return w, fs, ws
