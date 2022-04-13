import numpy as np

# X : n x k
# Y : n
def ridge_analytic(X, Y, lam):
    (n, k) = X.shape
    xm = np.mean(X, axis = 0, keepdims = True)   # 1 x n
    ym = np.mean(Y)                              # 1 x 1
    Z = X - xm                                   # d x n
    T = Y - ym                                   # 1 x n
    th = np.linalg.solve(np.dot(Z.T, Z) + lam * np.identity(k), np.dot(Z.T, T))
    # th_0 account for the centering
    th_0 = (ym - np.dot(xm, th))                 # 1 x 1
    return th.reshape((k,1)), float(th_0)

def update_U(data, vs_from_u, x, k, lam):
    (u, b_u, v, b_v) = x
    for a in range(len(u)):
        if len(vs_from_u) > 0:
            X = []
            Y = []
            for (i, r) in vs_from_u[a]:
                X.append(v[i][:,0])
                Y.append(r - b_v[i])
            u[a], b_u[a] = ridge_analytic(np.array(X), np.array(Y), lam)
    return x

def update_V(data, us_from_v, x, k, lam):
    (u, b_u, v, b_v) = x
    for i in range(len(v)):
        if len(us_from_v) > 0:
            X = []
            Y = []
            for (a, r) in us_from_v[i]:
                X.append(u[a][:,0])
                Y.append(r - b_u[a])
            v[i], b_v[i] = ridge_analytic(np.array(X), np.array(Y), lam)
    return x
