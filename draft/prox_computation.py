"""
Compute the proximal operator.
"""

import numpy as np
from breakpoints import G
from Rando_search import rando_search, norm_0


def u_i(eta, x, lambda_, i, eps=1e-7):
    if x[i] <= eps: return 0 

    if eta <= lambda_ / np.abs(x[i]):
        return 0
    elif eta >= (lambda_+ 1) / np.abs(x[i]):
        return 1
    else:
        return np.abs(x[i]) * eta - lambda_


def prox_lambdSk(x, lambda_, k=15):
    normx, x = norm_0(x, better_storage=False)
    print('norm', normx)
    if normx <= k:
        print("normx prox")
        return 1 / (lambda_ + 1) * x
    else:
        alpha_1 = np.abs(x[np.where(x >= 1e-7)])
        beta_1 = np.zeros_like(alpha_1)
        alpha_2 = np.zeros_like(alpha_1)
        beta_2 = np.ones_like(alpha_1)
        gamma = 1 / alpha_1
        #eta_tilde = rando_search(x, alpha_1, alpha_2,
        #               beta_1, beta_2, gamma, k, func = ?, best_store=False) #how ?
        # to compute eta_tilde : use decomp btm page 16
        # compute u_i(eta_tilde) for i = 1,...n  
        #u = # array of all the ui(eta_tilde)
        w = (x * u) / (lambda_ + u)
    return w

def fista(A, b, maxit, k=15, prox=prox_lambdSk):
    """ Based from https://gist.github.com/agramfort/ac52a57dc6551138e89b
    Modified with Python 3 for an arbitrary penalty
    """

    x = np.zeros(A.shape[1])
    t = 1
    z = x.copy()
    L = np.linalg.norm(A, ord=2) **2
    for j in range(maxit):
        print("################ Fista", j)
        xold = x.copy()
        z = z + A.T.dot(b - A.dot(z)) / L
        x = prox(z, k)
        t0 = t
        t = (1. + np.sqrt(1. + 4. * t ** 2)) / 2.
        z = x + ((t0 - 1.) / t) * (x - xold)
    return x

if __name__ == "__main__":
    n = 50
    x_true = np.array([3]*15 + [0]*25)
    sigma = 1
    w = np.random.rand(50)
    z1, z2, z3 = np.random.rand(n), np.random.rand(n), np.random.rand(n)
    Z1 = np.tile(z1.reshape(n, -1), 5)
    Z2 = np.tile(z2.reshape(n, -1), 5)
    Z3 = np.tile(z2.reshape(n, -1), 5)
    W1, W2, W3 = np.random.rand(n, 5), np.random.rand(n, 5), np.random.rand(n, 5)
    W = np.random.rand(n, 25)
    A = np.concatenate((Z1 + .01 * W1, Z2 + .01 * W2, Z3 + .01 * W3, W ),
                   axis=1)
    b = A @ x_true + sigma * w
    
    x_fin = fista(A, b, 50, 15)
    print(x_fin)