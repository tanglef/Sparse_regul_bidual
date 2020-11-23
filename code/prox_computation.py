"""
Compute the proximal operator.
"""

import numpy as np
from Rando_search import norm_0, dichoto_h, u_i


def prox_lambdSk(x, lambda_, k=15, eps=1e-7):
    """Param k is supposed to be close to the treu l_0 norm of x.

    The comments in the function introduce the random
    search algorithm even if the bisector method
    is used here.
    """
    normx, x = norm_0(x, better_storage=False, eps=eps)
    if normx <= k:
        return 1 / (lambda_ + 1) * x
    else:
        # alpha_1 = np.abs(x[np.where(x >= 1e-7)])
        # beta_1 = np.zeros_like(alpha_1)
        # alpha_2 = np.zeros_like(alpha_1)
        # beta_2 = np.ones_like(alpha_1)
        # gamma = 1 / alpha_1
        # random search
        # to compute eta_tilde : use decomp btm page 16
        eta_tilde = dichoto_h(x, lambda_, k, eps)
        u = np.array([u_i(eta_tilde, x, lambda_, i) for i in range(len(x))])
        w = (x * u) / (lambda_ + u)
    return w


def fista(A, b, lambda_, maxit, k=15, prox=prox_lambdSk, eps=1e-7):
    """FISTA algorithm.

    Implem. based on https://gist.github.com/agramfort/ac52a57dc6551138e89b
    Modified with Python 3 for an arbitrary penalty.
    """
    x = np.zeros(A.shape[1])
    t = 1
    z = x.copy()
    L = np.linalg.norm(A, ord=2)**2
    for j in range(maxit):
        xold = x.copy()
        z = z + A.T.dot(b - A.dot(z)) / L
        x = prox(z, lambda_, k, eps)
        t0 = t
        t = (1. + np.sqrt(1. + 4. * t ** 2)) / 2.
        z = x + ((t0 - 1.) / t) * (x - xold)
    return x

if __name__ == "__main__":
    n = 50
    x_true = np.array([3]*15 + [0] * 25)
    sigma = 1
    w = np.random.rand(50)
    z1, z2, z3 = np.random.rand(n), np.random.rand(n), np.random.rand(n)
    Z1 = np.tile(z1.reshape(n, -1), 5)
    Z2 = np.tile(z2.reshape(n, -1), 5)
    Z3 = np.tile(z2.reshape(n, -1), 5)
    W1, W2, W3 = np.random.rand(n, 5), np.random.rand(n, 5), np.random.rand(n, 5)
    W = np.random.rand(n, 25)
    A = np.concatenate((Z1 + .01 * W1, Z2 + .01 * W2, Z3 + .01 * W3, W),
                       axis=1)
    b = A @ x_true + sigma * w

    x_fin = fista(A, b, .04, 500, 15)
    print(x_fin)