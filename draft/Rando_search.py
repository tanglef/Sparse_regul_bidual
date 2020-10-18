"""
Try to implement the algorithm described for the randomized search.
"""

import numpy as np
import time


def fista(A, b, l, maxit, penalty): # TO DO : find the penalty and change it
    """ Based from https://gist.github.com/agramfort/ac52a57dc6551138e89b
    Modified with Python 3 for an arbitrary penalty
    """

    def soft_thresh(x, l):
        return np.sign(x) * np.maximum(np.abs(x) - l, 0.)

    x = np.zeros(A.shape[1])
    t = 1
    pobj = []
    z = x.copy()
    L = np.linalg.norm(A, ord=2) **2
    time0 = time.time()
    for _ in range(maxit):
        xold = x.copy()
        z = z + A.T.dot(b - A.dot(z)) / L
        x = soft_thresh(z, l / L)
        t0 = t
        t = (1. + np.sqrt(1. + 4. * t ** 2)) / 2.
        z = x + ((t0 - 1.) / t) * (x - xold)
        this_pobj = 0.5 * np.linalg.norm(A.dot(x) - b) ** 2 + l * penalty(x)
        pobj.append((time.time() - time0, this_pobj))

    times, pobj = map(np.array, zip(*pobj))
    return x, pobj, times



def norm_0(x, better_storage=False, eps=1e-7):
    """ Compute the 0-norm of a vector
    Args: 
    -------
        x: a vector to compute the L0 norm
        better_storage: should we store only the non-zero inplace ?
        eps: precision to decide that the component is zero

    Output:
    -------
        the 0 norm
        the new x (different than the original if better_storage is True)
    """ 
    idx = np.where(x > eps)[0]
    if better_storage:
        x = x[idx]
    return len(idx), x


def F_j(eta, x, j):
    """ Compute F_j(eta)
    Args:
    --------
        eta: point in which to evaluate F_j
        x: vector of 0-norm q, must be of size q.
        j: index in [[1,q]]
    """

    abs_ = np.abs(x[j]) 
    ans =  abs_ * eta if eta <= 1 / abs_ else 1
    return ans

def F(eta, alpha_t, beta_t, omega, x): # eta is gamma_p in the algo
    res_Fj = [F_j(eta, x, j) for j in omega]
    sum_ = np.sum(res_Fj)
    return alpha_t * eta + beta_t + sum_


def rando_search(x, alpha_1, alpha_2, beta_1, beta_2, gamma, delta):
    a_tilde = 0
    b_tilde = - delta
    norm0, x = norm_0(x, better_storage=True)
    omega = range(norm0)

    while(len(omega) != 0):
        p = np.random.choice(omega)
        F_gamma_p = F(gamma[p], a_tilde, b_tilde, omega, x)
        # TO DO





    