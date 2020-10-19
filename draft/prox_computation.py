"""
Compute the proximal operator.
"""

import numpy as np
from breakpoints import G
from Rando_search import Rando_search, norm_0


def u_i(eta, x, lambda_, i, eps=1e-7):
    if x[i] <= eps: return 0 

    if eta <= lambda_ / np.abs(x[i]):
        return 0
    elif eta >= (lambda_+ 1) / np.abs(x[i]):
        return 1
    else:
        return np.abs(x[i]) * eta - lambda_


def prox_lambdSk(x, lambda_, k):
    normx = norm_0(x, better_storage=False)
    if normx <= k:
        return 1 / (lambda_ + 1) * x
    else:
        eta_tilde = # randosearch for eta_tilde over h
        # to compute eta_tilde : use decomp btm page 16
        # compute u_i(eta_tilde) for i = 1,...n  
        u = # array of all the ui(eta_tilde)
        w = (x * u) / (lambda_ + u)

    return w
