"""
Try to implement the algorithm described for the randomized search.
"""

import numpy as np
import time


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
        x_new = x[idx]
        return len(idx), x_new
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
    ans = abs_ * eta if eta <= 1 / abs_ else 1
    return ans


def F(eta, alpha_t, beta_t, omega, x):  # eta is gamma_p in the algo
    res_Fj = [F_j(eta, x, j) for j in omega]
    sum_ = np.sum(res_Fj)
    return alpha_t * eta + beta_t + sum_


def u_i(eta, x, lambda_, i, eps=1e-7):
    if x[i] <= eps:
        return 0

    if eta <= lambda_ / np.abs(x[i]):
        return 0
    elif eta >= (lambda_ + 1) / np.abs(x[i]):
        return 1
    else:
        return np.abs(x[i]) * eta - lambda_


def h_eta(eta, x, k, lambda_):
    sum_ = np.sum([u_i(eta, x, lambda_, i) for i in range(len(x))])
    return sum_ - k


def dichoto_h(x, lambda_, k, eps):
    _, x_sp = norm_0(x, better_storage=True)
    binf = lambda_ / np.max(np.abs(x_sp))
    bsup = (lambda_ + 1) / np.abs(np.min(x_sp))
    bridge = bsup - binf
    while np.abs(bridge) > eps:
        eta = (binf + bsup) / 2
        if h_eta(eta, x, k, lambda_) > 0:
            bsup = eta
        else:
            binf = eta
        bridge = bsup - binf
    return eta


def rando_search(x, alpha_1, alpha_2, beta_1, beta_2, gamma, delta, func=F, best_store=True):
    a_tilde = 0
    b_tilde = - delta
    norm0, x = norm_0(x, better_storage=best_store)
    omega = np.arange(norm0)

    while(len(omega) != 0):
        p = np.random.choice(omega)
        F_gamma_p = func(gamma[p], a_tilde, b_tilde, omega, x)
        if np.isclose(F_gamma_p, 0):
            return gamma[p]
        elif F_gamma_p > 0:
            idx_in = np.where(gamma < gamma[p])
            idx_out = np.where(gamma >= gamma[p])
            choice = [idx_in[0][i] in omega for i in range(len(idx_in[0]))]
            A = omega[np.isin(omega, idx_in[0][choice])]
            a_tilde += np.sum(alpha_1[idx_out])
            b_tilde += np.sum(beta_1[idx_out])
            omega = A
        else:
            idx_in = np.where(gamma > gamma[p])
            idx_out = np.where(gamma <= gamma[p])
            choice = [idx_in[0][i] in omega for i in range(len(idx_in[0]))]
            A = omega[np.isin(omega, idx_in[0][choice])]
            a_tilde += np.sum(alpha_2[idx_out])
            b_tilde += np.sum(beta_2[idx_out])
            omega = A
    return - b_tilde / a_tilde

if __name__ == '__main__':
    # show that the random search works on an example
    n = 50
    x_true = np.array([3]*15 + [0]*25)
    
    alpha_1 = np.abs(x_true[np.where(x_true >= 1e-7)])
    beta_1 = np.zeros_like(alpha_1)
    alpha_2 = np.zeros_like(alpha_1)
    beta_2 = np.ones_like(alpha_1)
    gamma = 1 / alpha_1
    delta = 15

    eta_t = rando_search(x_true, alpha_1, alpha_2, beta_1, beta_2, gamma, delta, func=F, best_store=True)
    print(eta_t)
    print(np.sum([min(np.abs(x_true[i]) * eta_t, 1) for i in range(len(x_true)) ]) - 15)