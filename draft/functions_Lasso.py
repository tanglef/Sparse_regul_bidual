"""Created on Wed Jun 25 09:39:45 2014.

Source to help reproduce the pictures for the course "Lasso"
# Author: Joseph Salmon <joseph.salmon@telecom-paristech.fr>
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from numpy.random import multivariate_normal
from sklearn.linear_model.base import LinearModel
from sklearn.model_selection import GridSearchCV
from sklearn.base import RegressorMixin
from sklearn.linear_model import Lasso
from scipy import linalg


def my_nonzeros(vector, eps_machine=1e-12):
    """Compute support up to eps_machine precision."""
    indexes_to_keep = np.where(np.abs(vector) > eps_machine)[0]
    return indexes_to_keep


def refitting(coefs, X, y, eps_machine=1e-12):
    """Compute the  LeastSquare fit over the support of coefs.

    Parameters
    ----------
    coefs: ndarray, shape (n_features, n_kinks); Matrix of n_kinks candidate
    beta. (eg. obtained with the lars algorithm). Implicitely assume the
    first

    X: ndarray,shape (n_samples, n_features); Design matrix aka covariates
    elements

    y : ndarray, shape = (n_samples,); noisy vector of observation

    Returns
    -------
    all_solutions: ndarray, shape (n_kinks, n_features); refitted solution
    based on the original coefs vectors

    index_list: list, shape(n_kinks) . list of the indexes chosen at each
    kink. Note that the first element of the list is the empty list
    (no variable used first)

    index_size:

    """
    regr = LinearRegression(fit_intercept=False)
    # print np.ndim(coefs)
    if np.ndim(coefs) == 1:
        # print 'Taille'
        n_features = coefs.size
        all_solutions = np.zeros(n_features)
        indexes = my_nonzeros(coefs, eps_machine=eps_machine)
        if len(indexes) == 0:
            indexes_to_keep = []
            index_size = 0
            all_solutions = all_solutions
        else:
            indexes_to_keep = np.reshape(indexes, -1)
            regr.fit(X[:, indexes_to_keep], y)
            all_solutions[indexes_to_keep] = regr.coef_
            index_list = indexes_to_keep
            index_size = len(indexes_to_keep)
    else:
        # print 'plusieurs coefs'
        n_features, n_kinks = coefs.shape
        all_solutions = np.zeros((n_features, n_kinks))
        index_list = []
        index_list.append([])
        index_size = np.zeros(n_kinks, int)
        for k in range(n_kinks - 1):
            indexes = np.nonzero(coefs[:, k + 1])[0]
            # print indexes
            indexes_to_keep = np.reshape(indexes, -1)

            if len(indexes) == 0:
                index_list.append([])
                index_size[k + 1] = 0
                all_solutions[..., k + 1] = np.zeros(n_features)
            else:
                # print indexes_to_keep
                regr.fit(X[:, indexes_to_keep], y)
                all_solutions[indexes_to_keep, k + 1] = regr.coef_
                index_list.append(indexes_to_keep)
                index_size[k + 1] = len(indexes_to_keep)

    return all_solutions, index_list, index_size


class LSLasso(LinearModel, RegressorMixin):
    """Docstring for LSLasso."""

    def __init__(self, alpha=1.0, eps_machine=1e-12,
                 max_iter=10000, tol=1e-7, fit_intercept=False):
        self.alpha = alpha
        self.max_iter = max_iter
        self.eps_machine = eps_machine
        self.tol = tol
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        regr = LinearRegression(fit_intercept=self.fit_intercept)
        lasso_clf = Lasso(alpha=self.alpha, fit_intercept=self.fit_intercept,
                          max_iter=self.max_iter, tol=self.tol)
        lasso_clf.fit(X, y)
        n_features = lasso_clf.coef_.shape
        indexes_to_keep = my_nonzeros(lasso_clf.coef_,
                                      eps_machine=self.eps_machine)
        if len(indexes_to_keep) == 0:
            coef = np.zeros(n_features[0],)
            self.intercept_ = np.mean(y)
        else:
            coef = np.zeros(n_features[0],)
            regr.fit(X[:, indexes_to_keep], y)
            coef[indexes_to_keep] = regr.coef_
            self.intercept_ = regr.intercept_
        self.coef_ = coef
        return self


def LSLassoCV(X, y, alpha_grid, n_jobs=1, cv=10, max_iter=10000, tol=1e-7,
              fit_intercept=False):
    """Compute the best Lasso through CV."""
    param_grid = dict(alpha=alpha_grid)
    sr_test = LSLasso(max_iter=max_iter, tol=tol, fit_intercept=fit_intercept)
    gs = GridSearchCV(sr_test, cv=cv, param_grid=param_grid, n_jobs=n_jobs)
    gs.fit(X, y)
    index_LSLassoCV = np.where(alpha_grid == gs.best_params_['alpha'])[0]
    coef_LSLassoCV = gs.best_estimator_.coef_
    return coef_LSLassoCV, index_LSLassoCV


def ridge_path(X, y, alphas):
    """Compute the Ridge path"""
    U, s, Vt = linalg.svd(X, full_matrices=False)
    d = s / (s[:, np.newaxis].T ** 2 + alphas[:, np.newaxis])
    return np.dot(d * U.T.dot(y), Vt).T


def PredictionError(X, coefs_path, beta):
    """Compute all the  ||X beta-X coefs_path[i]||^2/n_samples for true beta
    and observed coefficients coef_path.

    Parameters
    ----------
    X: shape (n_samples, n_features); Design matrix aka covariates elements

    coefs_path: shape (n_features, n_kinks); Matrix of n_kinks, beta estimation

    beta: shape (n_features, ); original coefficients

    Returns
    -------
    Err: shape (n_kinks, ) float vector with ith coordinate
    ||X beta-X coefs_path[i]||^2/n_samples

    """
    n_samples, n_features = X.shape
    if np.ndim(coefs_path) == 1:
        err = np.sum((np.dot(X, beta) - np.dot(X, coefs_path)) ** 2)
    else:
        n_features, n_kinks = coefs_path.shape
        err = np.sum((np.tile(np.dot(X, beta), (n_kinks, 1)).T -
                      np.dot(X, coefs_path)) ** 2, 0)

    return err / n_samples


def EstimationError(coefs_path, beta):
    """Compute the estimation error for the true signal"""
    if np.ndim(coefs_path) == 1:
        err = np.max(np.abs(beta - coefs_path))
    else:
        n_features, n_kinks = coefs_path.shape
        err = np.max(np.abs(np.tile(beta, (n_kinks, 1)).T - coefs_path), 0)

    return err


def ScenarioEquiCor(n_samples=10, n_features=50, sig_noise=0.1, rho=0.5, s=5,
                    normalize=True, noise_type='Normal'):
    """Compute an n-sample y=X b+ e where the covariates have a fixed
    correlation level and the noise added is Gaussian with std sig_noise

    Parameters
    ----------
    n_samples: number of independant sample to be generated (usually "n")

    n_features: number of features to be generated (usually "p")

    sig_noise: std deviation of the additive White Gaussian noise

    rho: correlation between the covariates (S_ii=1 and S_ij=rho) where
    S is the covariance matrix of the p covariates.

    s: sparsity index of the underlying true coefficient vector

    normalize : boolean, optional, default True
        If ``True``, the regressors X will be normalized before regression.

    Returns
    -------
    y : ndarray, shape = (n_samples,); Target values of the scenario

    X :  shape (n_samples, n_features); Design matrix aka covariates elements

    beta : ndarray, shape = (,n_features);

    """
    beta = np.zeros((n_features,))
    beta[0:s] = 1
    covar = (1 - rho) * np.eye(n_features) + \
        rho * np.ones([n_features, n_features])

    X = multivariate_normal(np.zeros(n_features,), covar, [n_samples])
    if normalize is True:
        X /= np.sqrt(np.sum(X ** 2, axis=0) / n_samples)
    else:
        X

    y = AddNoise(np.dot(X, beta), sig_noise, noise_type)
    return y, beta, X


def AddNoise(true_signal, sig_noise, noise_type='Normal'):
    """Adding noise part to underlying true signal."""
    n_samples = true_signal.shape[0]
    if noise_type == 'Normal':
        epsilon = sig_noise * np.random.randn(n_samples,)
    elif noise_type == 'Laplace':
        epsilon = np.random.laplace(0, sig_noise / np.sqrt(2), n_samples)
    else:
        epsilon = sig_noise * np.random.randn(n_samples,)
    y = true_signal + epsilon
    return y
