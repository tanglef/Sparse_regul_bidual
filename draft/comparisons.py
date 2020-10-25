import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.construct import rand
import seaborn as sns
sns.set()
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import os
import sys
from sklearn.linear_model import LinearRegression
path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(path, "..")) # to find utils from draft
from utils import save_fig
params = {'axes.labelsize': 18,
          'font.size': 16,
          'legend.fontsize': 'xx-large',
          'figure.figsize': (8, 6)}
plt.rcParams.update(params)
from prox_computation import fista
SEED = 11235813
np.random.seed(SEED)

n = 50
x_true = np.array([3]*15 + [0]*25)
sigma = .1
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

CV = 3
alphas = np.logspace(np.log10(10), np.log10(1e-7),
                     num=50)
print("################# Finished LASSO" )

#########################
# Make LASSO
#########################

lasso = linear_model.LassoCV(alphas=alphas, fit_intercept=False, normalize=False,
                                     cv=CV, random_state=SEED)
lasso.fit(A, b)
_, x_lasso, _ = linear_model.lasso_path(A, b, alphas=alphas, fit_intercept=False,
                               return_models=False)
alpha_CV = lasso.alpha_
index_lasso = np.where(alphas == alpha_CV)[0][0]

#########################
# Elastic NET
#########################

l1_ratio = np.logspace(np.log10(.99), np.log10(1e-4),
                     num=20)

el_net =  linear_model.ElasticNetCV(alphas=alphas, l1_ratio=l1_ratio,
                     fit_intercept=False, normalize=False, cv=CV,
                      max_iter=1e5, random_state=SEED)
el_net.fit(A, b)
alpha_cv_net = el_net.alpha_
l1_cv_net = el_net.l1_ratio_
_, x_enet, _ = linear_model.enet_path(A, b, l1_ratio = l1_cv_net, alphas=alphas,
                             fit_intercept=False, return_models=False)
index_enet = np.where(alphas == alpha_cv_net)
print("################# Finished Elastic-net" )

###########################
# Proximal
###########################

def choose_lambda(A, b, lambda_l, eps):
    err = []
    for lambda_ in lambda_l:
        A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=.3)
        reg = fista(A_train, b_train, lambda_, 500, 15, eps=eps)
        b_pred = A_test @ reg
        err.append(((b_pred - b_test) ** 2).sum())
    return lambda_l[np.argmin(err)]

lambda_list = np.logspace(np.log10(.99), np.log10(1e-4),
                     num=20)
lambda_best = choose_lambda(A, b, lambda_list, eps=1e-7)
print("The best lambda is", lambda_best)
x_prox = fista(A, b, lambda_best, 500, 15, eps=1e-10)
print("################# Finished Elastic-net bireg" )

###########################
# Plot all signals
###########################

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
ax.plot(x_true, label="True signal")
ax.plot(x_lasso[:, index_lasso], '--', label="Lasso")
ax.plot(x_enet[:, index_enet].reshape(-1), '-.', label="Elastic-Net")
plt.legend(fontsize='x-large', title_fontsize='20')
plt.savefig(save_fig(path, "Lasso_enet", "pdf"))


fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
ax.plot(x_true, label="True signal")
ax.plot(x_enet[:, index_enet].reshape(-1), '-.', label="Elastic-Net")
ax.plot(x_prox, '-', label="Proximal")
plt.legend(fontsize='x-large', title_fontsize='20')
plt.savefig(save_fig(path, "enet_proxi", "pdf"))
plt.show()