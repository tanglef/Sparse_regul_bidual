import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()
from sklearn import linear_model
SEED = 11235813


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

CV = 5
alphas = np.logspace(np.log10(10), np.log10(1e-7),
                     num=50)

#########################
# Make LASSO
#########################

lasso = linear_model.LassoCV(alphas=alphas, fit_intercept=False, normalize=False,
                                     cv=CV)
lasso.fit(A, b)
_, x_lasso, _ = linear_model.lasso_path(A, b, alphas=alphas, fit_intercept=False,
                               return_models=False)
alpha_CV = lasso.alpha_
index_lasso = np.where(alphas == alpha_CV)[0][0]

#########################
# Elastic NET
#########################

l1_ratio = np.logspace(np.log10(.9), np.log10(1e-9),
                     num=50)

el_net =  linear_model.ElasticNetCV(alphas=alphas, l1_ratio=l1_ratio,
                     fit_intercept=False, normalize=False, cv=CV,
                      max_iter=1e5)
el_net.fit(A, b)
alpha_cv_net = el_net.alpha_
l1_cv_net = el_net.l1_ratio_
_, x_enet, _ = linear_model.enet_path(A, b, l1_ratio = l1_cv_net, alphas=alphas,
                             fit_intercept=False, return_models=False)
index_enet = np.where(alphas == alpha_cv_net)

###########################
# Plot all signals
###########################

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
ax.plot(x_true, label="True signal")
ax.plot(x_lasso[:, index_lasso], '--', label="Lasso")
ax.plot(x_enet[:, index_enet].reshape(-1), '-.', label="Elastic-Net")
plt.legend()
plt.show()