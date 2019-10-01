"""
Demo the interpretable feature optimized to discriminate two
distributions.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import l1_two_sample_test

# Full sample size
n = 500
# dimension
dim = 2
# mean shift
my = 1
# GMD: Samples X,Y
cov = np.eye(dim)

mean_1 = np.zeros(dim)
X = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), n)

mean_shift = np.zeros(dim)
mean_shift[0] = my
Y = np.random.multivariate_normal(mean_shift, np.eye(dim), n)

# Location T_1
T_1 = np.zeros(2)
T_1[1] = 0
T_1[0] = -5
gwidth2 = 5


npts = 30
x = np.linspace(-6, 6, npts)
y = np.linspace(-6, 6, npts)

xf, yf = np.meshgrid(x, y)


Z = np.zeros((npts, npts))
for i in range(npts):
    for j in range(npts):
        Z[i, j] = l1_two_sample_test.compute_stat_ME(
            X, Y, np.vstack((T_1, (xf[i, j], yf[i, j]))), gwidth2
        )


fig, ax = plt.subplots(1, 1, figsize=(6, 4))
cs = plt.contourf(xf, yf, Z, cmap="PuBuGn")
norm_ = matplotlib.colors.Normalize(vmin=cs.cvalues.min(), vmax=cs.cvalues.max())
sm = plt.cm.ScalarMappable(norm=norm_, cmap=cs.cmap)
sm.set_array([])
fig.colorbar(sm, ticks=cs.levels)
plt.plot(X[:, 0], X[:, 1], "k.", label="$X\sim P$", alpha=0.7)
plt.plot(Y[:, 0], Y[:, 1], "r.", label="$Y\sim Q$", alpha=0.7)
plt.plot(T_1[0], T_1[1], "^", markersize=20, color="black", zorder=3)
plt.xticks([])
plt.yticks([])
plt.show()

fig.savefig("informative.png")
