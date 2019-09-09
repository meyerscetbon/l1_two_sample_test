import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import matplotlib
import l1_two_sample_test


def compute_stat_ME(X, Y, T_1, T_2, gwidth2):
    test_locs = np.vstack((T_1, T_2))

    n, d = X.shape
    m, d = Y.shape
    J, d = test_locs.shape

    t = n + m
    ro = n / t

    z_1 = l1_two_sample_test.gauss_kernel(X, test_locs, gwidth2)  # num_samples*J
    cov_1 = np.cov(z_1.T)

    z_2 = l1_two_sample_test.gauss_kernel(Y, test_locs, gwidth2)
    cov_2 = np.cov(z_2.T)

    cov = (1 / ro) * cov_1 + (1 / (1 - ro)) * cov_2

    reg = 1e-5
    S = np.mean(z_1, axis=0) - np.mean(z_2, axis=0)
    S = np.sqrt(t) * np.linalg.solve(sqrtm(cov + reg * np.eye(J)), S)

    S = np.sum(np.abs(S))

    return S


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
        Z[i, j] = compute_stat_ME(X, Y, T_1, np.hstack((xf[i, j], yf[i, j])), gwidth2)


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

fig.savefig("informative.pdf")
