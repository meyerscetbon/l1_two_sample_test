"""
Plot simple figures to give intuitions, for use in posters or
presentations.
"""

import numpy as np
import matplotlib.pyplot as plt

#### Contour plot ####

X, Y = np.mgrid[1:2:100j, 2:3:100j]


def gauss_kernel(X, test_locs, gwidth2):
    """Compute a X.shape[0] x test_locs.shape[0] Gaussian kernel matrix"""
    n, d = X.shape
    D2 = (
        np.sum(X ** 2, 1)[:, np.newaxis]
        - 2 * np.dot(X, test_locs.T)
        + np.sum(test_locs ** 2, 1)
    )
    K = np.exp(-D2 / (2.0 * gwidth2))
    return K


dim = 2
num_samples = 200
my = 1

np.random.seed(3)
X = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), num_samples)
x1min = X[:, 0].min()
x1max = X[:, 0].max()
x2min = X[:, 1].min()
x2max = X[:, 1].max()


mean_shift = np.zeros(dim)
mean_shift[0] = my
Y = np.random.multivariate_normal(mean_shift, np.eye(dim), num_samples)
y1min = Y[:, 0].min()
y1max = Y[:, 0].max()
y2min = Y[:, 1].min()
y2max = Y[:, 1].max()


z1min = min(x1min, y1min)
z1max = max(x1max, y1max)

z2min = min(x2min, y2min)
z2max = max(x2max, y2max)

T_1, T_2 = np.mgrid[z1min:z1max:100j, z2min:z2max:100j]
positions = np.vstack([T_1.ravel(), T_2.ravel()]).T

z_1 = gauss_kernel(X, positions, 1)
kernel_1 = np.mean(z_1, axis=0)
Z_1 = np.reshape(kernel_1, T_1.shape)

z_2 = gauss_kernel(Y, positions, 1)
kernel_2 = np.mean(z_2, axis=0)
Z_2 = np.reshape(kernel_2, T_1.shape)

Z_diff = np.abs(Z_1 - Z_2)


fig, ax = plt.subplots(figsize=(4, 2.5))
contours_1 = plt.contour(T_1, T_2, Z_1, 2, colors="C0", linestyles="-.", linewidths=2)
contours_2 = plt.contour(T_1, T_2, Z_2, 2, colors="C1", linestyles="-.", linewidths=2)
# img = plt.pcolormesh(
#    T_1, T_2, Z_diff, cmap="gray_r")
# imshow code rather than pcolormesh: it produces better figures
img = plt.imshow(
    Z_diff.T,
    extent=(T_2.min(), T_2.max(), T_1.min(), T_1.max()),
    origin="lower",
    cmap="gray_r",
    aspect="auto",
)
contours_3 = plt.contour(
    T_1, T_2, Z_diff, 2, colors="k", linestyles="-", linewidths=0.8
)
# plt.clabel(contours_3, inline=True, fontsize=8)
ax.plot(X[:, 0], X[:, 1], "o", color="C0", markersize=4, label="P")
ax.plot(Y[:, 0], Y[:, 1], "x", color="C1", markersize=5, label="Q")
ax.set_xlim([z1min, z1max])
ax.set_ylim([z2min, z2max])
plt.axis("off")
plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
plt.show()
fig.savefig("plot_contour.jpg")
fig.savefig("plot_contour.pdf")


##### l1_vs_l2 #####

num_angles = 1000
dim = 2
angles = np.linspace(0, 2 * np.pi, num_angles)
L2_unit = np.zeros((num_angles, dim))
L1_unit = np.zeros((num_angles, dim))
L1_root2 = np.zeros((num_angles, dim))
for k, theta in enumerate(angles):
    L2_unit[k, :] = np.array([np.cos(theta), np.sin(theta)])
    norm_1 = np.abs(np.cos(theta)) + np.abs(np.sin(theta))
    L1_unit[k, :] = np.array([np.cos(theta) / norm_1, np.sin(theta) / norm_1])
    root2 = 2 * np.sqrt(1 / 2)
    L1_root2[k, :] = np.array(
        [np.cos(theta) * root2 / norm_1, np.sin(theta) * root2 / norm_1]
    )

coor_0 = [0, 0]
d1 = [1, 0]
d2 = [np.sqrt(1 / 2), np.sqrt(1 / 2)]

fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(L1_unit[:, 0], L1_unit[:, 1], "-", color="C1")
ax.plot(L2_unit[:, 0], L2_unit[:, 1], "-", color="C2")
ax.plot(L1_root2[:, 0], L1_root2[:, 1], "-", color="C1")
ax.spines["left"].set_position("center")
ax.spines["bottom"].set_position("center")
ax.spines["right"].set_color("none")
ax.spines["top"].set_color("none")
ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")
plt.xticks([], [])
plt.yticks([], [])
plt.show()
fig.savefig("plot_l1_vs_l2.jpg")


##### Weak Convergence #####
n_samples = 1

X = np.random.normal(0, 1, n_samples)
Y = np.random.normal(3, 1, n_samples)


Z = np.zeros(n_samples)

sig = 0.2


def kernel(x, t):
    return np.exp(-((x - t) ** 2) / (2 * sig))


t = np.linspace(-3, 10, num=100)

K_X = []
for x in X:
    K_X.append(kernel(x, t))

K_Y = []
for y in Y:
    K_Y.append(kernel(y, t))


mean_X = np.array(K_X)
mean_X = mean_X.mean(axis=0)

mean_Y = np.array(K_Y)
mean_Y = mean_Y.mean(axis=0)


witness = mean_X - mean_Y


fig, ax = plt.subplots(1, 1)
plt.scatter(X, Z, s=100, color="b")  # ,label='P')
plt.scatter(Y, Z, s=100, color="r")  # ,label='Q')
ax.legend(bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.5)

plt.plot(t, mean_X, color="b")
plt.plot(t, mean_Y, color="r")

plt.plot(t, np.abs(witness), color="k", linestyle="-.")  # ,label="Witness")

plt.xlim(xmin=-3, xmax=7)
plt.ylim(ymin=-1.5, ymax=3)
plt.axis("off")
ax.legend()
plt.show()
fig.savefig("witness_dirac.jpg")
