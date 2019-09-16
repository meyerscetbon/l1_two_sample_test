import numpy as np
import matplotlib.pyplot as plt

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
num_samples = 300
my = 1

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


fig, ax = plt.subplots()
contours_1 = plt.contour(T_1, T_2, Z_1, 2, colors="green", linestyles="-.")
# plt.clabel(contours_1, inline=True, fontsize=8)
contours_2 = plt.contour(T_1, T_2, Z_2, 2, colors="blue", linestyles="-.")
# plt.clabel(contours_2, inline=True, fontsize=8)
contours_3 = plt.contour(
    T_1, T_2, Z_diff, 2, colors="red", linestyles="-", linewidth=100
)
plt.clabel(contours_3, inline=True, fontsize=8)
ax.plot(X[:, 0], X[:, 1], "go", markersize=3, label="P")
ax.plot(Y[:, 0], Y[:, 1], "bx", markersize=3, label="Q")
ax.set_xlim([z1min, z1max])
ax.set_ylim([z2min, z2max])
plt.axis("off")
# plt.subplots(left=0, right=1, bottom=0, top=1)
plt.show()
fig.savefig("plot_contour.jpg")