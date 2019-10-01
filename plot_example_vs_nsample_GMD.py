import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


alpha = 0.01
num_samples_test = np.arange(500, 6000, 500)

labels = ["L1_opt_J_ME", "L1_grid_J_ME", "L1_opt_J_SCF", "L1_grid_J_SCF"]
num_of_tests = len(labels)

# Import results from .csv file
data_l1 = pd.read_csv("l1_test_vs_nsample.csv", header=None)
columns_names = [
    "method",
    "num_of_samples",
    "L1_opt_J_ME",
    "L1_grid_J_ME",
    "L1_opt_J_SCF",
    "L1_grid_J_SCF",
]
data_l1.columns = columns_names


fig, ax = plt.subplots(1, 1)
linestyles = ["-", "-.", "-", "-."]
colors = ["b", "g", "r", "y"]

to_plot = np.zeros((np.shape(num_samples_test)[0], num_of_tests))
for i, num_sample_test in enumerate(num_samples_test):

    data_ = data_l1[data_l1["num_of_samples"] == num_sample_test]
    plot_points = data_.iloc[:, 2 : 2 + num_of_tests].mean().values
    to_plot[i, :] = plot_points

for j in range(num_of_tests):
    ax.plot(
        num_samples_test,
        to_plot[:, j],
        label=labels[j],
        linestyle=linestyles[j],
        color=colors[j],
    )

yticks = ax.get_yticks()
ax.set_yticks(yticks[::2])
ax.set_xticks([0, 2500, 5000])
ax.set_ylim(ymin=0)
ax.legend(loc=(1.1, 0.7))

plt.show()
fig.savefig("plot_l1_vs_nsample_GMD.pdf", bbox_inches="tight")
