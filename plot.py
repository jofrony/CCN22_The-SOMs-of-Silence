import matplotlib.pyplot as plt
import numpy as np

def plot_raster(spk_idx, spk_time):

    cell_number = np.unique(spk_idx)
    plt.figure()

    for i, c in enumerate(cell_number):

        idx = np.where(spk_idx == c)
        spkt = np.take(spk_time, idx)

        level = np.ones_like(spkt) * i

        plt.scatter(spkt, level, s=0.1, c="black")

    plt.ylabel("Index")
    plt.xlabel("Time (ms)")
    plt.show()