import numpy as np
from scipy import io


def load_data(filename):
    return io.loadmat(file_name=filename)


def make_connectivity_matrix(N1, N2, p, rng):
    c = np.zeros((N1, N2))  # storing connections
    c_t = rng.random((N1, N2))  # random matrix
    c[c_t < p] = 1

    return c
