import numpy as np


def copy_to_2d_floats(array: np.ndarray):
    return np.copy(array.astype(float)[:, :2])


def copy_to_3d_floats(array: np.ndarray):
    return np.copy(array.astype(float)[:, :3])