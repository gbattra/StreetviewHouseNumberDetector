# initialize epsilon for weights
import numpy as np


def epsilon_init(dimensions):
    ones = np.ones((1, len(dimensions)))
    e_init = np.sqrt(6) / np.square(np.sum(ones.dot(dimensions)))

    return e_init
