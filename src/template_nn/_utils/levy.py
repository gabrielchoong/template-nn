import math

import numpy as np


def Levy(dimensions) -> np.ndarray | np.float64:
    beta = 1.5

    numerator = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
    denominator = math.gamma(1 + beta) / 2 * beta * 2 ** ((beta - 1) / 2)
    quotient = numerator / denominator

    sigma = quotient ** (1 / beta)

    u = 0.01 * np.random.randn(dimensions) * sigma
    v = np.random.rand(dimensions)
    zz = np.power(np.absolute(v), (1 / beta))
    step = np.divide(u, zz)
    return step
