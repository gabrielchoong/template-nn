import numpy as np


def F1(x):

    s = np.sum(x**2)

    return s

def getFunctionDetails(a):

    param = { 0: ["F1", -100, 100, 30] }

    return param.get(a, "nothing")