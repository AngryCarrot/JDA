import numpy as np


def kernel(ker, X, X2=None, gamma=1.0):
    if ker == "linear":
        if X2 is None:
            return np.matmul(X.T, X)
        else:
            return np.matmul(X.T, X2)