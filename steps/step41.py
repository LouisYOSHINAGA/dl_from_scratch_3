if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Variable
import dezero.functions as F


if __name__ == "__main__":
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    c = np.dot(a, b)
    print(c)

    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    c = np.dot(a, b)
    print(c)

    x = Variable(np.random.randn(2, 3))
    W = Variable(np.random.randn(3, 4))
    y = F.matmul(x, W)
    y.backward()

    print(x.grad.shape)
    print(W.grad.shape)