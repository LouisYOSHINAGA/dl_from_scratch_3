if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from dezero import Variable
import dezero.functions as F


if __name__ == "__main__":
    x = Variable(np.random.randn(2, 3))
    W = Variable(np.random.randn(3, 4))
    y: Variable = F.matmul(x, W)
    y.backward()
    print(f"{x.grad.shape=}")
    print(f"{W.grad.shape=}")