if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from dezero import Variable


if __name__ == "__main__":
    x0 = Variable(np.array([1, 2, 3]))
    x1 = Variable(np.array([10]))
    y: Variable = x0 + x1
    y.backward()
    print(f"{y=}")
    print(f"{x0.grad=}")
    print(f"{x1.grad=}")