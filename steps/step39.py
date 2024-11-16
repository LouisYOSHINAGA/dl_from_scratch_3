if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from dezero import Variable
import dezero.functions as F


if __name__ == "__main__":
    x = Variable(np.array([1, 2, 3, 4, 5, 6]))
    y: Variable = F.sum(x)
    y.backward()
    print(f"{y=}")
    print(f"{x.grad=}")
    print()


    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y: Variable = F.sum(x)
    y.backward()
    print(f"{y=}")
    print(f"{x.grad=}")
    print()


    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y: Variable = F.sum(x, axis=0)
    y.backward()
    print(f"{y=}")
    print(f"{x.grad=}")
    print()


    x = Variable(np.ramdom.default_rng().random(2, 3, 4, 5))
    y: Variable = x.sum(keepdims=True)
    print(f"{y.shape=}")