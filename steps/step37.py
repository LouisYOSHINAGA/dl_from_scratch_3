if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from dezero import Variable
import dezero.functions as F


if __name__ == "__main__":
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    c = Variable(np.array([[10, 20, 30], [40, 50, 60]]))
    t: Variable = x + c
    y: Variable = F.sum(t)
    y.backward(retain_grad=True)
    print(f"{t=}")
    print(f"{y.grad=}")
    print(f"{t.grad=}")
    print(f"{x.grad=}")
    print(f"{c.grad=}")