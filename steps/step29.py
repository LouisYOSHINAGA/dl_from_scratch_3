if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from dezero import Variable


def f(x: Variable) -> Variable:
    return x ** 4 - 2 * x ** 2

def gx2(x: np.ndarray) -> np.ndarray:
    return 12 * x ** 2 - 4


if __name__ == "__main__":
    x = Variable(np.array(2.0))
    iters = 10
    for i in range(iters):
        y: Variable = f(x)
        x.cleargrad()
        y.backward()
        x.data -= x.grad / gx2(x.data)
        print(f"{i}: {x}")