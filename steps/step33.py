if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from dezero import Variable


def f(x: Variable) -> Variable:
    return x ** 4 - 2 * x ** 2


if __name__ == "__main__":
    iters: int = 10
    x = Variable(np.array(2.0))
    for i in range(iters):
        print(f"iter {i:02d}: {x=}")

        y: Variable = f(x)
        x.cleargrad()
        y.backward(create_graph=True)

        gx: Variable = x.grad
        x.cleargrad()
        gx.backward()

        gx2: Variable= x.grad
        x.data -= gx.data / gx2.data