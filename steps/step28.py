if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from dezero import Variable


def rosenbrock(x0: Variable, x1: Variable) -> Variable:
    return 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2


if __name__ == "__main__":
    lr: float = 0.001
    iters: int = 50000

    x0 = Variable(np.array(0.0))
    x1 = Variable(np.array(2.0))

    for i in range(iters):
        y: Variable = rosenbrock(x0, x1)

        x0.cleargrad()
        x1.cleargrad()
        y.backward()

        x0.data -= lr * x0.grad
        x1.data -= lr * x1.grad

        if i % 1000 == 0:
            print(f"{x0=}, {x1=}")