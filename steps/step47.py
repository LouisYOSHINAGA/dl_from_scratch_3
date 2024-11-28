if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from dezero import Variable, as_variable
import dezero.functions as F
from dezero.models import MLP


def softmax1d(x: np.ndarray|Variable) -> Variable:
    x = as_variable(x)
    y: Variable = F.exp(x)
    sum_y: Variable = F.sum(y)
    return y / sum_y


if __name__ == "__main__":
    model = MLP((10, 3))
    x = Variable(np.array([[0.2, -0.4]]))
    y: Variable = model(x)
    p: Variable = softmax1d(y)
    print(f"{y=}")
    print(f"{p=}")


    x: np.ndarray = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
    t: np.ndarray = np.array([2, 0, 1, 0])
    y: Variable = model(x)
    loss: Variable = F.softmax_cross_entropy(y, t)
    print(f"{loss=}")