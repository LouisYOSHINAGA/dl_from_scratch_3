if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import math
import numpy as np
from dezero import Function, Variable
from dezero.utils import plot_dot_graph


class Sin(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.sin(x)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x: np.ndarray = self.inputs[0].data
        return gy * np.cos(x)

def sin(x: Variable) -> Variable:
    return Sin()(x)


def my_sin(x: Variable, threshold: float =1e-4) -> Variable:
    y: Variable = 0
    for i in range(10000):
        c: float = (-1) ** i / math.factorial(2 * i + 1)
        t: Variable = c * x ** (2 * i + 1)
        y += t
        if abs(t.data) < threshold:
            break
    return y


if __name__ == "__main__":
    x = Variable(np.array(np.pi/4))
    y: Variable = sin(x)
    y.backward()
    print(f"{y.data=}")
    print(f"{x.grad=}")
    print()

    x = Variable(np.array(np.pi/4), name="x")
    y: Variable = my_sin(x)
    y.name = "y"
    y.backward()
    print(f"{y.data=}")
    print(f"{x.grad=}")
    print()
    plot_dot_graph(y, verbose=False, to_file="step27_1.png")

    x = Variable(np.array(np.pi/4), name="x")
    y: Variable = my_sin(x, threshold=1e-150)
    y.name = "y"
    y.backward()
    print(f"{y.data=}")
    print(f"{x.grad=}")
    plot_dot_graph(y, verbose=False, to_file="step27_2.png")