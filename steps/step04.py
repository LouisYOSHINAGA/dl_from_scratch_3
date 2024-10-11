import numpy as np
from typing import Callable


class Variable:
    def __init__(self, data: np.ndarray) -> None:
        self.data: np.ndarray = data


class Function:
    def __call__(self, input: Variable) -> Variable:
        x: np.ndarray = input.data
        y: np.ndarray = self.forward(x)
        return Variable(y)

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x ** 2


class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)


def numerical_diff(f: Callable[[Variable], Variable], x: Variable, eps: float =1e-4) -> np.ndarray:
    y0: Variable = f(Variable(x.data - eps))
    y1: Variable = f(Variable(x.data + eps))
    return (y1.data - y0.data) / (2 * eps)


if __name__ == "__main__":
    def f(x: Variable) -> Variable:
        A: Callable[[Variable], Variable] = Square()
        B: Callable[[Variable], Variable] = Exp()
        C: Callable[[Variable], Variable] = Square()
        return C(B(A(x)))

    x = Variable(np.array(0.5))
    dy: np.ndarray = numerical_diff(f, x)
    print(dy)