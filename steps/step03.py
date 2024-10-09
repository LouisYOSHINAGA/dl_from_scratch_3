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
    def forward(self, x: Variable) -> Variable:
        return x ** 2


class Exp(Function):
    def forward(self, x: Variable) -> Variable:
        return np.exp(x)


if __name__ == "__main__":
    A: Callable[[Variable], Variable] = Square()
    B: Callable[[Variable], Variable] = Exp()
    C: Callable[[Variable], Variable] = Square()

    x = Variable(np.array(0.5))
    a = A(x)  # x**2
    b = B(a)  # exp(x**2)
    y = C(b)  # exp(x**2)**2

    print(y.data)