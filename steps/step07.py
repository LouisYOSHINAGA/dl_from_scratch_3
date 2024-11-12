import numpy as np
from typing import Callable, Any, Self


class Variable:
    def __init__(self, data: np.ndarray) -> None:
        self.data: np.ndarray = data
        self.grad: np.ndarray|None = None
        self.creator: Callable[[Any], Self]|None = None

    def set_creator(self, func: Callable[[Any], Self]) -> None:
        self.creator = func

    def backward(self) -> None:
        f: Callable[[Any], Self] = self.creator
        if f is not None:
            x: Self = f.input
            x.grad = f.backward(self.grad)
            x.backward()


class Function:
    def __call__(self, input: Variable) -> Variable:
        x: np.ndarray = input.data
        y: np.ndarray = self.forward(x)
        output = Variable(y)
        output.set_creator(self)

        self.input: Variable = input
        self.output: Variable = output
        return output

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def backward(self, gy: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x ** 2

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x: np.ndarray = self.input.data
        return 2 * x * gy


class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    def backward(self, gy):
        x: np.ndarray = self.input.data
        return np.exp(x) * gy


def numerical_diff(f: Callable[[Variable], Variable], x: Variable, eps: float =1e-4) -> np.ndarray:
    y0: Variable = f(Variable(x.data - eps))
    y1: Variable = f(Variable(x.data + eps))
    return (y1.data - y0.data) / (2 * eps)


if __name__ == "__main__":
    A: Callable[[Variable], Variable] = Square()
    B: Callable[[Variable], Variable] = Exp()
    C: Callable[[Variable], Variable] = Square()

    x = Variable(np.array(0.5))
    a: Variable = A(x)
    b: Variable = B(a)
    y: Variable = C(b)

    assert y.creator == C
    assert y.creator.input == b
    assert y.creator.input.creator == B
    assert y.creator.input.creator.input == a
    assert y.creator.input.creator.input.creator == A
    assert y.creator.input.creator.input.creator.input == x

    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)