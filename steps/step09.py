import numpy as np
from typing import Callable, Any, Self


class Variable:
    def __init__(self, data: np.ndarray) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported.")
        self.data: np.ndarray = data
        self.grad: np.ndarray|None = None
        self.creator: Callable[[Any], Self]|None = None

    def set_creator(self, func: Callable[[Any], Self]) -> None:
        self.creator = func

    def backward(self) -> None:
        """
            example:
                L = r( q( p(u) ) )  // L = r(w), w = q(v), v = p(u)

                dL/du = dL/dw * dw/dv * dv/du   // dL/du  = r'(w) * q'(v) * p'(u)
                      = w.grad * dw/dv * dv/du  // w.grad = r'(w)
                      = v.grad * dv/du          // v.grad = w.grad * q'(v)
                      = u.grad                  // u.grad = v.grad * p'(u)

                funcs = [r]
                f = r, x = w, y = L          // L = r(w)
                w.grad = r.backward(L.grad)  // dL/dw = dL/dL * r'(w)

                funcs = [q]
                f = q, x = v, y = w          // w = q(v)
                v.grad = q.backward(w.grad)  // dL/dv = dL/dw * q'(v)

                funcs = [p]
                f = p, y = v, x = u          // v = p(u)
                u.grad = p.backward(v.grad)  // dL/du = dL/dv * p'(u)
        """
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs: list[Callable[[Any], Self]] = [self.creator]
        while funcs:
            f: Callable[[Any], Self] = funcs.pop()
            x: Self = f.input
            y: Self = f.output

            x.grad = f.backward(y.grad)
            if x.creator is not None:
                funcs.append(x.creator)


def as_array(x: int|float|np.ndarray) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, input: Variable) -> Variable:
        x: np.ndarray = input.data
        y: np.ndarray = self.forward(x)
        output = Variable(as_array(y))
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

def square(x: Variable) -> Variable:
    return Square()(x)


class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    def backward(self, gy):
        x: np.ndarray = self.input.data
        return np.exp(x) * gy

def exp(x: Variable) -> Variable:
    return Exp()(x)


def numerical_diff(f: Callable[[Variable], Variable], x: Variable, eps: float =1e-4) -> np.ndarray:
    y0: Variable = f(Variable(x.data - eps))
    y1: Variable = f(Variable(x.data + eps))
    return (y1.data - y0.data) / (2 * eps)


if __name__ == "__main__":
    x = Variable(np.array(0.5))
    y: Variable = square(exp(square(x)))
    y.backward()
    print(x.grad)