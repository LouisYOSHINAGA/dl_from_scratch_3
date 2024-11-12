import numpy as np
from typing import Callable, Any, Self
import unittest


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
    def __call__(self, *inputs: Variable) -> list[Variable]|Variable:
        xs: list[np.ndarray] = [x.data for x in inputs]
        ys: list[np.ndarray]|np.ndarray = self.forward(*xs)
        if not isinstance(ys, list):
            ys = [ys]

        outputs: list[Variable] = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)

        self.inputs: list[Variable] = inputs
        self.outputs: list[Variable] = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs: list[np.ndarray]) -> list[np.ndarray]:
        raise NotImplementedError()

    def backward(self, gys: list[np.ndarray]) -> list[np.ndarray]:
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


class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 + x1

def add(x0: Variable, x1: Variable) -> Variable:
    return Add()(x0, x1)


class SquareTest(unittest.TestCase):
    def test_forward(self) -> None:
        x = Variable(np.array(2.0))
        y: Variable = square(x)
        expected: np.ndarray = np.array(4.0)
        self.assertEqual(y.data, expected)


class ExpTest(unittest.TestCase):
    def test_forward(self) -> None:
        x = Variable(np.array(2.0))
        y: Variable = exp(x)
        expected: np.ndarray = np.array(7.38905609893065022723042746057500)
        self.assertEqual(y.data, expected)


class AddTest(unittest.TestCase):
    def test_forward(self) -> None:
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        y: Variable = add(x0, x1)
        expected: np.ndarray = np.array(5.0)
        self.assertEqual(y.data, expected)


if __name__ == "__main__":
    x0: np.ndarray = Variable(np.array(2))
    x1: np.ndarray = Variable(np.array(3))
    y: Variable = add(x0, x1)
    print(y.data)
    print(flush=True)

    unittest.main()