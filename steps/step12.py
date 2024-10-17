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
        self.creator: Callable[[Any], Self] = None

    def set_creator(self, func: Callable[[Any], Self]) -> None:
        self.creator = func

    def backward(self) -> None:
        """
            example:
                L = s( r( q(u) + p(t) ) ) )  // L = s(w), w = r(v), v = q(u) + p(t)

                dL/du = dL/dw * dw/dv * dv/du   // dL/du  = s'(w) * r'(v) * q'(u)
                      = w.grad * dw/dv * dv/du  // w.grad = s'(w)
                      = v.grad * dv/du          // v.grad = w.grad * r'(v)
                      = u.grad                  // u.grad = v.grad * q'(u)
                dL/dt = dL/dw * dw/dv * dv/dt   // dL/du  = s'(w) * r'(v) * p'(t)
                      = w.grad * dw/dv * dv/dt  // w.grad = s'(w)
                      = v.grad * dv/dt          // v.grad = w.grad * r'(v)
                      = t.grad                  // t.grad = v.grad * p'(t)

                funcs = [s]
                f = s, x = w, y = L          // L = s(w)
                w.grad = s.backward(L.grad)  // dL/dw = dL/ds * s'(w)

                funcs = [r]
                f = r, x = v, y = w          // w = r(v)
                v.grad = r.backward(w.grad)  // dL/dv = dL/dw * r'(v)

                funcs = [p, q]
                f = q, y = v, x = u          // v = q(u)
                u.grad = q.backward(v.grad)  // dL/du = dL/dv * q'(u)

                funcs = [p]
                f = p, y = v, x = t          // v = p(t)
                u.grad = p.backward(v.grad)  // dL/du = dL/dv * p'(t)
        """
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs: list[Callable[[Any], Variable]] = [self.creator]
        while funcs:
            f: Callable[[Any], Variable] = funcs.pop()
            x: Variable = f.input
            y: Variable = f.output

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

def add(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
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
        expected: np.ndarray = np.array(7.38905609893)
        self.assertTrue(np.allclose(y.data, expected))


class AddTest(unittest.TestCase):
    def test_forward(self) -> None:
        x0 = Variable(np.array(2))
        x1 = Variable(np.array(3))
        y: Variable = add(x0, x1)
        expected: np.ndarray = np.array(5)
        self.assertEqual(y.data, expected)


if __name__ == "__main__":
    unittest.main()

    x0: np.ndarray = Variable(np.array(2))
    x1: np.ndarray = Variable(np.array(3))
    y: Variable = add(x0, x1)
    print(y.data)
