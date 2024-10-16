import numpy as np
from typing import Callable, Any, Self


class Variable:
    def __init__(self, data: np.ndarray) -> None:
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
        funcs: list[Callable[[Any], Variable]] = [self.creator]
        while funcs:
            f: Callable[[Any], Variable] = funcs.pop()
            x: Variable = f.input
            y: Variable = f.output

            x.grad = f.backward(y.grad)
            if x.creator is not None:
                funcs.append(x.creator)


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

    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)