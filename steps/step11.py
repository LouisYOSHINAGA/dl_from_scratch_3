import numpy as np
from typing import Callable, Any, Self


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
    def __call__(self, inputs: list[Variable]) -> list[Variable]:
        xs: list[np.ndarray] = [x.data for x in inputs]
        ys: list[np.ndarray] = self.forward(xs)

        outputs: list[Variable] = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)

        self.inputs: list[Variable] = inputs
        self.outputs: list[Variable] = outputs
        return outputs

    def forward(self, xs: list[np.ndarray]) -> list[np.ndarray]:
        raise NotImplementedError()

    def backward(self, gys: list[np.ndarray]) -> list[np.ndarray]:
        raise NotImplementedError()


class Add(Function):
    def forward(self, xs: list[np.ndarray]) -> list[np.ndarray]:
        x0, x1 = xs
        y: np.ndarray = x0 + x1
        return [y]


if __name__ == "__main__":
    f: Callable[[list[np.ndarray]], list[np.ndarray]] = Add()
    xs: list[Variable] = [Variable(np.array(2)), Variable(np.array(3))]
    ys: list[Variable] = f(xs)
    y: Variable = ys[0]
    print(y.data)