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