from __future__ import annotations
import numpy as np
import weakref
import contextlib
from typing import TypeAlias, Generator, Callable, Any, Self

Scalar: TypeAlias = int | float | np.ndarray


class Config:
    enable_backprop: bool = True

@contextlib.contextmanager
def using_config(name: str, value: bool) -> Generator[None, None, None]:
    old_value: bool = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad() -> None:
    return using_config("enable_backprop", False)


class Variable:
    __array_priority__ = 200

    def __init__(self, data: np.ndarray, name: str|None =None) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported.")
        self.name: str = name
        self.data: np.ndarray = data
        self.grad: np.ndarray|None = None
        self.creator: Callable[[Any], Self]|None = None
        self.generation: int = 0

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        if self.data is None:
            return "Variable(None)"
        p: str = str(self.data).replace("\n", f"\n{' '*9}")
        return f"Variable({p})"

    def set_creator(self, func: Callable[[Any], Self]) -> None:
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad: bool =False) -> None:
        """
            example:
                L = r( add(q(t), p(s)) )  // L = r(w), w = v + u, v = q(t), u = p(s)

                dL/dt = dL/dw * dw/dv * dv/dt  // dL/dt = r'(w) * add'(v) * q'(t)
                      = w.grad * dv/dt         // w.grad = r'(w) * add'(v)
                      = t.grad                 // t.grad = r'(w) * add'(v) * q'(t)
                dL/dt = dL/dw * dw/dv * dv/ds  // dL/ds = r'(w) * add'(u) * p'(s)
                      = w.grad * dv/ds         // w.grad = r'(w) * add'(u)
                      = s.grad                 // s.grad = r'(w) * add'(u) * p'(s)

                funcs = [r]
                f = r, f.outputs = [L], f.inputs = [w]  // L = r(w)
                gys = [L.grad]                          // [dL/dL]
                gxs = [r.backward(L.grad)]              // [dL/dL * dL/dw]
                w.grad = r.backward(L.grad)             // dL/dw = dL/dL * dL/dw

                funcs = [add]
                f = add, f.outputs = [w], f.inputs = [v, u]  // w = v + u
                gys = [w.grad]                               // [dL/dw]
                gxs = [add.backward(w.grad)]                 // [dL/dw * dw/dv, dL/dw * dw/du]
                v.grad = add.backward(w.grad)                // dL/dv = dL/dw * dw/dv
                u.grad = add.backward(w.grad)                // dL/du = dL/dw * dw/du

                funcs = [q, p]
                f = q, f.outputs = [v], f.inputs = [t]  // v = q(t)
                gys = [v.grad]                          // [dL/dv]
                gxs = [q.backward(v.grad)               // [dL/dv * dv/dt]
                t.grad = q.backward(v.grad)             // dL/dt = dL/dv * dv/dt

                funcs = [p]
                f = p, f.outputs = [u], f.inputs = [s]  // u = p(s)
                gys = [u.grad]                          // [dL/du]
                gxs = [p.backward(u.grad)]              // [dL/du * du/ds]
                s.grad = p.backward(u.grad)             // dL/ds = dL/du * du/ds
        """
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs: list[Callable[[Any], Self]] = []
        seen_set: set[Callable[[Any], Self]] = set()

        def add_func(f: Callable[[Any], Self]) -> None:
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f: Callable[[Any], Self] = funcs.pop()
            gys: list[np.ndarray] = [output().grad for output in f.outputs]
            gxs: list[np.ndarray]|np.ndarray = f.backward(*gys)
            if not isinstance(gxs, list):
                gxs = [gxs]

            for x, gx in zip(f.inputs, gxs):
                x.grad = gx if x.grad is None else x.grad + gx
                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    def cleargrad(self) -> None:
        self.grad = None

def as_array(x: Scalar|np.ndarray) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs: Variable|np.ndarray) -> list[Variable]|Variable:
        vinputs: list[Variable] = [as_variable(x) for x in inputs]
        xs: list[np.ndarray] = [x.data for x in vinputs]
        ys: list[np.ndarray]|np.ndarray = self.forward(*xs)
        if not isinstance(ys, list):
            ys = [ys]

        outputs: list[Variable] = [Variable(as_array(y)) for y in ys]
        if Config.enable_backprop:
            self.generation: int = max([x.generation for x in vinputs])
            self.inputs: list[Variable] = vinputs
            self.outputs: list[Variable] = [weakref.ref(output) for output in outputs]
            for output in outputs:
                output.set_creator(self)
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs: list[np.ndarray]) -> list[np.ndarray]:
        raise NotImplementedError()

    def backward(self, gys: list[np.ndarray]) -> list[np.ndarray]:
        raise NotImplementedError()

def as_variable(obj: Variable|np.ndarray) -> Variable:
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Neg(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return -x

    def backward(self, gy: np.ndarray) -> np.ndarray:
        return -gy

def neg(x: Variable) -> Variable:
    return Neg()(x)


class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x ** 2

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x: np.ndarray = self.inputs[0].data
        return 2 * x * gy

def square(x: Variable) -> Variable:
    return Square()(x)


class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    def backward(self, gy):
        x: np.ndarray = self.inputs[0].data
        return np.exp(x) * gy

def exp(x: Variable) -> Variable:
    return Exp()(x)


class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 + x1

    def backward(self, gy: np.ndarray) -> list[np.ndarray]:
        return [gy, gy]

def add(x0: Variable, x1: Scalar|Variable) -> Variable:
    x1: np.ndarray = as_array(x1)
    return Add()(x0, x1)


class Sub(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 - x1

    def backward(self, gy: np.ndarray) -> list[np.ndarray]:
        return [gy, -gy]

def sub(x0: Variable, x1: Scalar|Variable) -> Variable:
    x1: np.ndarray = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0: Variable, x1: Scalar|Variable) -> Variable:
    x1: np.ndarray = as_array(x1)
    return Sub()(x1, x0)


class Mul(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 * x1

    def backward(self, gy: np.ndarray) -> list[np.ndarray]:
        x0: np.ndarray = self.inputs[0].data
        x1: np.ndarray = self.inputs[1].data
        return [gy * x1, gy * x0]

def mul(x0: Variable, x1: Scalar|Variable) -> Variable:
    x1: np.ndarray = as_array(x1)
    return Mul()(x0, x1)


class Div(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 / x1

    def backward(self, gy: np.ndarray) -> list[np.ndarray]:
        x0: np.ndarray = self.inputs[0].data
        x1: np.ndarray = self.inputs[1].data
        return [gy/x1, gy*(-x0/x1**2)]

def div(x0: Variable, x1: Scalar|Variable) -> Variable:
    x1: np.ndarray = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0: Variable, x1: Scalar|Variable) -> Variable:
    x1: np.ndarray = as_array(x1)
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c: Scalar) -> None:
        self.c = c

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x ** self.c

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x: np.ndarray = self.inputs[0].data
        return gy * self.c * x ** (self.c - 1)

def pow(x: Variable, c: Scalar) -> Variable:
    return Pow(c)(x)


def setup_variable() -> None:
    Variable.__neg__ = neg
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow