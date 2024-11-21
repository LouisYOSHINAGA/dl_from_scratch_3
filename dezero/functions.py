import numpy as np
from dezero.core import as_variable, Variable, Function
from dezero import utils


class Sin(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.sin(x)

    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        return gy * cos(x)

def sin(x: Variable) -> Variable:
    return Sin()(x)


class Cos(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.cos(x)

    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        return - gy * sin(x)

def cos(x: Variable) -> Variable:
    return Cos()(x)


class Tanh(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def backward(self, gy: Variable) -> Variable:
        y: Variable = self.outputs[0]()
        return gy * (1 - y**2)

def tanh(x: Variable) -> Variable:
    return Tanh()(x)


class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    def backward(self, gy: Variable) -> Variable:
        return gy * self.outputs[0]()

def exp(x: Variable) -> Variable:
    return Exp()(x)


class Log(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.log(x)
    
    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        return gy / x

def log(x: Variable) -> Variable:
    return Log()(x)


class Reshape(Function):
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape: tuple[int, ...] = shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape: tuple[int, ...] = x.shape
        return x.reshape(self.shape)

    def backward(self, gy: Variable) -> Variable:
        return reshape(gy, self.x_shape)

def reshape(x: Variable, shape: tuple[int, ...]) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, axes: tuple[int, ...]|None =None) -> None:
        self.axes: tuple[int, ...]|None = axes

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x.transpose(self.axes)

    def backward(self, gy: Variable) -> Variable:
        if self.axes is None:
            return transpose(gy)
        inv_axes: tuple[int, ...] = tuple(
            np.argsort([ax % len(self.axes) for ax in self.axes])
        )
        return transpose(gy, inv_axes)

def transpose(x: Variable, axes: tuple[int, ...]|None =None) -> Variable:
    return Transpose(axes)(x)


class Sum(Function):
    def __init__(self, axis: int|None, keepdims: bool) -> None:
        self.axis: int|None = axis
        self.keepdims: bool = keepdims

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape: tuple[int, ...] = x.shape
        return x.sum(axis=self.axis, keepdims=self.keepdims)

    def backward(self, gy: Variable) -> Variable:
        gy: Variable = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        return broadcast_to(gy, self.x_shape)

def sum(x: Variable, axis: int|None =None, keepdims: bool =False) -> Variable:
    return Sum(axis, keepdims)(x)


class BroadCastTo(Function):
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape: tuple[int, ...] = shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape: tuple[int, ...] = x.shape
        return np.broadcast_to(x, self.shape)

    def backward(self, gy: Variable) -> Variable:
        return sum_to(gy, self.x_shape)

def broadcast_to(x: Variable, shape: tuple[int, ...]) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return BroadCastTo(shape)(x)


class SumTo(Function):
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape: tuple[int, ...] = shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape: tuple[int, ...] = x.shape
        return utils.sum_to(x, self.shape)

    def backward(self, gy: Variable) -> Variable:
        return broadcast_to(gy, self.x_shape)

def sum_to(x: Variable, shape: tuple[int, ...]) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class MatMul(Function):
    def forward(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        return x.dot(W)

    def backward(self, gy: Variable) -> list[Variable]:
        x, W = self.inputs
        gx: Variable = matmul(gy, W.T)
        gW: Variable = matmul(x.T, gy)
        return [gx, gW]

def matmul(x: Variable, W: Variable) -> Variable:
    return MatMul()(x, W)


class MeanSquaredError(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        diff: Variable = x0 - x1
        return (diff ** 2).sum() / len(diff)

    def backward(self, gy: Variable) -> list[Variable]:
        x0, x1 = self.inputs
        diff: Variable = x0 - x1
        gy: Variable = broadcast_to(gy, diff.shape)
        gx0: Variable = gy * diff * (2 / len(diff))
        gx1: Variable = - gx0
        return [gx0, gx1]

def mean_squared_error(x0: Variable, x1: Variable) -> Variable:
    return MeanSquaredError()(x0, x1)


class Linear(Function):
    def forward(self, x: np.ndarray, W: np.ndarray, b: np.ndarray|None) -> np.ndarray:
        y: np.ndarray = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy: Variable) -> list[Variable]:
        x, W, b = self.inputs
        gb: Variable|None = None if b.data is None else sum_to(gy, b.shape)
        gx: Variable = matmul(gy, W.T)
        gW: Variable = matmul(x.T, gy)
        return [gx, gW, gb]

def linear(x: Variable, W: Variable, b: Variable|None) -> Variable:
    return Linear()(x, W, b)

def linear_simple(x: np.ndarray|Variable, W: np.ndarray|Variable, b: np.ndarray|Variable|None =None) -> Variable:
    x = as_variable(x)
    W = as_variable(W)
    t: Variable = matmul(x, W)
    if b is None:
        return t
    y: Variable = t + b
    t.data = None
    return y


class Sigmoid(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 0.5 + 0.5 * np.tanh(0.5 * x) 

    def backward(self, gy: Variable) -> Variable:
        y: Variable = self.outputs[0]()
        return gy * y * (1 - y)

def sigmoid(x: Variable) -> Variable:
    return Sigmoid()(x)

def sigmoid_simple(x: np.ndarray|Variable) -> Variable:
    x = as_variable(x)
    return 1 / (1 + exp(-x))