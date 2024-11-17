import numpy as np
from dezero.core import as_variable, Variable, Function
from dezero import utils
from typing import Any


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


class Reshape(Function):
    def __init__(self, shape: tuple[Any]) -> None:
        self.shape: tuple[Any] = shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape: tuple[Any] = x.shape
        return x.reshape(self.shape)

    def backward(self, gy: Variable) -> Variable:
        return reshape(gy, self.x_shape)

def reshape(x: Variable, shape: tuple[Any]) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, axes: tuple[Any]|None =None) -> None:
        self.axes: tuple[Any]|None = axes

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x.transpose(self.axes)

    def backward(self, gy: Variable) -> Variable:
        if self.axes is None:
            return transpose(gy)
        inv_axes: tuple[Any] = tuple(
            np.argsort([ax % len(self.axes) for ax in self.axes])
        )
        return transpose(gy, inv_axes)

def transpose(x: Variable, axes: tuple[Any]|None =None) -> Variable:
    return Transpose(axes)(x)


class Sum(Function):
    def __init__(self, axis: int|None, keepdims: bool) -> None:
        self.axis: int|None = axis
        self.keepdims: bool = keepdims

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape: tuple[Any] = x.shape
        return x.sum(axis=self.axis, keepdims=self.keepdims)

    def backward(self, gy: Variable) -> Variable:
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepsdims)
        return broadcast_to(gy, self.x_shape)

def sum(x: Variable, axis: int|None =None, keepdims: bool =False) -> Variable:
    return Sum(axis, keepdims)(x)


class BroadCastTo(Function):
    def __init__(self, shape: tuple[Any]) -> None:
        self.shape: tuple[Any] = shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape: tuple[Any] = x.shape
        return np.broadcast_to(x, self.shape)

    def backward(self, gy: Variable) -> Variable:
        return sum_to(gy, self.x_shape)

def broadcast_to(x: Variable, shape: tuple[Any]) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return BroadCastTo(shape)(x)


class SumTo(Function):
    def __init__(self, shape: tuple[Any]) -> None:
        self.shape: tuple[Any] = shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape: tuple[Any] = x.shape
        return utils.sum_to(x, self.shape)

    def backward(self, gy: Variable) -> Variable:
        return broadcast_to(gy, self.x_shape)

def sum_to(x: Variable, shape: tuple[Any]) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)