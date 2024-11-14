import numpy as np
from dezero.core import as_variable, Variable, Function
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