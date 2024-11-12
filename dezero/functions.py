import numpy as np
from dezero.core import Variable, Function


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