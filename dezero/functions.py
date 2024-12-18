import numpy as np
import dezero
from dezero.core import as_array, as_variable, Variable, Function
from dezero import utils
from dezero.cuda import xpy, xpndarray


class Sin(Function):
    def forward(self, x: xpndarray) -> xpndarray:
        return xpy.sin(x)

    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        return gy * cos(x)

def sin(x: Variable) -> Variable:
    return Sin()(x)


class Cos(Function):
    def forward(self, x: xpndarray) -> xpndarray:
        return xpy.cos(x)

    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        return - gy * sin(x)

def cos(x: Variable) -> Variable:
    return Cos()(x)


class Tanh(Function):
    def forward(self, x: xpndarray) -> xpndarray:
        return xpy.tanh(x)

    def backward(self, gy: Variable) -> Variable:
        y: Variable = self.outputs[0]()
        return gy * (1 - y**2)

def tanh(x: Variable) -> Variable:
    return Tanh()(x)


class Exp(Function):
    def forward(self, x: xpndarray) -> xpndarray:
        return xpy.exp(x)

    def backward(self, gy: Variable) -> Variable:
        return gy * self.outputs[0]()

def exp(x: Variable) -> Variable:
    return Exp()(x)


class Log(Function):
    def forward(self, x: xpndarray) -> xpndarray:
        return xpy.log(x)

    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        return gy / x

def log(x: Variable) -> Variable:
    return Log()(x)


class Reshape(Function):
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape: tuple[int, ...] = shape

    def forward(self, x: xpndarray) -> xpndarray:
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

    def forward(self, x: xpndarray) -> xpndarray:
        return x.transpose(self.axes)

    def backward(self, gy: Variable) -> Variable:
        if self.axes is None:
            return transpose(gy)
        inv_axes: tuple[int, ...] = tuple(
            xpy.argsort([ax % len(self.axes) for ax in self.axes])
        )
        return transpose(gy, inv_axes)

def transpose(x: Variable, axes: tuple[int, ...]|None =None) -> Variable:
    return Transpose(axes)(x)


class Sum(Function):
    def __init__(self, axis: int|None, keepdims: bool) -> None:
        self.axis: int|None = axis
        self.keepdims: bool = keepdims

    def forward(self, x: xpndarray) -> xpndarray:
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

    def forward(self, x: xpndarray) -> xpndarray:
        self.x_shape: tuple[int, ...] = x.shape
        return xpy.broadcast_to(x, self.shape)

    def backward(self, gy: Variable) -> Variable:
        return sum_to(gy, self.x_shape)

def broadcast_to(x: Variable, shape: tuple[int, ...]) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return BroadCastTo(shape)(x)


class SumTo(Function):
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape: tuple[int, ...] = shape

    def forward(self, x: xpndarray) -> xpndarray:
        self.x_shape: tuple[int, ...] = x.shape
        return utils.sum_to(x, self.shape)

    def backward(self, gy: Variable) -> Variable:
        return broadcast_to(gy, self.x_shape)

def sum_to(x: Variable, shape: tuple[int, ...]) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class MatMul(Function):
    def forward(self, x: xpndarray, W: xpndarray) -> xpndarray:
        return x.dot(W)

    def backward(self, gy: Variable) -> list[Variable]:
        x, W = self.inputs
        gx: Variable = matmul(gy, W.T)
        gW: Variable = matmul(x.T, gy)
        return [gx, gW]

def matmul(x: Variable, W: Variable) -> Variable:
    return MatMul()(x, W)


class Linear(Function):
    def forward(self, x: xpndarray, W: xpndarray, b: xpndarray|None) -> xpndarray:
        y: xpndarray = x.dot(W)
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

def linear_simple(x: xpndarray|Variable, W: xpndarray|Variable, b: xpndarray|Variable|None =None) -> Variable:
    x = as_variable(x)
    W = as_variable(W)
    t: Variable = matmul(x, W)
    if b is None:
        return t
    y: Variable = t + b
    t.data = None
    return y


class Sigmoid(Function):
    def forward(self, x: xpndarray) -> xpndarray:
        return 0.5 + 0.5 * xpy.tanh(0.5 * x)

    def backward(self, gy: Variable) -> Variable:
        y: Variable = self.outputs[0]()
        return gy * y * (1 - y)

def sigmoid(x: Variable) -> Variable:
    return Sigmoid()(x)

def sigmoid_simple(x: xpndarray|Variable) -> Variable:
    x = as_variable(x)
    return 1 / (1 + exp(-x))


class Softmax(Function):
    def __init__(self, axis: int =1) -> None:
        self.axis: int = axis

    def forward(self, x: xpndarray) -> xpndarray:
        y: Variable = x - x.max(axis=self.axis, keepdims=True)
        y = xpy.exp(y)
        return y / y.sum(axis=self.axis, keepdims=True)

def softmax(x: Variable, axis: int =1) -> Variable:
    return Softmax(axis)(x)

def softmax_simple(x: xpndarray|Variable, axis: int =1) -> Variable:
    x = as_variable(x)
    y: Variable = exp(x)
    return y / sum(y, axis=axis, keepdims=True)


class ReLU(Function):
    def forward(self, x: xpndarray) -> xpndarray:
        return xpy.maximum(x, 0.0)

    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        return gy * (x.data > 0)

def relu(x: Variable) -> Variable:
    return ReLU()(x)


class MeanSquaredError(Function):
    def forward(self, x0: xpndarray, x1: xpndarray) -> xpndarray:
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


class SoftmaxCrossEntropy(Function):
    def forward(self, x: xpndarray, t: xpndarray) -> xpndarray:
        N: int = x.shape[0]
        log_p: Variable = x - utils.logsumexp(x, axis=1)
        log_p = log_p[xpy.arange(N), t.ravel()]
        return -log_p.sum() / xpy.float32(N)

    def backward(self, gy: Variable) -> Variable:
        x, t = self.inputs
        N, CLS_NUM = x.shape
        gy /= N
        y: Variable = softmax(x)
        t_onehot: xpndarray = xpy.eye(CLS_NUM, dtype=t.dtype)[t.data]
        return (y - t_onehot) * gy

def softmax_cross_entropy(x: Variable, t: Variable) -> float:
    return SoftmaxCrossEntropy()(x, t)

def softmax_cross_entropy_simple(x: xpndarray|Variable, t: xpndarray|Variable) -> float:
    x = as_variable(x)
    t = as_variable(t)
    N: int = x.shape[0]
    p: Variable = clip(softmax_simple(x), 1e-15, 1.0)
    log_p: Variable = log(p)
    tlog_q: Variable = log_p[xpy.arange(N), t.data]
    return - sum(tlog_q) / N


class GetItem(Function):
    def __init__(self, slices: int|tuple[int, ...]) -> None:
        self.slices: int|tuple[int, ...] = slices

    def forward(self, x: xpndarray) -> xpndarray:
        return x[self.slices]

    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        return GetItemGrad(self.slices, x.shape)(gy)

def get_item(x: Variable, slices: int|tuple[int, ...]) -> Variable:
    return GetItem(slices)(x)


class GetItemGrad(Function):
    def __init__(self, slices: int|tuple[int, ...], in_shape: int|tuple[int, ...]) -> None:
        self.slices: int|tuple[int, ...] = slices
        self.in_shape: int|tuple[int, ...] = in_shape

    def forward(self, gy: Variable) -> Variable:
        gx: xpndarray = xpy.zeros(self.in_shape)
        xpy.add.at(gx, self.slices, gy)
        return gx

    def backward(self, ggx: Variable) -> Variable:
        return get_item(ggx, self.slices)


class Clip(Function):
    def __init__(self, x_min: float, x_max: float) -> None:
        self.x_min: float = x_min
        self.x_max: float = x_max

    def forward(self, x: xpndarray) -> xpndarray:
        return xpy.clip(x, self.x_min, self.x_max)

    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        mask: bool = (self.x_min <= x) * (x.data <= self.x_max)
        return gy * mask

def clip(x: Variable, x_min: float, x_max: float) -> Variable:
    return Clip(x_min, x_max)(x)


def accuracy(ys: xpndarray|Variable, ts: xpndarray|Variable) -> Variable:
    ys = as_variable(ys)
    ts = as_variable(ts)
    pred: xpndarray = ys.data.argmax(axis=1).reshape(ts.shape)
    acc: float = (pred == ts.data).mean()
    return Variable(as_array(acc))


def dropout(x: Variable, dropout_ratio: float =0.5) -> Variable:
    x: Variable = as_variable(x)
    if not dezero.Config.train:
        return x
    mask: xpndarray = xpy.array(np.random.default_rng().random(*x.shape) > dropout_ratio)
    scale: xpndarray = xpy.array(1 - dropout_ratio).astype(x.dtype)
    y: Variable = x * mask / scale
    return y