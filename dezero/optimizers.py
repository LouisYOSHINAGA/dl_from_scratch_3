import numpy as np
from dezero import Layer, Parameter
from typing import Callable, Self, Any


class Optimizer:
    def __init__(self) -> None:
        self.target: Layer|None = None
        self.hooks: list[Callable[[list[Parameter]], None]] = []

    def setup(self, target: Layer) -> Self:
        self.target = target
        return self

    def add_hook(self, f: Callable[[list[Parameter]], None]) -> None:
        self.hooks.append(f)

    def update(self) -> None:
        params: list[Parameter] = [p for p in self.target.params() if p.grad is not None]

        # preprocess
        for f in self.hooks:
            f(params)

        # update
        for param in params:
            self.update_one(param)

    def update_one(self, param: Parameter) -> None:
        raise NotImplementedError()


class SGD(Optimizer):
    def __init__(self, lr: float =0.01) -> None:
        super().__init__()
        self.lr = lr

    def update_one(self, param: Parameter) -> None:
        param.data -= self.lr * param.grad.data


class MomentumSGD(Optimizer):
    def __init__(self, lr: float =0.01, momentum: float =0.9) -> None:
        super().__init__()
        self.lr: float = lr
        self.momentum: float = momentum
        self.vs: dict[int, np.ndarray] = {}

    def update_one(self, param: Parameter) -> None:
        v_key: int = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param.data)
        v: np.ndarray = self.momentum * self.vs[v_key] - self.lr * param.grad.data
        param.data += v


class AdaGrad(Optimizer):
    def __init__(self, lr: float =0.01, eps: float =1e-8) -> None:
        super().__init__()
        self.lr: float = lr
        self.eps: float = eps
        self.hs: dict[int, np.ndarray] = {}
        
    def update_one(self, param: Parameter) -> None:
        h_key: int = id(param)
        if h_key not in self.hs:
            self.hs[h_key] = np.zeros_like(param.data)
        self.hs[h_key] += param.grad.data ** 2
        param.data -= self.lr * param.grad.data / (np.sqrt(self.hs[h_key] + self.eps))


class AdaDelta(Optimizer):
    def __init__(self, rho: float =0.95, eps: float =1e-6) -> None:
        super().__init__()
        self.rho: float = rho
        self.eps: float = eps
        self.msg: dict[int, np.ndarray] = {}
        self.msdx: dict[int, np.ndarray] = {}

    def update_one(self, param: Parameter) -> None:
        key: int = id(param)
        if key not in self.msg:
            self.msg[key] = np.zeros_like(param.data)
            self.msdx[key] = np.zeros_like(param.data)
        self.msg[key] = self.rho * self.msg[key] + (1 - self.rho) * param.grad.data ** 2
        dx: float = np.sqrt((self.msdx[key] + self.eps) / (self.msg[key] + self.eps)) * param.grad.data
        param.data -= dx
        self.msdx[key] = self.rho * self.msdx[key] + (1 - self.rho) * dx ** 2


class Adam(Optimizer):
    def __init__(self, alpha: float =0.001, beta1: float =0.9, beta2: float =0.999, eps: float =1e-8) -> None:
        super().__init__()
        self.t: int = 0
        self.alpha: float = alpha
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.eps: float = eps
        self.ms: dict[int, np.ndarray] = {}
        self.vs: dict[int, np.ndarray] = {}

    def update(self, *args: Any, **kwargs: Any) -> None:
        self.t += 1
        super().update(*args, **kwargs)
    
    @property
    def lr(self) -> float:
        fix1: float = 1 - np.power(self.beta1, self.t)
        fix2: float = 1 - np.power(self.beta2, self.t)
        return self.alpha * np.sqrt(fix2) / fix1

    def update_one(self, param: Parameter) -> None:
        key: int = id(param)
        if key not in self.ms:
            self.ms[key] = np.zeros_like(param.data)
            self.vs[key] = np.zeros_like(param.data)
        self.ms[key] = self.beta1 * self.ms[key] + (1 - self.beta1) * param.grad.data
        self.vs[key] = self.beta2 * self.vs[key] + (1 - self.beta2) * param.grad.data ** 2
        param.data -= self.lr * self.ms[key] / (np.sqrt(self.vs[key]) + self.eps)