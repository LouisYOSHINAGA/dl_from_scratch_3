import numpy as np
from dezero import Layer, Parameter
from typing import Callable, Self


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
