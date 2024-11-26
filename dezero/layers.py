import numpy as np
import weakref
from dezero.core import Parameter, Variable
import dezero.functions as F
from typing import Any, Generator


class Layer:
    def __init__(self) -> None:
        self._params: set[Parameter] = set()

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs: Variable|np.ndarray) -> list[Variable]|Variable:
        outputs: list[Variable]|Variable = self.forward(*inputs)
        if not isinstance(outputs, list):
            outputs = [outputs]

        self.inputs: list[Variable|np.ndarray] = [weakref.ref(x) for x in inputs]
        self.outputs: list[Variable] = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def params(self) -> Generator[Parameter, None, None]:
        for name in self._params:
            obj: Layer|Parameter = self.__dict__[name]
            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def cleargrads(self) -> None:
        for param in self.params():
            param.cleargrad()


class Linear(Layer):
    def __init__(self, out_size: int, in_size: int|None =None, nobias: bool =False,
                 dtype: np.dtype =np.float32) -> None:
        super().__init__()
        self.in_size: int|None = in_size
        self.out_size: int = out_size
        self.dtype: np.dtype = dtype

        self.W = Parameter(None, name="W")
        if self.in_size is not None:
            self._init_W()

        self.b: Parameter|None = None if nobias \
                                 else Parameter(np.zeros(out_size, dtype=dtype), name="b")

    def _init_W(self) -> None:
        I: int = self.in_size
        O: int = self.out_size
        self.W.data = np.random.default_rng().random(size=(I, O)).astype(self.dtype) * np.sqrt(1/I)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()
        return F.linear(x, self.W, self.b)