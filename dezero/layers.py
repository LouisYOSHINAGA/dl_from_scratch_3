import os
import numpy as np
import weakref
from dezero import Parameter, Variable
import dezero.functions as F
from dezero.cuda import xpy, xpndarray
from typing import Any, Generator


class Layer:
    def __init__(self) -> None:
        self._params: set[Parameter|Layer] = set()

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs: Variable|xpndarray) -> list[Variable]|Variable:
        outputs: list[Variable]|Variable = self.forward(*inputs)
        if not isinstance(outputs, list):
            outputs = [outputs]

        self.inputs: list[Variable|xpndarray] = [weakref.ref(x) for x in inputs]
        self.outputs: list[Variable] = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs: xpndarray) -> xpndarray:
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

    def to_cpu(self) -> None:
        for param in self.params():
            param.to_cpu()

    def to_gpu(self) -> None:
        for param in self.params():
            param.to_gpu()

    def _flatten_params(self, params_dict: dict[str, Parameter], parent_key: str ="") -> None:
        for name in self._params:
            obj: Parameter|Layer = self.__dict__[name]
            key: str = f"{parent_key}/{name}" if parent_key else name
            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def load_weights(self, path: str) -> None:
        npz: dict[str, Parameter] = np.load(path)
        params_dict: dict[str, Parameter] = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]

    def save_weights(self, path: str) -> None:
        self.to_cpu()

        params_dict: dict[str, Parameter] = {}
        self._flatten_params(params_dict)
        array_dict: dict[str, Parameter] = {
            key: param.data for key, param in params_dict.items() if param is not None
        }

        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise


class Linear(Layer):
    def __init__(self, out_size: int, in_size: int|None =None, nobias: bool =False,
                 dtype: type =np.float32) -> None:
        super().__init__()
        self.in_size: int|None = in_size
        self.out_size: int = out_size
        self.dtype: xpy.dtype = dtype

        self.W = Parameter(None, name="W")
        if self.in_size is not None:
            self._init_W()

        self.b: Parameter|None = None if nobias \
                                 else Parameter(xpy.zeros(out_size, dtype=dtype), name="b")

    def _init_W(self) -> None:
        I: int = self.in_size
        O: int = self.out_size
        self.W.data = xpy.array(np.random.default_rng().random(size=(I, O)).astype(self.dtype)) \
                    * xpy.sqrt(1/I)

    def forward(self, x: xpndarray) -> xpndarray:
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()
        return F.linear(x, self.W, self.b)