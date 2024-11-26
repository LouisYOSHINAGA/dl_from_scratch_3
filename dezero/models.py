import numpy as np
from dezero import Variable, Layer
from dezero import utils
import dezero.functions as F
import dezero.layers as L
from typing import Callable


class Model(Layer):
    def plot(self, *inputs: Variable|np.ndarray, to_file: str ="model.png") -> None:
        y: list[Variable]|Variable = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)


class MLP(Model):
    def __init__(self, fc_output_sizes: list[int],
                 activation: Callable[[Variable], Variable] =F.sigmoid) -> None:
        super().__init__()
        self.activation: Callable[[Variable], Variable] = activation
        self.layers: list[Layer] = []
        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, f"l{i}", layer)
            self.layers.append(layer)

    def forward(self, x: Variable) -> Variable:
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)