if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from dezero import Variable, Model
import dezero.layers as L
import dezero.functions as F
from dezero.models import MLP


class TwoLayerNet(Model):
    def __init__(self, hidden_size: int, out_size: int) -> None:
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x: Variable) -> Variable:
        y: Variable = F.sigmoid(self.l1(x))
        return self.l2(y)


if __name__ == "__main__":
    model = MLP((100, 10))
    x: np.ndarray = Variable(np.random.default_rng().random(size=(5, 10)), name="x")
    model.plot(x, to_file="step45_model.png")


    rng: np.random.Generator = np.random.default_rng(0)
    x: np.ndarray = rng.random(size=(100, 1))
    y: np.ndarray = np.sin(2 * np.pi * x) + rng.random(size=(100, 1))

    hidden_size: int = 10
    # model = TwoLayerNet(hidden_size, 1)
    model = MLP((hidden_size, 1))

    lr: float = 0.2
    max_iter: int = 10000
    for i in range(max_iter):
        y_pred: Variable = model(x)
        loss: Variable = F.mean_squared_error(y, y_pred)

        model.cleargrads()
        loss.backward()

        for p in model.params():
            p.data -= lr * p.grad.data

        if i % 1000 == 0:
            print(f"{loss=}")