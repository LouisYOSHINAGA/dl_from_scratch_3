if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Layer, Variable, Model
import dezero.layers as L
import dezero.functions as F
from dezero.models import MLP


def predict(model, x):
    y = model.l1(x)
    y = F.sigmoid(y)
    y = model.l2(y)
    return y


class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y


if __name__ == "__main__":
    model = Layer()
    model.l1 = L.Linear(5)
    model.l2 = L.Linear(3)

    for p in model.params():
        print(p)

    model.cleargrads()


    x = Variable(np.random.randn(5, 10), name="x")
    model = TwoLayerNet(100, 10)
    model.plot(x, to_file="step45_model.png")


    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

    lr = 0.2
    max_iter = 10000
    hidden_size = 10

    model = TwoLayerNet(hidden_size, 1)

    for i in range(max_iter):
        y_pred = model(x)
        loss = F.mean_squared_error(y, y_pred)

        model.cleargrads()
        loss.backward()

        for p in model.params():
            p.data -= lr * p.grad.data

        if i % 1000 == 0:
            print(loss)


    model = MLP((10, 1))
    model = MLP((10, 20, 30, 40, 1))