if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from dezero import Variable
from dezero.models import MLP
import dezero.optimizers as O
import dezero.functions as F


if __name__ == "__main__":
    rng: np.random.Generator = np.random.default_rng(0)
    x: np.ndarray = rng.random(size=(100, 1))
    y: np.ndarray = np.sin(3 * np.pi * x) + rng.random(size=(100, 1))

    hidden_size: int = 10
    lr: float = 0.2
    model = MLP((hidden_size, 1))
    # optimizer = O.SGD(lr).setup(model)
    optimizer = O.MomentumSGD(lr).setup(model)

    max_iter: int = 10000
    for i in range(max_iter):
        y_pred: Variable = model(x)
        loss: Variable = F.mean_squared_error(y, y_pred)

        model.cleargrads()
        loss.backward()
        optimizer.update()

        if i % 1000 == 0:
            print(f"{loss=}")