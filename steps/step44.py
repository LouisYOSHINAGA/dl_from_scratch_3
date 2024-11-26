if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from dezero.core import Variable
import dezero.functions as F
import dezero.layers as L


if __name__ == "__main__":
    rng: np.random.Generator = np.random.default_rng(0)
    x: np.ndarray = rng.random(size=(100, 1))
    y: np.ndarray = np.sin(2 * np.pi * x) + rng.random(size=(100, 1))

    l1 = L.Linear(10)
    l2 = L.Linear(1)

    lr: float = 0.2
    iters: int = 10000
    for i in range(iters):
        y_pred: Variable = l2(F.sigmoid(l1(x)))
        loss: Variable = F.mean_squared_error(y, y_pred)

        l1.cleargrads()
        l2.cleargrads()
        loss.backward()

        for l in [l1, l2]:
            for p in l.params():
                p.data -= lr * p.grad.data

        if i % 1000 == 0:
            print(f"{loss=}")