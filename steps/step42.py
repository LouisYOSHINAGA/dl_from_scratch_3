if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from dezero import Variable
import dezero.functions as F
import matplotlib.pyplot as plt


def gen_data(shape: tuple[int, ...], seed: int =0) -> tuple[Variable, Variable]:
    rng: np.random.Generator = np.random.default_rng(seed)
    x: np.ndarray = rng.random(size=shape)
    y: np.ndarray = 2 * x + 5 + rng.random(size=shape)
    return x, y

def predict(x: Variable, W: Variable, b: Variable) -> Variable:
    return F.matmul(x, W) + b

def mean_squared_error(x0: Variable, x1: Variable) -> Variable:
    diff: Variable = x0 - x1
    return F.sum(diff ** 2) / len(diff)

def plot(x: np.ndarray, y: np.ndarray, W: Variable|None =None, b: Variable|None =None) -> None:
    plt.figure(figsize=(8, 6))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(x, y, color=plt.get_cmap("tab10")(0))
    if W is not None and b is not None:
        xs: Variable = Variable(np.arange(0.0, 1.0, 0.01)[:, np.newaxis])
        ys: Variable = predict(xs, W, b)
        plt.plot(xs.data, ys.data, c=plt.get_cmap("tab10")(1))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    x, y = gen_data(shape=(100, 1))
    plot(x, y)  # fig. 42-1

    W = Variable(np.zeros((1, 1)))
    b = Variable(np.zeros(1))

    lr: float = 0.1
    iters: int = 100
    for i in range(iters):
        y_pred: Variable = predict(x, W, b)
        # loss: Variable = mean_squared_error(y, y_pred)
        loss: Variable = F.mean_squared_error(y, y_pred)

        W.cleargrad()
        b.cleargrad()
        loss.backward()

        W.data -= lr * W.grad.data
        b.data -= lr * b.grad.data

        if i % 10 == 0:
            print(f"{W=}, {b=}, {loss=}")

    plot(x, y, W, b)  # fig. 42-5