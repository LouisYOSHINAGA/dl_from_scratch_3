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
    y: np.ndarray = np.sin(2 * np.pi * x) + rng.random(size=shape)
    return x, y

def predict(x: Variable, W1: Variable, b1: Variable, W2: Variable, b2: Variable) -> Variable:
    y: Variable = F.linear(x, W1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, W2, b2)
    return y

def plot(x: np.ndarray, y: np.ndarray,
         W1: Variable|None =None, b1: Variable|None =None, W2: Variable|None =None, b2: Variable|None =None) -> None:
    plt.figure(figsize=(8, 6))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(x, y, color=plt.get_cmap("tab10")(0))
    if W1 is not None and b1 is not None and W2 is not None and b2 is not None:
        xs: Variable = Variable(np.arange(0.0, 1.0, 0.01)[:, np.newaxis])
        ys: Variable = predict(xs, W1, b1, W2, b2)
        plt.plot(xs.data, ys.data, c=plt.get_cmap("tab10")(1))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    x, y = gen_data(shape=(100, 1))
    plot(x, y)  # fig. 43-2

    in_size: int = 1
    hidden_size: int = 10
    out_size: int =1
    rng: np.random.Generator = np.random.default_rng()
    W1 = Variable(0.01 * rng.normal(size=(in_size, hidden_size)))
    b1 = Variable(np.zeros(hidden_size))
    W2 = Variable(0.01 * rng.normal(size=(hidden_size, out_size)))
    b2 = Variable(np.zeros(out_size))

    lr: float = 0.2
    iters: int = 10000
    for i in range(iters):
        y_pred: Variable = predict(x, W1, b1, W2, b2)
        loss: Variable = F.mean_squared_error(y, y_pred)

        W1.cleargrad()
        b1.cleargrad()
        W2.cleargrad()
        b2.cleargrad()
        loss.backward()

        W1.data -= lr * W1.grad.data
        b1.data -= lr * b1.grad.data
        W2.data -= lr * W2.grad.data
        b2.data -= lr * b2.grad.data

        if i % 1000 == 0:
            print(f"{loss=}")

    plot(x, y, W1, b1, W2, b2)  # fig. 43-4