if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from dezero import Variable, no_grad
from dezero.datasets import get_spiral
from dezero.models import MLP
import dezero.functions as F
import dezero.optimizers as O
import matplotlib.pyplot as plt


def plot_data(xs: np.ndarray, ys: np.ndarray, ts: np.ndarray, model: MLP|None =None) -> None:
    markers: list[str] = ["x", "o", "^"]
    plt.figure(figsize=(7, 5))
    if model is not None:
        xx, yy = np.meshgrid(np.linspace(-1, 1, 1000), np.linspace(-1 ,1, 1000))
        with no_grad():
            z: Variable = model(np.c_[xx.ravel(), yy.ravel()]).data.argmax(axis=1).reshape(xx.shape)
        plt.contourf(xx, yy, z, cmap=plt.cm.viridis)
    for i in np.unique(ts):
        plt.scatter(xs[ts == i], ys[ts == i], label=i, marker=markers[i])
    plt.show()

def plot_loss(losses: np.ndarray) -> None:
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(losses)
    plt.show()


if __name__ == "__main__":
    rng: np.random.Generator = np.random.default_rng()

    xs, ts = get_spiral(train=True)
    plot_data(xs[:, 0], xs[:, 1], ts)  # fig. 48-1

    hidden_size: int = 10
    lr: float = 1.0
    model = MLP((hidden_size, 3))
    opt = O.SGD(lr).setup(model)

    max_epoch: int = 300
    data_size: int = len(xs)
    batch_size: int = 30
    max_iter: int = data_size // batch_size + int(data_size % batch_size != 0)
    losses: np.ndarray = np.zeros(max_epoch)

    for epoch in range(max_epoch):
        index: np.ndarray = rng.permutation(data_size)
        sum_loss: float = 0

        for i in range(max_iter):
            indexes: np.ndarray = index[i*batch_size:(i+1)*batch_size]
            loss: Variable = F.softmax_cross_entropy(model(xs[indexes]), ts[indexes])
            model.cleargrads()
            loss.backward()
            opt.update()
            sum_loss += float(loss.data) * batch_size

        print(f"epoch {epoch+1}, loss {sum_loss/data_size:.2f}")
        losses[epoch] = sum_loss

    plot_loss(losses)  # fig. 48-2

    xs, ts = get_spiral(train=False)
    plot_data(xs[:, 0], xs[:, 1], ts, model)  # fig. 48-3