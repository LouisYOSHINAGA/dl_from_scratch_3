if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from dezero.core import Variable
import dezero.functions as F


if __name__ == "__main__":
    rng: np.random.Generator = np.random.default_rng()


    x1: np.ndarray = rng.random(size=(1, 3, 7, 7))
    col1: np.ndarray = F.im2col(x1, kernel_size=5, stride=1, pad=0, to_matrix=True)
    print(f"{col1.shape=}")

    x2: np.ndarray = rng.random(size=(10, 3, 7, 7))
    kernel_size: tuple[int, int] = (5, 5)
    stride: tuple[int, int] = (1, 1)
    pad: tuple[int, int] = (0, 0)
    col2: np.ndarray = F.im2col(x2, kernel_size, stride, pad, to_matrix=True)
    print(f"{col2.shape=}")


    N, C, H, W = 1, 5, 15, 15
    OC, (KH, KW) = 8, (3, 3)

    x = Variable(rng.standard_normal(size=(N, C, H, W)))
    W: np.ndarray = rng.standard_normal(size=(OC, C, KH, KW))
    y: Variable = F.conv2d_simple(x, W, b=None, stride=1, pad=1)
    y.backward()

    print(f"{y.shape=}")
    print(f"{x.grad.shape=}")