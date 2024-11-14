if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from dezero import Variable
import dezero.functions as F


if __name__ == "__main__":
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y: Variable = F.reshape(x, (6, ))
    y.backward(retain_grad=True)
    print(f"{x.grad=}")
    print()


    x = Variable(np.random.default_rng().random((1, 2, 3)))
    print(f"{x=}")
    y: Variable = x.reshape([2, 3])
    print(f"{y=}")
    z: Variable = x.reshape(2, 3)
    print(f"{z=}")
    print()


    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y: Variable = F.transpose(x)
    y.backward()
    print(f"{x.grad=}")
    print()


    x = Variable(np.random.default_rng().random((2, 3)))
    print(f"{x=}")
    y: Variable = x.transpose()
    print(f"{y=}")
    z: Variable = x.T
    print(f"{z=}")