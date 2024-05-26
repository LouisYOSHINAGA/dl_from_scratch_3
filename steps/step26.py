if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph
from dezero.utils import _dot_var, _dot_func


def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) \
      * (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z


if __name__ == "__main__":
    x = Variable(np.random.randn(2, 3))
    x.name = "x"
    print(_dot_var(x))
    print(_dot_var(x, verbose=True))

    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(1.0))
    y = x0 + x1
    txt = _dot_func(y.creator)
    print(txt)


    x = Variable(np.array(1.0))
    y = Variable(np.array(1.0))
    z = goldstein(x, y)
    z.backward()

    x.name = "x"
    y.name = "y"
    z.name = "z"

    plot_dot_graph(z, verbose=False, to_file="step26_goldstein.png")