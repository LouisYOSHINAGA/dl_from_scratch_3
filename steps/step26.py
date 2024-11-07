if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph


def goldstein(x, y):
    return (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) \
         * (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))


if __name__ == "__main__":
    x = Variable(np.array(1.0), name="x")
    y = Variable(np.array(1.0), name="y")
    z: Variable = goldstein(x, y)
    z.name = "z"
    z.backward()
    plot_dot_graph(z, verbose=False, to_stdout=True, to_file="step26.png")