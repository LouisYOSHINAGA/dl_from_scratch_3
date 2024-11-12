if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from dezero import Variable


if __name__ == "__main__":
    """
        dy/dx = 2x
        z = (dy/dx)^3 + y = 8x^3 + x^2
        dz/dx = 24x^2 + 2x
    """
    x = Variable(np.array(2.0))
    y: Variable = x ** 2
    y.backward(create_graph=True)

    gx: Variable = x.grad
    x.cleargrad()

    z: Variable = gx ** 3 + y
    z.backward()
    print(f"{x.grad=}")