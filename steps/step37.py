if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import dezero.functions as F
from dezero import Variable


if __name__ == "__main__":
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    c = Variable(np.array([[10, 20, 30], [40, 50, 60]]))

    y = F.sin(x)
    print(y)

    y = x + c
    print(y)

    t = x + c
    y = F.sum(t)
    y.backward(retain_grad=True)
    print(y.grad)
    print(t.grad)
    print(x.grad)
    print(c.grad)