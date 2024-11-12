if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F


if __name__ == "__main__":
    x = Variable(np.array(1.0), name="x")
    y: Variable = F.tanh(x)
    y.name = "y"
    y.backward(create_graph=True)

    iters: int = 0
    for i in range(iters):
        gx: Variable = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)

    gx = x.grad
    gx.name = f"gx{iters+1:d}"
    plot_dot_graph(gx, verbose=False, to_file=f"step35_tanh_{iters+1}.png")