if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from dezero import Variable
import dezero.functions as F
import matplotlib.pyplot as plt


if __name__ == "__main__":
    x = Variable(np.linspace(-7, 7, 200))
    y: Variable = F.sin(x)
    y.backward(create_graph=True)

    logs: list[np.ndarray] = [y.data.flatten()]

    for i in range(3):
        logs.append(x.grad.data.flatten())
        gx: Variable = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)

    # fig. 34-1
    labels = ["y=sin(x)", "y'", "y''", "y'''"]
    for i, v in enumerate(logs):
        plt.plot(x.data, logs[i], label=labels[i])
    plt.legend(loc="lower right")
    plt.show()