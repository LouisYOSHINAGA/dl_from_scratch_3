if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from dezero import Variable, test_mode
import dezero.functions as F


if __name__ == "__main__":
    x: np.ndarray = np.ones(5)
    print(x)

    y: Variable = F.dropout(x)
    print(y)

    with test_mode():
        y = F.dropout(x)
        print(y)