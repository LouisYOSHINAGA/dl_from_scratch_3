if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import dezero
from dezero.dataloaders import SeqDataLoader


if __name__ == "__main__":
    train_set = dezero.datasets.SinCurve(train=True)
    dataloader = SeqDataLoader(train_set, batch_size=3)
    x, t = next(dataloader)
    print(x)
    print("--------------------")
    print(t)