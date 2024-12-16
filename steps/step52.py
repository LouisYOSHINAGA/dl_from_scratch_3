if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import time
import dezero
from dezero.core import Variable
from dezero.datasets import Dataset, MNIST
from dezero import DataLoader
from dezero.models import MLP
import dezero.functions as F
import dezero.optimizers as O


if __name__ == "__main__":
    batch_size: int = 100
    train_set: Dataset = MNIST(train=True)
    train_loader = DataLoader(train_set, batch_size)
    model = MLP((1000, 10))
    opt = O.SGD().setup(model)

    if dezero.cuda.gpu_enable:
        train_loader.to_gpu()
        model.to_gpu()

    max_epoch: int = 5
    for epoch in range(max_epoch):
        start: float = time.time()
        sum_loss: float = 0

        for x, t in train_loader:
            y = model(x)
            loss: Variable = F.softmax_cross_entropy(y, t)
            model.cleargrads()
            loss.backward()
            opt.update()
            sum_loss += float(loss.data) * len(t)

        elapsed_time: float = time.time() - start
        print(f"epoch: {epoch+1}, loss: {sum_loss/len(train_set):.4f}, time: {elapsed_time:.4f}")