if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import os
from dezero.core import Variable
from dezero.datasets import Dataset, MNIST
from dezero.dataloaders import DataLoader
import dezero.optimizers as O
import dezero.functions as F
from dezero.models import MLP


if __name__ == "__main__":
    batch_size: int = 100
    train_set: Dataset = MNIST(train=True)
    train_loader = DataLoader(train_set, batch_size)
    model = MLP((1000, 10))
    opt = O.SGD().setup(model)

    if os.path.exists("my_mlp.npz"):
        model.load_weights("my_mlp.npz")

    max_epoch: int = 3
    for epoch in range(max_epoch):
        sum_loss: float = 0
        for x, t in train_loader:
            y: Variable = model(x)
            loss: Variable = F.softmax_cross_entropy(y, t)
            model.cleargrads()
            loss.backward()
            opt.update()
            sum_loss += float(loss.data) * len(t)
        print(f"epoch: {epoch+1}, loss: {sum_loss/len(train_set):.4f}")

    model.save_weights("step53_my_mlp.npz")