if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import dezero
from dezero.core import Variable
import dezero.functions as F
from dezero.datasets import Dataset, Spiral
from dezero.dataloaders import DataLoader
from dezero.models import MLP
import dezero.optimizers as O
import matplotlib.pyplot as plt


def plot(train: list[float], test: list[float], label: str) -> None:
    plt.figure(figsize=(7, 5))
    plt.xlabel("epoch")
    plt.ylabel(label)
    plt.plot(range(len(train)), train, label="train")
    plt.plot(range(len(test)), test, label="test")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    batch_size: int = 30
    train_set: Dataset = Spiral(train=True)
    test_set: Dataset = Spiral(train=False)
    train_loader = DataLoader(train_set, batch_size)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    hidden_size: int = 10
    lr: float = 1.0
    model = MLP((hidden_size, 3))
    opt: O.Optimizer = O.SGD(lr).setup(model)

    max_epoch: int = 300
    train_losses: list[float] = []
    train_accs: list[float] = []
    test_losses: list[float] = []
    test_accs: list[float] = []
    for epoch in range(max_epoch):

        # train
        sum_loss: float = 0
        sum_acc: float = 0
        for x, t in train_loader:
            y: Variable = model(x)
            loss: Variable = F.softmax_cross_entropy(y, t)
            acc: Variable = F.accuracy(y, t)

            model.cleargrads()
            loss.backward()
            opt.update()

            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)
        train_losses.append(sum_loss/len(train_set))
        train_accs.append(sum_acc/len(train_set))

        print(f"epoch {epoch+1:03d}")
        print(f"train loss: {sum_loss/len(train_set):.4f}, accuracy: {sum_acc/len(train_set):.4f}")

        # test
        sum_loss = 0
        sum_acc = 0
        with dezero.no_grad():
            for x, t in test_loader:
                y: Variable = model(x)
                loss: Variable = F.softmax_cross_entropy(y, t)
                acc: Variable = F.accuracy(y, t)

                sum_loss += float(loss.data) * len(t)
                sum_acc += float(acc.data) * len(t)
            test_losses.append(sum_loss/len(test_set))
            test_accs.append(sum_acc/len(test_set))

        print(f"test loss: {sum_loss/len(test_set):.4f}, accuracy: {sum_acc/len(test_set):.4f}")

    plot(train_losses, test_losses, "loss")  # fig. 50-1
    plot(train_accs, test_accs, "accuracy")  # fig. 50-1