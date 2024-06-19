if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
import dezero
from dezero.models import MLP
from dezero import optimizers
import dezero.functions as F
from dezero import transforms


if __name__ == "__main__":
    train_set = dezero.datasets.Spiral(train=True)
    print(train_set[0])
    print(len(train_set))

    train_set = dezero.datasets.Spiral()
    batch_index = [0, 1, 2]
    batch = [train_set[i] for i in batch_index]
    x = np.array([example[0] for example in batch])
    t = np.array([example[1] for example in batch])
    print(x.shape)
    print(t.shape)


    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    lr = 1.0

    train_set = dezero.datasets.Spiral()
    model = MLP((hidden_size, 10))
    optimizer = optimizers.SGD(lr).setup(model)

    data_size = len(train_set)
    max_iter = math.ceil(data_size / batch_size)

    for epoch in range(max_epoch):
        index = np.random.permutation(data_size)
        sum_loss = 0

        for i in range(max_iter):
            batch_index = index[i*batch_size : (i+1)*batch_size]
            batch = [train_set[i] for i in batch_index]
            batch_x = np.array([example[0] for example in batch])
            batch_t = np.array([example[1] for example in batch])

            y = model(batch_x)
            loss = F.softmax_cross_entropy(y, batch_t)
            model.cleargrads()
            loss.backward()
            optimizer.update()

            sum_loss += float(loss.data) * len(batch_t)

        avg_loss = sum_loss / data_size
        print(f"epoch {epoch+1}, loss={avg_loss:.2f}")


    def f(x):
        y = x / 2.0
        return y

    train_set = dezero.datasets.Spiral(transform=f)


    f = transforms.Normalize(mean=0.0, std=2.0)
    train_set = dezero.datsets.Spiral(transform=F)


    f = transforms.Compose([transforms.Normalize(mean=0.0, std=2.0),
                            transforms.AsType(np.float64)])