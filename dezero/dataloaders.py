import math
import numpy as np
from dezero.datasets import Dataset


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool =True) -> None:
        self.dataset: Dataset = dataset
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self.data_size: int = len(dataset)
        self.max_iter: int = math.ceil(self.data_size / batch_size)
        self.reset()

    def reset(self) -> None:
        self.iteration: int = 0
        self.index: int = np.random.default_rng().permutation(len(self.dataset)) if self.shuffle \
                          else np.arange(len(self.dataset))

    def __iter__(self) -> int:
        return self

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration()
        batch_index: np.ndarray = self.index[self.iteration*self.batch_size:(self.iteration+1)*self.batch_size]
        batch: list[np.ndarray] = [self.dataset[i] for i in batch_index]
        xs: np.ndarray = np.array([example[0] for example in batch])
        ts: np.ndarray = np.array([example[1] for example in batch])
        self.iteration += 1
        return xs, ts

    def next(self) -> tuple[np.ndarray, np.ndarray]:
        return self.__next__()


class SeqDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, batch_size: int) -> None:
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=False)

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration
        jump: int = self.data_size // self.batch_size
        batch_index: list[int] = [(i * jump + self.iteration) % self.data_size for i in range(self.batch_size)]
        batch: list[np.ndarray] = [self.dataset[i] for i in batch_index]
        xs: np.ndarray = np.array([example[0] for example in batch])
        ts: np.ndarray = np.array([example[1] for example in batch])
        self.iteration += 1
        return xs, ts