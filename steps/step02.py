import numpy as np


class Variable:
    def __init__(self, data: np.ndarray) -> None:
        self.data: np.ndarray = data


class Function:
    def __call__(self, input: Variable) -> Variable:
        x: np.ndarray = input.data
        y: np.ndarray = self.forward(x)
        return Variable(y)

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class Square(Function):
    def forward(self, x: Variable) -> Variable:
        return x ** 2


if __name__ == "__main__":
    x = Variable(np.array(10))
    f = Square()
    y = f(x)

    print(f"{type(y)=}")
    print(f"{y.data=}")