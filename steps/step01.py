import numpy as np

class Variable:
    def __init__(self, data: np.ndarray) -> None:
        self.data: np.ndarray = data

data: np.ndarray = np.array(1.0)
x = Variable(data)
print(x.data)

x.data = np.array(2.0)
print(x.data)