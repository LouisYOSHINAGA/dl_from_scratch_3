import numpy as np
from dezero import Variable, Layer
import dezero.functions as F
import dezero.layers as L
from dezero.transforms import Image
from dezero import utils
from dezero.cuda import xpy
from typing import Callable


class Model(Layer):
    def plot(self, *inputs: Variable|np.ndarray, to_file: str ="model.png") -> None:
        y: list[Variable]|Variable = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)


class MLP(Model):
    def __init__(self, fc_output_sizes: list[int],
                 activation: Callable[[Variable], Variable] =F.sigmoid) -> None:
        super().__init__()
        self.activation: Callable[[Variable], Variable] = activation
        self.layers: list[Layer] = []
        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, f"l{i}", layer)
            self.layers.append(layer)

    def forward(self, x: Variable) -> Variable:
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)


class VGG16(Model):
    WEIGHTS_PATH: str = "https://github.com/koki0702/dezero-models/releases/download/v0.1/vgg16.npz"

    def __init__(self, pretrained: bool =False) -> None:
        super().__init__()
        self.conv1_1 = L.Conv2d( 64, kernel_size=3, stride=1, pad=1)
        self.conv1_2 = L.Conv2d( 64, kernel_size=3, stride=1, pad=1)
        self.conv2_1 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
        self.conv2_2 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
        self.conv3_1 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv3_2 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv3_3 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv4_1 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv4_2 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv4_3 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_1 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_2 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_3 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.fc6 = L.Linear(4096)
        self.fc7 = L.Linear(4096)
        self.fc8 = L.Linear(1000)

        if pretrained:
            self.load_weights(utils.get_file(VGG16.WEIGHTS_PATH))

    @staticmethod
    def preprocess(image: Image.Image, size: tuple[int, int] =(224, 224),
                   dtype: xpy.dtype =np.float32) -> None:
        image = image.convert("RGB")
        image = image.resize(size)
        image = np.asarray(image, dtype=dtype)[:, :, ::-1]
        image -= np.array([103.939, 116.779, 123.68], dtype=dtype)
        image = image.transpose((2, 0, 1))
        return image

    def forward(self, x: Variable) -> Variable:
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = F.pooling(x, 2, 2)
        x = F.reshape(x, (x.shape[0], -1))
        x = F.dropout(F.relu(self.fc6(x)))
        x = F.dropout(F.relu(self.fc7(x)))
        return self.fc8(x)