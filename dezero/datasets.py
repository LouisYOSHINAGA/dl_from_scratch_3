import os
import gzip
import tarfile
import pickle
import numpy as np
from dezero.transforms import Compose, Flatten, ToFloat, Normalize
from dezero.utils import get_file, cache_dir
from typing import Callable, Any
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, train: bool =True, transform: Callable[[Any], Any]|None =None,
                 target_transform: Callable[[Any], Any]|None =None):
        self.train: bool = train
        self.transform: Callable[[Any], Any] = transform if transform is not None \
                                               else lambda x: x
        self.target_transform: Callable[[Any], Any] = target_transform if target_transform is not None \
                                                      else lambda x: x
        self.data: Any = None
        self.label: Any = None
        self.prepare()

    def prepare(self) -> None:
        pass

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        assert np.isscalar(index)
        if self.label is None:
            return self.transform(self.data[index]), None
        else:
            return self.transform(self.data[index]), self.target_transform(self.label[index])

    def __len__(self) -> int:
        return len(self.data)


def get_spiral(train: bool= True) -> tuple[np.ndarray, np.ndarray]:
    rng: np.random.Generator = np.random.default_rng(1984 if train else 2020)

    n_class: int = 3
    n_data: int = 100
    input_dim: int = 2
    data_size: int = n_class * n_data

    xs: np.ndarray = np.zeros((data_size, input_dim), dtype=np.float32)
    ts: np.ndarray = np.zeros(data_size, dtype=int)
    for i in range(n_class):
        radius: np.ndarray = np.arange(n_data) / n_data
        thetas: np.ndarray = 4 * i + 4 * radius + 0.2 * rng.random(size=n_data)
        xs[i*n_data:(i+1)*n_data] = np.array([radius * np.cos(thetas), radius * np.sin(thetas)]).T
        ts[i*n_data:(i+1)*n_data] = i

    indexes: np.ndarray = rng.permutation(n_data * n_class)
    return xs[indexes], ts[indexes]


class Spiral(Dataset):
    def prepare(self) -> None:
        self.data, self.label = get_spiral(self.train)


class MNIST(Dataset):
    def __init__(self, train: bool =True,
                 transform: Callable[[Any], Any] =Compose([Flatten(), ToFloat(), Normalize(0, 255)]),
                 target_transform: Callable[[Any], Any]|None =None) -> None:
        super().__init__(train, transform, target_transform)

    def prepare(self) -> None:
        #url = 'http://yann.lecun.com/exdb/mnist/'
        url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
        train_files: dict[str, str] = {'target': 'train-images-idx3-ubyte.gz',
                                       'label': 'train-labels-idx1-ubyte.gz'}
        test_files: dict[str, str] = {'target': 't10k-images-idx3-ubyte.gz',
                                      'label': 't10k-labels-idx1-ubyte.gz'}
        files: str = train_files if self.train else test_files
        data_path: str = get_file(url + files['target'])
        label_path: str = get_file(url + files['label'])

        self.data: np.ndarray = self._load_data(data_path)
        self.label: np.ndarray = self._load_label(label_path)

    def _load_label(self, filepath: str) -> np.ndarray:
        with gzip.open(filepath, 'rb') as f:
            labels: np.ndarray = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels

    def _load_data(self, filepath: str) -> np.ndarray:
        with gzip.open(filepath, 'rb') as f:
            data: np.ndarray = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data

    def show(self, row: int =10, col: int =10) -> None:
        H: int = 28
        W: int = 28
        img: np.ndarray = np.zeros((H*row, W*col))
        for r in range(row):
            for c in range(col):
                img[r*H:(r+1)*H, c*W:(c+1)*W] = self.data[np.random.default_rng().integers(0, len(self.data)-1)].reshape(H, W)
        plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.show()

    @staticmethod
    def labels() -> dict[int, str]:
        return {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}


class CIFAR10(Dataset):

    def __init__(self, train=True,
                 transform=Compose([ToFloat(), Normalize(mean=0.5, std=0.5)]),
                 target_transform=None):
        super().__init__(train, transform, target_transform)

    def prepare(self):
        url='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        self.data, self.label = load_cache_npz(url, self.train)
        if self.data is not None:
            return
        filepath = get_file(url)
        if self.train:
            self.data = np.empty((50000, 3 * 32 * 32))
            self.label = np.empty((50000), dtype=int)
            for i in range(5):
                self.data[i * 10000:(i + 1) * 10000] = self._load_data(
                    filepath, i + 1, 'train')
                self.label[i * 10000:(i + 1) * 10000] = self._load_label(
                    filepath, i + 1, 'train')
        else:
            self.data = self._load_data(filepath, 5, 'test')
            self.label = self._load_label(filepath, 5, 'test')
        self.data = self.data.reshape(-1, 3, 32, 32)
        save_cache_npz(self.data, self.label, url, self.train)


    def _load_data(self, filename, idx, data_type='train'):
        assert data_type in ['train', 'test']
        with tarfile.open(filename, 'r:gz') as file:
            for item in file.getmembers():
                if ('data_batch_{}'.format(idx) in item.name and data_type == 'train') or ('test_batch' in item.name and data_type == 'test'):
                    data_dict = pickle.load(file.extractfile(item), encoding='bytes')
                    data = data_dict[b'data']
                    return data

    def _load_label(self, filename, idx, data_type='train'):
        assert data_type in ['train', 'test']
        with tarfile.open(filename, 'r:gz') as file:
            for item in file.getmembers():
                if ('data_batch_{}'.format(idx) in item.name and data_type == 'train') or ('test_batch' in item.name and data_type == 'test'):
                    data_dict = pickle.load(file.extractfile(item), encoding='bytes')
                    return np.array(data_dict[b'labels'])

    def show(self, row=10, col=10):
        H, W = 32, 32
        img = np.zeros((H*row, W*col, 3))
        for r in range(row):
            for c in range(col):
                img[r*H:(r+1)*H, c*W:(c+1)*W] = self.data[np.random.randint(0, len(self.data)-1)].reshape(3,H,W).transpose(1,2,0)/255
        plt.imshow(img, interpolation='nearest')
        plt.axis('off')
        plt.show()

    @staticmethod
    def labels():
        return {0: 'ariplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}


class CIFAR100(CIFAR10):

    def __init__(self, train=True,
                 transform=Compose([ToFloat(), Normalize(mean=0.5, std=0.5)]),
                 target_transform=None,
                 label_type='fine'):
        assert label_type in ['fine', 'coarse']
        self.label_type = label_type
        super().__init__(train, transform, target_transform)

    def prepare(self):
        url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
        self.data, self.label = load_cache_npz(url, self.train)
        if self.data is not None:
            return

        filepath = get_file(url)
        if self.train:
            self.data = self._load_data(filepath, 'train')
            self.label = self._load_label(filepath, 'train')
        else:
            self.data = self._load_data(filepath, 'test')
            self.label = self._load_label(filepath, 'test')
        self.data = self.data.reshape(-1, 3, 32, 32)
        save_cache_npz(self.data, self.label, url, self.train)

    def _load_data(self, filename, data_type='train'):
        with tarfile.open(filename, 'r:gz') as file:
            for item in file.getmembers():
                if data_type in item.name:
                    data_dict = pickle.load(file.extractfile(item), encoding='bytes')
                    data = data_dict[b'data']
                    return data

    def _load_label(self, filename, data_type='train'):
        assert data_type in ['train', 'test']
        with tarfile.open(filename, 'r:gz') as file:
            for item in file.getmembers():
                if data_type in item.name:
                    data_dict = pickle.load(file.extractfile(item), encoding='bytes')
                    if self.label_type == 'fine':
                        return np.array(data_dict[b'fine_labels'])
                    elif self.label_type == 'coarse':
                        return np.array(data_dict[b'coarse_labels'])

    @staticmethod
    def labels(label_type='fine'):
        coarse_labels = dict(enumerate(['aquatic mammals','fish','flowers','food containers','fruit and vegetables','household electrical device','household furniture','insects','large carnivores','large man-made outdoor things','large natural outdoor scenes','large omnivores and herbivores','medium-sized mammals','non-insect invertebrates','people','reptiles','small mammals','trees','vehicles 1','vehicles 2']))
        fine_labels = []
        return fine_labels if label_type == 'fine' else coarse_labels





# =============================================================================
# Big datasets
# =============================================================================
class ImageNet(Dataset):

    def __init__(self):
        NotImplemented

    @staticmethod
    def labels():
        url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt'
        path = get_file(url)
        with open(path, 'r') as f:
            labels = eval(f.read())
        return labels


# =============================================================================
# Sequential datasets: SinCurve, Shapekspare
# =============================================================================
class SinCurve(Dataset):

    def prepare(self):
        num_data = 1000
        dtype = np.float64

        x = np.linspace(0, 2 * np.pi, num_data)
        noise_range = (-0.05, 0.05)
        noise = np.random.uniform(noise_range[0], noise_range[1], size=x.shape)
        if self.train:
            y = np.sin(x) + noise
        else:
            y = np.cos(x)
        y = y.astype(dtype)
        self.data = y[:-1][:, np.newaxis]
        self.label = y[1:][:, np.newaxis]


class Shakespear(Dataset):

    def prepare(self):
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        file_name = 'shakespear.txt'
        path = get_file(url, file_name)
        with open(path, 'r') as f:
            data = f.read()
        chars = list(data)

        char_to_id = {}
        id_to_char = {}
        for word in data:
            if word not in char_to_id:
                new_id = len(char_to_id)
                char_to_id[word] = new_id
                id_to_char[new_id] = word

        indices = np.array([char_to_id[c] for c in chars])
        self.data = indices[:-1]
        self.label = indices[1:]
        self.char_to_id = char_to_id
        self.id_to_char = id_to_char


# =============================================================================
# Utils
# =============================================================================
def load_cache_npz(filename, train=False):
    filename = filename[filename.rfind('/') + 1:]
    prefix = '.train.npz' if train else '.test.npz'
    filepath = os.path.join(cache_dir, filename + prefix)
    if not os.path.exists(filepath):
        return None, None

    loaded = np.load(filepath)
    return loaded['data'], loaded['label']

def save_cache_npz(data, label, filename, train=False):
    filename = filename[filename.rfind('/') + 1:]
    prefix = '.train.npz' if train else '.test.npz'
    filepath = os.path.join(cache_dir, filename + prefix)

    if os.path.exists(filepath):
        return

    print("Saving: " + filename + prefix)
    try:
        np.savez_compressed(filepath, data=data, label=label)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise
    print(" Done")
    return filepath