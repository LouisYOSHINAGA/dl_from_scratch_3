import numpy as np
try:
    import Image  # type: ignore
except ImportError:
    from PIL import Image
from dezero.utils import pair
from typing import Callable, Any


class Compose:
    def __init__(self, transforms: list[Callable[[Any], Any]] =[]):
        self.transforms: list[Callable[[Any], Any]] = transforms

    def __call__(self, img: Any) -> Any:
        if not self.transforms:
            return img
        for t in self.transforms:
            img = t(img)
        return img


class Convert:
    def __init__(self, mode: str ='RGB') -> None:
        self.mode: str = mode

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.mode == 'BGR':
            img: Image.Image = img.convert('RGB')
            r, g, b = img.split()
            img = Image.merge('RGB', (b, g, r))
            return img
        else:
            return img.convert(self.mode)


class Resize:
    def __init__(self, size: int, mode: int =Image.BILINEAR) -> None:
        self.size: tuple[int, int] = pair(size)
        self.mode: int = mode

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.resize(self.size, self.mode)


class CenterCrop:
    def __init__(self, size: int) -> None:
        self.size = pair(size)

    def __call__(self, img: Image.Image) -> Image.Image:
        W, H = img.size
        OW, OH = self.size
        left: int = (W - OW) // 2
        right: int = W - ((W - OW) // 2 + (W - OW) % 2)
        up: int = (H - OH) // 2
        bottom: int = H - ((H - OH) // 2 + (H - OH) % 2)
        return img.crop((left, up, right, bottom))


class ToArray:
    def __init__(self, dtype: np.dtype =np.float32) -> None:
        self.dtype: np.dtype = dtype

    def __call__(self, img: np.ndarray|Image.Image) -> np.ndarray:
        if isinstance(img, np.ndarray):
            return img
        if isinstance(img, Image.Image):
            img = np.asarray(img)
            img = img.transpose(2, 0, 1)
            img = img.astype(self.dtype)
            return img
        else:
            raise TypeError


class ToPIL:
    def __call__(self, array: np.ndarray) -> Image.Image:
        data = array.transpose(1, 2, 0)
        return Image.fromarray(data)


class RandomHorizontalFlip:
    pass


class Normalize:
    def __init__(self, mean: float|np.ndarray =0, std: float|np.ndarray =1) -> None:
        self.mean: float|np.ndarray = mean
        self.std: float|np.ndarray = std

    def __call__(self, array: np.ndarray) -> np.ndarray:
        mean: float|np.ndarray = self.mean
        std: float|np.ndarray = self.std
        if not np.isscalar(mean):
            mshape = [1] * array.ndim
            mshape[0] = len(array) if len(mean) == 1 else len(mean)
            mean = np.array(mean, dtype=array.dtype).reshape(*mshape)
        if not np.isscalar(std):
            rshape = [1] * array.ndim
            rshape[0] = len(array) if len(std) == 1 else len(std)
            std = np.array(std, dtype=array.dtype).reshape(*rshape)
        return (array - mean) / std


class Flatten:
    def __call__(self, array: np.ndarray) -> np.ndarray:
        return array.flatten()


class AsType:
    def __init__(self, dtype: np.dtype =np.float32) -> None:
        self.dtype = dtype

    def __call__(self, array: np.ndarray) -> np.ndarray:
        return array.astype(self.dtype)


ToFloat = AsType


class ToInt(AsType):
    def __init__(self, dtype: np.dtype =int) -> None:
        self.dtype: np.dtype = dtype