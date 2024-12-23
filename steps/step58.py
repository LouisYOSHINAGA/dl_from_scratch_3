if "__file__" in globals():
	import os, sys
	sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import dezero
from dezero.models import VGG16
from PIL import Image


if __name__ == "__main__":
	"""
	model = VGG16(pretrained=True)
	x: np.ndarray = np.random.default_rng().normal(size=(1, 3, 224, 224)).astype(np.float32)
	model.plot(x, to_file="step58_model.png")  # fig. 58-2
	"""

	"""
	img_path: str = dezero.utils.get_file(
		url="https://github.com/oreilly-japan/deep-learning-from-scratch-3/raw/images/zebra.jpg"
	)
	img = Image.open(img_path)
	img.show()  # fig. 58-3
	"""


	img_path: str = dezero.utils.get_file(
		url="https://github.com/oreilly-japan/deep-learning-from-scratch-3/raw/images/zebra.jpg"
	)
	img: Image.Image = Image.open(img_path)
	x = VGG16.preprocess(img)
	x = x[np.newaxis]

	model = VGG16(pretrained=True)
	with dezero.test_mode():
		y = model(x)
	predict_id: np.ndarray = np.argmax(y.data)

	model.plot(x, to_file="step58_vgg.pdf")  # fig. 58-2
	labels: dict[int, str] = dezero.datasets.ImageNet.labels()
	print(labels[predict_id])