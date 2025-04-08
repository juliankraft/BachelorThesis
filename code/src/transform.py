import numpy as np
from PIL import Image
from pathlib import Path
from typing import Any, Sequence

from os import PathLike


class ImagePipeline:

    """

    A image processing pipeline that allows for a series of transformations to be applied.
    This list is defined by a list of (method_name, kwargs) passed as steps.

    Example:

    pipeline = ImagePipeline([
        ('load', {'path': 'path/to/image.jpg'}),
        ('to_rgb', {}),
        ('crop_by_bb', {}),
        ('resize', {'size': 128}),
    ])

    image = pipeline('path/to/image.jpg', bbox=[0.1, 0.1, 0.8, 0.8])

    """

    def __init__(
            self,
            steps: list[tuple[str, dict]] | None = None
            ):

        if steps is None:
            self.steps = []
        else:
            self.steps = steps

        self.img: Image.Image = self.create_dummy_image()

    def load(
            self,
            path: str | PathLike
            ):

        try:
            self.img = Image.open(Path(path))
        except Exception as e:
            raise RuntimeError(f"Failed to load image from {path}: {e}")
        return self

    def to_rgb(self):
        self.img = self.img.convert("RGB")
        return self

    def crop_by_bb(
            self,
            bbox: list[float] | tuple[float, float, float, float]
            ):

        width, height = self.img.size

        x1 = int(bbox[0] * width)
        y1 = int(bbox[1] * height)
        x2 = int((bbox[0] + bbox[2]) * width)
        y2 = int((bbox[1] + bbox[3]) * height)

        self.img = self.img.crop((x1, y1, x2, y2))
        return self

    def crop_center_sample(
            self,
            bbox: list[float] | tuple[float, float, float, float],
            size: int | Sequence[int] = 50
            ):

        size = self.process_size(size)

        width, height = self.img.size

        center_x = int(bbox[0] * width) + int(bbox[2] * width / 2)
        center_y = int(bbox[1] * height) + int(bbox[3] * height / 2)

        x1 = center_x - (size[0] // 2)
        y1 = center_y - (size[1] // 2)
        x2 = x1 + size[0]
        y2 = y1 + size[1]

        self.img = self.img.crop((x1, y1, x2, y2))
        return self

    def resize(
            self,
            size: Sequence[int] | int,
            ):

        size = self.process_size(size)

        self.img = self.img.resize(size)
        return self

    def create_dummy_image(
            self,
            size=(8, 8),
            channels=3) -> Image.Image:

        size = self.process_size(size)

        array = np.random.randint(0, 256, size + (channels,), dtype=np.uint8)
        return Image.fromarray(array)

    def get(self) -> Image.Image:
        return self.img

    def process_size(
            self,
            size: Sequence[int] | int,
            ) -> tuple[int, int]:

        if isinstance(size, int):
            size = (size, size)
        elif isinstance(size, Sequence):
            if len(size) == 1:
                size = (size[0], size[0])
            elif len(size) == 2:
                size = (size[0], size[1])
            else:
                raise ValueError("size must be an int or a sequence of maximum two integers.")

        return size

    def __call__(
            self,
            path: str | PathLike,
            bbox: Sequence[float],
            ) -> Any:

        self.load(path)

        for step, kwargs in self.steps:
            method = getattr(self, step)
            if method is None:
                raise AttributeError(f"No method named '{step}' found in ImagePipeline")
            elif 'bbox' in method.__code__.co_varnames:
                self = method(bbox, **kwargs)
            else:
                self = method(**kwargs)

        return self.get()
