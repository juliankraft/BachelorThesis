from PIL import Image
from pathlib import Path
from typing import Sequence

from os import PathLike


class ImagePipeline:
    def __init__(
            self,
            path: str | PathLike,
            ):

        self.img: Image.Image = Image.open(path)

    def load(
            self,
            path: str | Path
            ):
        self.img = Image.open(path)
        return self

    def to_rgb(self):
        self.img = self.img.convert("RGB")
        return self

    def crop_by_bbox(
            self,
            bbox: list[float] | tuple[float, float, float, float]
            ):

        width, height = self.img.size

        x1 = int(bbox[0] * width)
        y1 = int(bbox[1] * height)
        x2 = int((bbox[0] + bbox[2]) * width)
        y2 = int((bbox[1] + bbox[3]) * height)

        self.img = self.img.crop((x1, y1, x2, y2))
        self.img = self.img.crop((x1, y1, x2, y2))

    def crop_center_sample(
            self,
            bbox: list[float] | tuple[float, float, float, float],
            sample_size: int | Sequence[int] = (50, 50)
            ):

        if isinstance(sample_size, int):
            target_width = target_height = sample_size
        elif isinstance(sample_size, Sequence):
            if len(sample_size) == 1:
                target_width = target_height = sample_size[0]
            elif len(sample_size) == 2:
                target_width, target_height = sample_size
            else:
                raise ValueError("sample_size must be an int or a sequence of maximum two integers.")

        width, height = self.img.size

        center_x = int(bbox[0] * width) + int(bbox[2] * width / 2)
        center_y = int(bbox[1] * height) + int(bbox[3] * height / 2)

        x1 = center_x - (target_width // 2)
        y1 = center_y - (target_height // 2)
        x2 = x1 + target_width
        y2 = y1 + target_height

        self.img = self.img.crop((x1, y1, x2, y2))
        return self

    def resize(
            self,
            size: Sequence[int] | int,
            ):
        if isinstance(size, int):
            size = (size, size)
        elif isinstance(size, Sequence):
            if len(size) == 1:
                size = (size[0], size[0])
            elif len(size) == 2:
                size = (size[0], size[1])
            else:
                raise ValueError("size must be an int or a sequence of maximum two integers.")
        self.img = self.img.resize(size)
        return self

    def get(self) -> Image.Image:
        return self.img
