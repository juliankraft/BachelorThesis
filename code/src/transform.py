from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from torch import Tensor
from pathlib import Path
from typing import Sequence, Callable, Any, cast
from os import PathLike

BBox = Sequence[float]


class ImagePipeline:

    """

    A image processing pipeline that allows for a series of transformations to be applied.
    This list is defined by a list of (method_name, kwargs) passed as steps.

    Parameters
    ----------
    path_to_dataset : str | PathLike | None
        Path to the dataset. If None, a dummy image will be created.
    steps : list[tuple[str, dict]] | None
        List of steps to be applied to the image. Each step is a tuple of (method_name, kwargs).
        If None, no steps will be applied. Image will be loaded and returned as is.
    transform : Callable | None
        A instance of the torchvision.transforms.Compose class or any other callable that takes a PIL image
        and returns a transformed image. If None, no transformation will be applied.

    Example:

    pipeline = ImagePipeline(
        path_to_dataset='path/to/dataset',
        steps=[
            ('load', {'path': 'path/to/image.jpg'}),
            ('to_rgb', {}),
            ('crop_by_bb', {}),
            ('resize', {'size': 128}),
            ]
        transform=v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize((224, 224)),
        ])
        )

    image = pipeline('path/to/image.jpg', bbox=[0.1, 0.1, 0.8, 0.8])

    """

    def __init__(
            self,
            path_to_dataset: str | PathLike,
            steps: list[tuple[str, dict]] | None = None,
            transform: Callable | None = None
            ):

        self.img: Image.Image | Tensor | None = None

        self.path_to_dataset = Path(path_to_dataset)

        if steps is None:
            steps = []
        self.steps = steps

        self.transform = transform

    def load(
            self,
            path: str | PathLike
            ):

        try:
            self.img = Image.open(self.path_to_dataset / path)
        except Exception as e:
            raise RuntimeError(f"Failed to load image from {self.path_to_dataset / path}: {e}")
        return self

    def to_rgb(self):
        self.img = self._pil().convert("RGB")
        return self

    def crop_by_bb(
            self,
            bbox: BBox,
            ):

        width, height = self._pil().size

        x1 = int(bbox[0] * width)
        y1 = int(bbox[1] * height)
        x2 = int((bbox[0] + bbox[2]) * width)
        y2 = int((bbox[1] + bbox[3]) * height)

        self.img = self._pil().crop((x1, y1, x2, y2))
        return self

    def crop_center_sample(
            self,
            bbox: BBox,
            size: int | Sequence[int] = 50
            ):

        size = self._process_size(size)
        width, height = self._pil().size
        center_x = int(bbox[0] * width) + int(bbox[2] * width / 2)
        center_y = int(bbox[1] * height) + int(bbox[3] * height / 2)

        x1 = center_x - (size[0] // 2)
        y1 = center_y - (size[1] // 2)
        x2 = x1 + size[0]
        y2 = y1 + size[1]

        self.img = self._pil().crop((x1, y1, x2, y2))
        return self

    def resize(
            self,
            size: Sequence[int] | int,
            ):

        size = self._process_size(size)

        self.img = self._pil().resize(size)
        return self

    def get(self) -> Image.Image | Tensor:
        if self.img is None:
            raise ValueError("Image has not been processed yet.")
        return self.img

    def _pil(self) -> Image.Image:
        if self.img is None:
            raise ValueError("Image is not loaded yet. Call `.load()` first.")
        if not isinstance(self.img, Image.Image):
            raise TypeError("Operation requires PIL image. Call `.to_tensor()` after this step.")
        return cast(Image.Image, self.img)

    def _process_size(
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
            bbox: BBox,
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

        result = self.get()

        if self.transform:
            return self.transform(result)
        return result


class BatchImagePipeline(ImagePipeline):

    def __init__(
            self,
            path_to_dataset: str | PathLike,
            num_workers: int = 4,
            steps: list[tuple[str, dict]] | None = None,
            transform: Callable | None = None    
            ):

        super().__init__(
            path_to_dataset=path_to_dataset,
            steps=steps,
            transform=transform
            )

        self.num_workers = num_workers

    def __call__(
            self,
            paths: list[str | PathLike],
            bboxes: list[Sequence[float]],
            ) -> list[Any]:

        if len(paths) != len(bboxes):
            raise ValueError("paths and bboxes must have the same length.")

        def process_one(args):
            path, bbox = args
            pipeline = ImagePipeline(
                                path_to_dataset=self.path_to_dataset,
                                steps=self.steps.copy(),
                                transform=self.transform
                                )
            return pipeline(path, bbox)

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(process_one, zip(paths, bboxes)))

        return results
