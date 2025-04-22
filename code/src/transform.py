from PIL import Image, ImageOps
from concurrent.futures import ThreadPoolExecutor

from torch import Tensor
from pathlib import Path
from typing import Sequence, Callable, Any, cast
from os import PathLike

from ba_dev.utils import BBox


class ImagePipeline:

    """

    A image processing pipeline that allows for a series of initial operations to be applied.
    This list is defined by a list of (method_name, kwargs) passed as pre_ops.
    Additionally, a transform can be applied to the image after the pre_ops.

    Parameters
    ----------
    path_to_dataset : str | PathLike | None
        Path to the dataset. If None, a dummy image will be created.
    pre_ops : list[tuple[str, dict]] | None
        List of pre_ops to be applied to the image. Each step is a tuple of (method_name, kwargs).
        If None, no pre_ops will be applied. Image will only be loaded and returned as is.
    transform : Callable | None
        A instance of the torchvision.transforms.Compose class or any other callable that takes a PIL image
        and returns a transformed image. If None, no transformation will be applied.

    Example:

    pipeline = ImagePipeline(
        path_to_dataset='path/to/dataset',
        pre_ops=[
            ('to_rgb', {}),
            ('crop_by_bb', {'crop_shape': 1.0})
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
            path_to_dataset: str | PathLike | None = None,
            pre_ops: list[tuple[str, dict]] | None = None,
            transform: Callable | None = None
            ):

        self.img: Image.Image | Tensor | None = None

        self.path_to_dataset = Path(path_to_dataset) if path_to_dataset is not None else None

        if pre_ops is None:
            pre_ops = []
        self.pre_ops = pre_ops

        self.transform = transform

    def load(
            self,
            path: str | PathLike
            ):
        if self.path_to_dataset is None:
            raise ValueError("Path to dataset has not been set in the pipeline.")
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
            crop_shape: None | float | int | Sequence[int] = None,
            ):

        """
        Flexible crop using bounding box.

        Parameters
        ----------
        bbox : Sequence[float]
            Normalized bounding box: [x, y, width, height]
        crop_shape : None | float | int | Sequence[int]
            - None: crop exact bounding box
            - float: desired aspect ratio (width / height), will pad if needed
            - int or (w, h): fixed size center crop around bbox center, will pad if needed
        """

        img = self._pil()

        width, height = img.size

        x = bbox[0] * width
        y = bbox[1] * height
        w = bbox[2] * width
        h = bbox[3] * height

        if crop_shape is None:
            x1 = int(x)
            y1 = int(y)
            x2 = int(x + w)
            y2 = int(y + h)

        else:
            if isinstance(crop_shape, float):
                if w / h > crop_shape:
                    new_w = w
                    new_h = w / crop_shape
                else:
                    new_h = h
                    new_w = h * crop_shape

            else:
                new_w, new_h = self._process_size(crop_shape)

            cx = x + w / 2
            cy = y + h / 2

            x1 = cx - new_w / 2
            y1 = cy - new_h / 2
            x2 = cx + new_w / 2
            y2 = cy + new_h / 2

            pad_left = max(0, -int(x1))
            pad_top = max(0, -int(y1))
            pad_right = max(0, int(x2) - width)
            pad_bottom = max(0, int(y2) - height)

            if any([pad_left, pad_top, pad_right, pad_bottom]):
                img = ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0)
                x1 += pad_left
                x2 += pad_left
                y1 += pad_top
                y2 += pad_top

        self.img = img.crop((x1, y1, x2, y2))
        return self

    def resize(
            self,
            size: Sequence[int] | int,
            ):

        size = self._process_size(size)

        self.img = self._pil().resize(size)
        return self

    def reduce_resolution(
            self,
            factor: float
            ):
        if factor <= 0 or factor > 1:
            raise ValueError("factor must be between 0 and 1.")

        img = self._pil()
        width, height = img.size
        new_width = int(width * factor)
        new_height = int(height * factor)

        self.img = img.resize((new_width, new_height))

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

        for step, kwargs in self.pre_ops:
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
            path_to_dataset: str | PathLike | None = None,
            num_workers: int = 1,
            pre_ops: list[tuple[str, dict]] | None = None,
            transform: Callable | None = None
            ):

        super().__init__(
            path_to_dataset=path_to_dataset,
            pre_ops=pre_ops,
            transform=transform
            )

        if num_workers < 1:
            raise ValueError("num_workers must be greater than 0.")
        self.num_workers = num_workers

    def set_path_to_dataset(self, path: str | PathLike):
        self.path_to_dataset = Path(path)

    def __call__(
            self,
            paths: list[str | PathLike],
            bboxes: list[Sequence[float]],
            ) -> list[Any]:

        if self.path_to_dataset is None:
            raise ValueError("Path to dataset must be set before calling BatchImagePipeline.")

        if len(paths) != len(bboxes):
            raise ValueError("paths and bboxes must have the same length.")

        def process_one(args):
            path, bbox = args
            pipeline = ImagePipeline(
                                path_to_dataset=self.path_to_dataset,
                                pre_ops=self.pre_ops.copy(),
                                transform=self.transform
                                )
            return pipeline(path, bbox)

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(process_one, zip(paths, bboxes)))

        return results
