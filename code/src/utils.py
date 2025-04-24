import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
from typing import Sequence
from matplotlib.figure import Figure
from matplotlib.axes import Axes

BBox = Sequence[float]


def load_config_yaml(path_to_config):
    with open(path_to_config, 'r') as f:
        path_config = yaml.safe_load(f)
    return {k: Path(v) for k, v in path_config.items()}


def best_weighted_split(
        weights: np.ndarray,
        target_left_sum: float
        ) -> int:

    total = weights.sum()
    target_right = total - target_left_sum

    best_cut = 0
    best_error = abs(0 - target_left_sum) + abs(total - target_right)

    for idx in range(1, len(weights) + 1):
        left = weights[:idx].sum()
        right = total - left
        err = abs(left - target_left_sum) + abs(right - target_right)
        if err < best_error:
            best_error = err
            best_cut = idx

    return best_cut


def plot_image_with_bbox(
        image: Image.Image,
        bbox: BBox,
        conf: float | None = None) -> Figure:

    """
    Plot an image with a bounding box.
    Args:
        image (Image.Image): The image to plot.
        bbox (BBox): The bounding box to plot, in the format [x, y, w, h].
        conf (float | None): The confidence score to display. If None, no score is displayed.
    Returns:
        Figure: The matplotlib figure object.
    """

    width, height = image.size

    x_abs = bbox[0] * width
    y_abs = bbox[1] * height
    w_abs = bbox[2] * width
    h_abs = bbox[3] * height

    fig, ax = plt.subplots()
    ax.imshow(image)

    rect = patches.Rectangle(
        (x_abs, y_abs), w_abs, h_abs,
        linewidth=1, edgecolor='red', facecolor='none'
    )

    ax.add_patch(rect)

    if conf is not None:
        ax.text(
            x_abs + 5, y_abs - 10,
            f"conf = {conf:.2f}",
            fontsize=8,
            color='white',
            bbox=dict(facecolor='red', alpha=0.5, edgecolor='none', pad=1.5)
        )

    ax.axis('off')
    plt.tight_layout()

    plt.close(fig)

    return fig


def draw_bbox_on_ax(
        ax: Axes,
        image: Image.Image,
        bbox: BBox,
        conf: float | None = None
        ) -> None:
    """
    Draw an image with a bounding box onto an existing Axes.

    Args:
        ax (Axes): The axes to draw into.
        image (Image.Image): The image to plot.
        bbox (BBox): Bounding box [x, y, w, h] in relative coords (0–1).
        conf (float, optional): Confidence score to annotate.
    """
    width, height = image.size
    x_abs, y_abs = bbox[0] * width, bbox[1] * height
    w_abs, h_abs = bbox[2] * width, bbox[3] * height

    ax.imshow(image)
    rect = patches.Rectangle(
        (x_abs, y_abs), w_abs, h_abs,
        linewidth=1, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect)

    if conf is not None:
        ax.annotate(
            f"conf = {conf:.2f}",
            xy=(x_abs, y_abs),
            xytext=(2, 8),
            textcoords="offset points",
            ha="left", va="top",
            fontsize=8,
            color="white",
            bbox=dict(facecolor="red", alpha=0.5, edgecolor="none", pad=1.5),
            clip_on=False
        )
    ax.axis('off')
