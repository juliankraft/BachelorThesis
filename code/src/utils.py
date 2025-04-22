import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
from typing import Sequence
from matplotlib.figure import Figure

BBox = Sequence[float]


def load_config_yaml(path_to_config):
    with open(path_to_config, 'r') as f:
        path_config = yaml.safe_load(f)
    return {k: Path(v) for k, v in path_config.items()}


def best_weighted_split(
        weights: np.ndarray,
        target_left_sum: float
        ):

    total_sum = weights.sum()
    target_right_sum = total_sum - target_left_sum

    best_cut_idx = None
    best_error = float('inf')

    for idx in range(1, len(weights)):
        left = weights[:idx].sum()
        right = total_sum - left

        error = abs(left - target_left_sum) + abs(right - target_right_sum)

        if error < best_error:
            best_error = error
            best_cut_idx = idx

    if best_cut_idx is None:
        raise ValueError("No valid cut found")

    return best_cut_idx


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

    dpi = 72
    figsize = (width / dpi, height / dpi)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
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
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    plt.close(fig)

    return fig
