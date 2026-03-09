"""Utility functions for image I/O and visualization."""

import os

import cv2
import numpy as np


def load_image(path: str, grayscale: bool = False) -> np.ndarray:
    """Load an image from disk.

    Parameters
    ----------
    path : str
        Path to the image file.
    grayscale : bool
        If ``True``, load the image in grayscale.

    Returns
    -------
    np.ndarray
        The loaded image in BGR or grayscale format.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def save_image(path: str, image: np.ndarray) -> None:
    """Save an image to disk.

    Parameters
    ----------
    path : str
        Destination file path.
    image : np.ndarray
        Image array to save.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, image)


def draw_detections(
    image: np.ndarray,
    detections: list[dict],
    color_map: dict[str, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Draw bounding boxes and labels on an image.

    Parameters
    ----------
    image : np.ndarray
        BGR image to annotate.
    detections : list[dict]
        Each dict must contain ``"bbox"`` (x, y, w, h) and ``"label"``.
    color_map : dict, optional
        Mapping from label string to BGR colour tuple.

    Returns
    -------
    np.ndarray
        Annotated copy of *image*.
    """
    if color_map is None:
        color_map = {
            "melt_pool": (0, 255, 0),
            "spatter": (0, 0, 255),
            "missing_melt_pool": (0, 255, 255),
        }
    vis = image.copy()
    for det in detections:
        x, y, w, h = det["bbox"]
        label = det.get("label", "unknown")
        color = color_map.get(label, (255, 255, 255))
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            vis, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )
    return vis


def compute_iou(box_a: tuple, box_b: tuple) -> float:
    """Compute Intersection-over-Union for two (x, y, w, h) boxes.

    Parameters
    ----------
    box_a, box_b : tuple
        Bounding boxes as ``(x, y, w, h)``.

    Returns
    -------
    float
        IoU value in ``[0, 1]``.
    """
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    if union == 0:
        return 0.0
    return inter / union
