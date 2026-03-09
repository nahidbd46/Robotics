"""Detection of melt pools and spatters in additive-manufacturing images.

Uses classical image-processing techniques (thresholding, morphological
operations, and contour analysis) to locate melt-pool and spatter regions.
"""

from __future__ import annotations

import cv2
import numpy as np


class MeltPoolDetector:
    """Detect melt-pool and spatter regions via image processing.

    Parameters
    ----------
    melt_pool_thresh : int
        Brightness threshold used to segment the melt pool (high-intensity
        region at the laser interaction zone).
    spatter_thresh : int
        Brightness threshold for spatter particles (bright but smaller
        than the melt pool).
    min_melt_pool_area : int
        Minimum contour area (in pixels) to be considered a melt pool.
    max_spatter_area : int
        Maximum contour area for a detection to be classified as spatter.
    morph_kernel_size : int
        Size of the morphological kernel used for noise removal.
    """

    def __init__(
        self,
        melt_pool_thresh: int = 200,
        spatter_thresh: int = 180,
        min_melt_pool_area: int = 500,
        max_spatter_area: int = 400,
        morph_kernel_size: int = 5,
    ) -> None:
        self.melt_pool_thresh = melt_pool_thresh
        self.spatter_thresh = spatter_thresh
        self.min_melt_pool_area = min_melt_pool_area
        self.max_spatter_area = max_spatter_area
        self.morph_kernel_size = morph_kernel_size

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale and apply Gaussian blur."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return blurred

    def detect(self, image: np.ndarray) -> list[dict]:
        """Run detection on a single image.

        Parameters
        ----------
        image : np.ndarray
            Input image (BGR or grayscale).

        Returns
        -------
        list[dict]
            Each dict contains:
            - ``"bbox"``: ``(x, y, w, h)``
            - ``"label"``: ``"melt_pool"`` or ``"spatter"``
            - ``"area"``: contour area in pixels
            - ``"contour"``: the raw OpenCV contour
        """
        gray = self._preprocess(image)
        detections: list[dict] = []

        # --- Melt-pool detection (bright, large region) ---
        _, mp_mask = cv2.threshold(
            gray, self.melt_pool_thresh, 255, cv2.THRESH_BINARY
        )
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_size, self.morph_kernel_size),
        )
        mp_mask = cv2.morphologyEx(mp_mask, cv2.MORPH_CLOSE, kernel)
        mp_mask = cv2.morphologyEx(mp_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            mp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_melt_pool_area:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append(
                    {
                        "bbox": (x, y, w, h),
                        "label": "melt_pool",
                        "area": area,
                        "contour": cnt,
                    }
                )

        # --- Spatter detection (bright, small particles) ---
        _, sp_mask = cv2.threshold(
            gray, self.spatter_thresh, 255, cv2.THRESH_BINARY
        )
        sp_mask = cv2.morphologyEx(sp_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            sp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 0 < area <= self.max_spatter_area:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append(
                    {
                        "bbox": (x, y, w, h),
                        "label": "spatter",
                        "area": area,
                        "contour": cnt,
                    }
                )

        return detections

    def has_melt_pool(self, image: np.ndarray) -> bool:
        """Return ``True`` if at least one melt pool is detected."""
        detections = self.detect(image)
        return any(d["label"] == "melt_pool" for d in detections)

    def get_melt_pool_mask(self, image: np.ndarray) -> np.ndarray:
        """Return a binary mask highlighting the melt-pool region(s).

        Parameters
        ----------
        image : np.ndarray
            Input image (BGR or grayscale).

        Returns
        -------
        np.ndarray
            Binary mask with the same height/width as *image*.
        """
        h = image.shape[0]
        w = image.shape[1]
        mask = np.zeros((h, w), dtype=np.uint8)
        for det in self.detect(image):
            if det["label"] == "melt_pool":
                cv2.drawContours(mask, [det["contour"]], -1, 255, -1)
        return mask
