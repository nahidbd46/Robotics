"""Melt-pool regeneration for frames with missing melt pools.

Offers two complementary strategies:

1. **Inpainting** – uses OpenCV inpainting to fill the expected melt-pool
   region when the raw frame shows it is absent.
2. **Temporal interpolation** – reconstructs the melt pool by blending
   neighbouring frames that *do* contain a melt pool.
"""

from __future__ import annotations

import cv2
import numpy as np

from melt_pool_analysis.detector import MeltPoolDetector


class MeltPoolRegenerator:
    """Regenerate missing melt pools in a sequence of frames.

    Parameters
    ----------
    detector : MeltPoolDetector or None
        Detector instance used to determine whether a melt pool is present.
        If ``None``, a default detector is created.
    inpaint_radius : int
        Radius for the OpenCV inpainting algorithm.
    """

    def __init__(
        self,
        detector: MeltPoolDetector | None = None,
        inpaint_radius: int = 5,
    ) -> None:
        self.detector = detector or MeltPoolDetector()
        self.inpaint_radius = inpaint_radius

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _estimate_melt_pool_region(
        self, reference_frames: list[np.ndarray]
    ) -> np.ndarray | None:
        """Build an average melt-pool mask from reference frames.

        Parameters
        ----------
        reference_frames : list[np.ndarray]
            Frames that are known to contain melt pools.

        Returns
        -------
        np.ndarray or None
            Averaged binary mask, or ``None`` if no melt pool was found
            in any reference frame.
        """
        masks: list[np.ndarray] = []
        for frame in reference_frames:
            mask = self.detector.get_melt_pool_mask(frame)
            if mask.any():
                masks.append(mask.astype(np.float32) / 255.0)
        if not masks:
            return None
        avg = np.mean(masks, axis=0)
        return (avg > 0.3).astype(np.uint8) * 255

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def regenerate_inpaint(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Regenerate melt pool via OpenCV inpainting.

        Parameters
        ----------
        frame : np.ndarray
            Frame with a missing melt pool.
        mask : np.ndarray
            Binary mask indicating where the melt pool *should* be.

        Returns
        -------
        np.ndarray
            Inpainted frame.
        """
        return cv2.inpaint(
            frame, mask, self.inpaint_radius, cv2.INPAINT_TELEA
        )

    def regenerate_temporal(
        self,
        missing_frame: np.ndarray,
        prev_frame: np.ndarray | None,
        next_frame: np.ndarray | None,
    ) -> np.ndarray:
        """Regenerate a missing melt pool by blending temporal neighbours.

        The melt-pool region from the previous and/or next frame is alpha-
        blended into the frame that is missing its melt pool.

        Parameters
        ----------
        missing_frame : np.ndarray
            The frame with no melt pool detected.
        prev_frame, next_frame : np.ndarray or None
            Neighbouring frames that contain a melt pool.

        Returns
        -------
        np.ndarray
            Frame with the melt-pool region reconstructed.
        """
        refs = [f for f in (prev_frame, next_frame) if f is not None]
        if not refs:
            return missing_frame.copy()

        mask = self._estimate_melt_pool_region(refs)
        if mask is None:
            return missing_frame.copy()

        # Build a composite melt-pool patch from the available neighbours
        patches = []
        for ref in refs:
            patches.append(ref.astype(np.float32))
        avg_patch = np.mean(patches, axis=0).astype(np.uint8)

        # Alpha blend into missing frame
        if len(missing_frame.shape) == 2:
            alpha = mask.astype(np.float32) / 255.0
        else:
            alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(
                np.float32
            ) / 255.0

        result = (
            (1 - alpha) * missing_frame.astype(np.float32)
            + alpha * avg_patch.astype(np.float32)
        )
        return result.astype(np.uint8)

    def process_sequence(
        self, frames: list[np.ndarray]
    ) -> list[dict]:
        """Process a sequence of frames, regenerating missing melt pools.

        Parameters
        ----------
        frames : list[np.ndarray]
            Ordered list of video / image-sequence frames.

        Returns
        -------
        list[dict]
            One entry per frame with keys:
            - ``"frame"``: the (possibly regenerated) frame
            - ``"had_melt_pool"``: whether the original frame had a melt pool
            - ``"regenerated"``: whether the melt pool was regenerated
        """
        has_mp = [self.detector.has_melt_pool(f) for f in frames]
        results: list[dict] = []

        for i, frame in enumerate(frames):
            if has_mp[i]:
                results.append(
                    {
                        "frame": frame,
                        "had_melt_pool": True,
                        "regenerated": False,
                    }
                )
            else:
                # Find nearest neighbours with melt pools
                prev_ref = None
                for j in range(i - 1, -1, -1):
                    if has_mp[j]:
                        prev_ref = frames[j]
                        break
                next_ref = None
                for j in range(i + 1, len(frames)):
                    if has_mp[j]:
                        next_ref = frames[j]
                        break

                regenerated = self.regenerate_temporal(
                    frame, prev_ref, next_ref
                )
                results.append(
                    {
                        "frame": regenerated,
                        "had_melt_pool": False,
                        "regenerated": True,
                    }
                )

        return results
