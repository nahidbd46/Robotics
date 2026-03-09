"""End-to-end pipeline for melt-pool and spatter analysis.

Combines detection, classification, and regeneration into a single
convenient entry point.
"""

from __future__ import annotations

import cv2
import numpy as np

from melt_pool_analysis.classifier import MeltPoolClassifier
from melt_pool_analysis.detector import MeltPoolDetector
from melt_pool_analysis.regenerator import MeltPoolRegenerator
from melt_pool_analysis.utils import draw_detections


class MeltPoolPipeline:
    """Full analysis pipeline.

    Parameters
    ----------
    detector : MeltPoolDetector or None
        Custom detector instance.  A default is created when ``None``.
    classifier : MeltPoolClassifier or None
        Custom classifier instance.  A default is created when ``None``.
    regenerator : MeltPoolRegenerator or None
        Custom regenerator instance.  A default is created when ``None``.
    """

    def __init__(
        self,
        detector: MeltPoolDetector | None = None,
        classifier: MeltPoolClassifier | None = None,
        regenerator: MeltPoolRegenerator | None = None,
    ) -> None:
        self.detector = detector or MeltPoolDetector()
        self.classifier = classifier or MeltPoolClassifier()
        self.regenerator = regenerator or MeltPoolRegenerator(self.detector)

    # ------------------------------------------------------------------
    # Single frame
    # ------------------------------------------------------------------

    def analyse_frame(self, frame: np.ndarray) -> dict:
        """Analyse a single frame.

        Steps:
        1. Detect melt-pool and spatter regions.
        2. Classify each detected region.

        Parameters
        ----------
        frame : np.ndarray
            Input image (BGR).

        Returns
        -------
        dict
            ``"detections"`` – list of detection dicts enriched with
            ``"classification"`` results.
            ``"has_melt_pool"`` – boolean.
            ``"annotated_frame"`` – image with drawn bounding boxes.
        """
        detections = self.detector.detect(frame)
        for det in detections:
            x, y, w, h = det["bbox"]
            crop = frame[y : y + h, x : x + w]
            if crop.size == 0:
                det["classification"] = {
                    "label": det["label"],
                    "class_id": None,
                    "confidence": 0.0,
                }
                continue
            det["classification"] = self.classifier.predict(crop)

        has_melt_pool = any(d["label"] == "melt_pool" for d in detections)
        annotated = draw_detections(frame, detections)

        return {
            "detections": detections,
            "has_melt_pool": has_melt_pool,
            "annotated_frame": annotated,
        }

    # ------------------------------------------------------------------
    # Sequence
    # ------------------------------------------------------------------

    def analyse_sequence(
        self,
        frames: list[np.ndarray],
        regenerate_missing: bool = True,
    ) -> list[dict]:
        """Analyse an ordered sequence of frames.

        Parameters
        ----------
        frames : list[np.ndarray]
            Video or image-sequence frames.
        regenerate_missing : bool
            If ``True``, frames with missing melt pools are regenerated
            before analysis.

        Returns
        -------
        list[dict]
            Per-frame analysis results.  Each dict has the same keys as
            :meth:`analyse_frame` plus ``"regenerated"`` (bool).
        """
        if regenerate_missing:
            regen_results = self.regenerator.process_sequence(frames)
        else:
            regen_results = [
                {"frame": f, "had_melt_pool": True, "regenerated": False}
                for f in frames
            ]

        outputs: list[dict] = []
        for rr in regen_results:
            result = self.analyse_frame(rr["frame"])
            result["regenerated"] = rr["regenerated"]
            outputs.append(result)
        return outputs

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @staticmethod
    def summary(results: list[dict]) -> dict:
        """Produce a summary of analysis results.

        Parameters
        ----------
        results : list[dict]
            Output of :meth:`analyse_sequence`.

        Returns
        -------
        dict
            High-level statistics including total frames, frames with
            melt pools, regenerated count, and classification counts.
        """
        total = len(results)
        with_mp = sum(1 for r in results if r["has_melt_pool"])
        regenerated = sum(1 for r in results if r.get("regenerated", False))

        class_counts: dict[str, int] = {}
        for r in results:
            for det in r["detections"]:
                cls_info = det.get("classification", {})
                label = cls_info.get("label", det.get("label", "unknown"))
                class_counts[label] = class_counts.get(label, 0) + 1

        return {
            "total_frames": total,
            "frames_with_melt_pool": with_mp,
            "frames_regenerated": regenerated,
            "classification_counts": class_counts,
        }
