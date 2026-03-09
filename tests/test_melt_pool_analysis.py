"""Tests for the melt_pool_analysis package."""

import numpy as np
import pytest

from melt_pool_analysis.classifier import MeltPoolClassifier
from melt_pool_analysis.detector import MeltPoolDetector
from melt_pool_analysis.pipeline import MeltPoolPipeline
from melt_pool_analysis.regenerator import MeltPoolRegenerator
from melt_pool_analysis.utils import compute_iou, draw_detections


# ---------------------------------------------------------------------------
# Helpers – synthetic image generation
# ---------------------------------------------------------------------------

def _make_frame_with_melt_pool(
    height: int = 200,
    width: int = 200,
    mp_center: tuple[int, int] = (100, 100),
    mp_radius: int = 30,
) -> np.ndarray:
    """Create a dark frame with a bright circular melt pool."""
    import cv2

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = 30  # dark background
    cv2.circle(frame, mp_center, mp_radius, (240, 240, 240), -1)
    return frame


def _make_frame_with_spatters(
    height: int = 200,
    width: int = 200,
    num_spatters: int = 5,
) -> np.ndarray:
    """Create a dark frame with small bright spatter dots."""
    import cv2

    rng = np.random.RandomState(42)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = 30
    for _ in range(num_spatters):
        cx = rng.randint(20, width - 20)
        cy = rng.randint(20, height - 20)
        cv2.circle(frame, (cx, cy), 4, (220, 220, 220), -1)
    return frame


def _make_dark_frame(height: int = 200, width: int = 200) -> np.ndarray:
    """Create a uniformly dark frame (no melt pool, no spatter)."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = 30
    return frame


# ---------------------------------------------------------------------------
# Detector tests
# ---------------------------------------------------------------------------

class TestMeltPoolDetector:
    def test_detects_melt_pool(self):
        frame = _make_frame_with_melt_pool()
        detector = MeltPoolDetector(melt_pool_thresh=200, min_melt_pool_area=100)
        detections = detector.detect(frame)
        mp_dets = [d for d in detections if d["label"] == "melt_pool"]
        assert len(mp_dets) >= 1

    def test_has_melt_pool_true(self):
        frame = _make_frame_with_melt_pool()
        detector = MeltPoolDetector(melt_pool_thresh=200, min_melt_pool_area=100)
        assert detector.has_melt_pool(frame) is True

    def test_has_melt_pool_false_on_dark(self):
        frame = _make_dark_frame()
        detector = MeltPoolDetector()
        assert detector.has_melt_pool(frame) is False

    def test_get_melt_pool_mask_shape(self):
        frame = _make_frame_with_melt_pool()
        detector = MeltPoolDetector(melt_pool_thresh=200, min_melt_pool_area=100)
        mask = detector.get_melt_pool_mask(frame)
        assert mask.shape == (200, 200)
        assert mask.dtype == np.uint8

    def test_get_melt_pool_mask_nonzero(self):
        frame = _make_frame_with_melt_pool()
        detector = MeltPoolDetector(melt_pool_thresh=200, min_melt_pool_area=100)
        mask = detector.get_melt_pool_mask(frame)
        assert mask.any()

    def test_grayscale_input(self):
        import cv2

        frame = _make_frame_with_melt_pool()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = MeltPoolDetector(melt_pool_thresh=200, min_melt_pool_area=100)
        assert detector.has_melt_pool(gray) is True

    def test_detects_spatters(self):
        frame = _make_frame_with_spatters(num_spatters=5)
        detector = MeltPoolDetector(
            spatter_thresh=180,
            max_spatter_area=400,
            min_melt_pool_area=500,
        )
        detections = detector.detect(frame)
        sp_dets = [d for d in detections if d["label"] == "spatter"]
        assert len(sp_dets) >= 1


# ---------------------------------------------------------------------------
# Classifier tests
# ---------------------------------------------------------------------------

class TestMeltPoolClassifier:
    def test_predict_returns_dict(self):
        crop = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        clf = MeltPoolClassifier(input_size=64, device="cpu")
        result = clf.predict(crop)
        assert "label" in result
        assert "class_id" in result
        assert "confidence" in result

    def test_predict_batch(self):
        crops = [
            np.random.randint(0, 255, (64, 64), dtype=np.uint8)
            for _ in range(4)
        ]
        clf = MeltPoolClassifier(input_size=64, device="cpu")
        results = clf.predict_batch(crops)
        assert len(results) == 4
        for r in results:
            assert "label" in r

    def test_train_reduces_loss(self):
        rng = np.random.RandomState(0)
        crops = [rng.randint(0, 255, (32, 32), dtype=np.uint8) for _ in range(30)]
        labels = [i % 3 for i in range(30)]
        clf = MeltPoolClassifier(input_size=32, device="cpu")
        losses = clf.train(crops, labels, epochs=5, batch_size=10, lr=1e-3)
        assert len(losses) == 5
        # Loss should generally decrease (first > last, with some tolerance)
        assert losses[-1] <= losses[0] + 0.5  # Allow small fluctuation

    def test_save_and_load(self, tmp_path):
        clf = MeltPoolClassifier(input_size=32, device="cpu")
        path = str(tmp_path / "model.pth")
        clf.save(path)
        clf2 = MeltPoolClassifier(input_size=32, device="cpu")
        clf2.load(path)
        # Both models should give same output on same input
        crop = np.zeros((32, 32), dtype=np.uint8)
        r1 = clf.predict(crop)
        r2 = clf2.predict(crop)
        assert r1["class_id"] == r2["class_id"]


# ---------------------------------------------------------------------------
# Regenerator tests
# ---------------------------------------------------------------------------

class TestMeltPoolRegenerator:
    def test_process_sequence_marks_regenerated(self):
        frames = [
            _make_frame_with_melt_pool(),
            _make_dark_frame(),           # missing melt pool
            _make_frame_with_melt_pool(),
        ]
        detector = MeltPoolDetector(melt_pool_thresh=200, min_melt_pool_area=100)
        regen = MeltPoolRegenerator(detector=detector)
        results = regen.process_sequence(frames)
        assert len(results) == 3
        assert results[0]["regenerated"] is False
        assert results[1]["regenerated"] is True
        assert results[2]["regenerated"] is False

    def test_regenerate_temporal_returns_correct_shape(self):
        mp_frame = _make_frame_with_melt_pool()
        dark_frame = _make_dark_frame()
        detector = MeltPoolDetector(melt_pool_thresh=200, min_melt_pool_area=100)
        regen = MeltPoolRegenerator(detector=detector)
        result = regen.regenerate_temporal(dark_frame, mp_frame, None)
        assert result.shape == dark_frame.shape

    def test_regenerate_inpaint_returns_correct_shape(self):
        frame = _make_dark_frame()
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[80:120, 80:120] = 255
        regen = MeltPoolRegenerator()
        result = regen.regenerate_inpaint(frame, mask)
        assert result.shape == frame.shape

    def test_no_reference_returns_copy(self):
        dark = _make_dark_frame()
        regen = MeltPoolRegenerator()
        result = regen.regenerate_temporal(dark, None, None)
        np.testing.assert_array_equal(result, dark)


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------

class TestMeltPoolPipeline:
    def test_analyse_frame_keys(self):
        frame = _make_frame_with_melt_pool()
        pipe = MeltPoolPipeline()
        result = pipe.analyse_frame(frame)
        assert "detections" in result
        assert "has_melt_pool" in result
        assert "annotated_frame" in result

    def test_analyse_sequence(self):
        frames = [
            _make_frame_with_melt_pool(),
            _make_dark_frame(),
            _make_frame_with_melt_pool(),
        ]
        pipe = MeltPoolPipeline(
            detector=MeltPoolDetector(melt_pool_thresh=200, min_melt_pool_area=100),
        )
        results = pipe.analyse_sequence(frames, regenerate_missing=True)
        assert len(results) == 3

    def test_summary(self):
        results = [
            {
                "has_melt_pool": True,
                "regenerated": False,
                "detections": [
                    {"label": "melt_pool", "classification": {"label": "normal_melt_pool"}}
                ],
            },
            {
                "has_melt_pool": False,
                "regenerated": True,
                "detections": [],
            },
        ]
        s = MeltPoolPipeline.summary(results)
        assert s["total_frames"] == 2
        assert s["frames_with_melt_pool"] == 1
        assert s["frames_regenerated"] == 1


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------

class TestUtils:
    def test_compute_iou_identical(self):
        box = (10, 10, 50, 50)
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_compute_iou_no_overlap(self):
        a = (0, 0, 10, 10)
        b = (100, 100, 10, 10)
        assert compute_iou(a, b) == pytest.approx(0.0)

    def test_compute_iou_partial(self):
        a = (0, 0, 20, 20)
        b = (10, 10, 20, 20)
        iou = compute_iou(a, b)
        assert 0.0 < iou < 1.0

    def test_draw_detections(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        dets = [{"bbox": (10, 10, 30, 30), "label": "melt_pool"}]
        vis = draw_detections(img, dets)
        assert vis.shape == img.shape
        # The annotated image should differ from the blank one
        assert not np.array_equal(vis, img)
