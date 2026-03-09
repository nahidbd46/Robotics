"""CNN-based classifier for melt pools and spatters.

Provides a lightweight convolutional neural network that classifies
cropped regions into one of three categories:

* ``normal_melt_pool`` – healthy melt pool
* ``abnormal_melt_pool`` – defective / irregular melt pool
* ``spatter`` – ejected particles
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

CLASS_NAMES: list[str] = ["normal_melt_pool", "abnormal_melt_pool", "spatter"]


class _ClassifierNet(nn.Module):
    """Small CNN for melt-pool / spatter classification."""

    def __init__(self, num_classes: int = 3, input_size: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class MeltPoolClassifier:
    """Train and run a CNN classifier for melt-pool / spatter crops.

    Parameters
    ----------
    input_size : int
        Height and width to which each crop is resized before inference.
    num_classes : int
        Number of output classes (default ``3``).
    device : str or None
        PyTorch device string.  ``None`` selects CUDA when available.
    """

    def __init__(
        self,
        input_size: int = 64,
        num_classes: int = 3,
        device: str | None = None,
    ) -> None:
        self.input_size = input_size
        self.num_classes = num_classes
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        self.model = _ClassifierNet(num_classes, input_size).to(self.device)
        self.class_names = CLASS_NAMES[:num_classes]

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def _prepare_crop(self, crop: np.ndarray) -> torch.Tensor:
        """Resize, normalise and convert a single crop to a tensor."""
        import cv2

        if len(crop.shape) == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(
            crop, (self.input_size, self.input_size)
        ).astype(np.float32)
        resized /= 255.0
        return torch.from_numpy(resized).unsqueeze(0)  # (1, H, W)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        crops: Sequence[np.ndarray],
        labels: Sequence[int],
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-3,
    ) -> list[float]:
        """Train the classifier on a collection of labelled crops.

        Parameters
        ----------
        crops : sequence of np.ndarray
            Image crops (grayscale or BGR).
        labels : sequence of int
            Integer class labels corresponding to *crops*.
        epochs : int
            Number of training epochs.
        batch_size : int
            Mini-batch size.
        lr : float
            Learning rate for the Adam optimiser.

        Returns
        -------
        list[float]
            Per-epoch training loss.
        """
        tensors = torch.stack([self._prepare_crop(c) for c in crops])
        targets = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(tensors, targets)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        losses: list[float] = []
        for _ in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                out = self.model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)
            losses.append(epoch_loss / len(dataset))
        return losses

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, crop: np.ndarray) -> dict:
        """Classify a single crop.

        Parameters
        ----------
        crop : np.ndarray
            Image crop (grayscale or BGR).

        Returns
        -------
        dict
            ``{"label": str, "class_id": int, "confidence": float}``
        """
        self.model.eval()
        tensor = self._prepare_crop(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, cls_id = probs.max(dim=1)
        return {
            "label": self.class_names[cls_id.item()],
            "class_id": cls_id.item(),
            "confidence": conf.item(),
        }

    def predict_batch(self, crops: Sequence[np.ndarray]) -> list[dict]:
        """Classify a batch of crops.

        Parameters
        ----------
        crops : sequence of np.ndarray
            Image crops.

        Returns
        -------
        list[dict]
            One result dict per crop (same format as :meth:`predict`).
        """
        self.model.eval()
        tensors = torch.stack(
            [self._prepare_crop(c) for c in crops]
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(tensors)
            probs = torch.softmax(logits, dim=1)
            confs, cls_ids = probs.max(dim=1)
        results = []
        for i in range(len(crops)):
            results.append(
                {
                    "label": self.class_names[cls_ids[i].item()],
                    "class_id": cls_ids[i].item(),
                    "confidence": confs[i].item(),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save model weights to *path*."""
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model weights from *path*."""
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.model.eval()
