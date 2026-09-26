"""Vision stream: linear spectrogram -> ResNet50 -> temporal pyramid pooling (14336-d).

Faithful port of ``vision_stream/src/representation_linear_spectogram.py`` (n_fft=1024,
hop=256, power spectrogram in dB, normalized to [0,1], 3-channel) +
``inference_resnet50.py`` (ImageNet ResNet50, 224-frame chunks, avgpool -> 2048) +
``flattening_temproal_pyramid.py`` (levels=3 -> 7 bins -> 14336). Requires the ``vision``
extra (torch + torchvision).
"""
from __future__ import annotations

import numpy as np

from .base import BaseStream
from .schemas import VISION_SCHEMA
from .pooling import temporal_pyramid_pool
from ._backends import require_torch, require_torchvision, torch_device

# spectrogram config
N_FFT = 1024
HOP_LENGTH = 256
TOP_DB = 80.0
# resnet config
MODEL_INPUT_SIZE = 224
CHUNK_FRAMES = 224
STRIDE_FRAMES = 224
BATCH_SIZE = 64
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PYRAMID_LEVELS = 3   # 2**3 - 1 = 7 bins; 7 x 2048 = 14336


class VisionStream(BaseStream):
    name = "vision"
    schema = VISION_SCHEMA
    extra = "vision"

    def __init__(self, device=None):
        self._device = device
        self._model = None

    def _ensure_backbone(self):
        if self._model is not None:
            return
        require_torch(self.extra)
        require_torchvision(self.extra)
        import torch
        import torch.nn as nn
        from torchvision.models import resnet50, ResNet50_Weights

        self._device = self._device or torch_device()
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(self._device).eval()

        class ResNet50Feat(nn.Module):
            def __init__(self, b):
                super().__init__()
                self.stem = nn.Sequential(b.conv1, b.bn1, b.relu, b.maxpool)
                self.layer1, self.layer2 = b.layer1, b.layer2
                self.layer3, self.layer4 = b.layer3, b.layer4
                self.pool = b.avgpool

            def forward(self, x):
                x = self.stem(x)
                x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
                return torch.flatten(self.pool(x), 1)  # N x 2048

        self._torch = torch
        self._model = ResNet50Feat(base).to(self._device).eval()

    # -- spectrogram (linear, 3-channel, normalized) ---------------------------
    @staticmethod
    def _spectrogram(y: np.ndarray, sr: int) -> np.ndarray:
        import librosa
        D = librosa.stft(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH)
        S = np.abs(D) ** 2.0
        S_db = librosa.power_to_db(S, ref=np.max, top_db=TOP_DB)
        S_norm = np.clip((S_db + TOP_DB) / TOP_DB, 0.0, 1.0).astype(np.float32)
        return np.stack([S_norm, S_norm, S_norm], axis=-1)  # H x W x 3

    def _extract_frames(self, y: np.ndarray, sr: int) -> np.ndarray:
        self._ensure_backbone()
        import torch.nn.functional as F
        torch = self._torch

        spec3 = self._spectrogram(y, sr)          # H x W x 3
        H, W, C = spec3.shape

        # split along time (width) into 224-frame chunks, zero-padded last
        chunks = []
        if W <= CHUNK_FRAMES:
            pad = np.zeros((H, CHUNK_FRAMES - W, C), dtype=spec3.dtype)
            chunks.append(np.concatenate([spec3, pad], axis=1))
        else:
            for start in range(0, max(W - CHUNK_FRAMES, 0) + 1, STRIDE_FRAMES):
                end = start + CHUNK_FRAMES
                sl = spec3[:, start:end, :] if end <= W else np.concatenate(
                    [spec3[:, start:W, :], np.zeros((H, end - W, C), dtype=spec3.dtype)], axis=1)
                chunks.append(sl)

        mean = torch.tensor(IMAGENET_MEAN).view(1, -1, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, -1, 1, 1)
        feats = []
        with torch.inference_mode():
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = [torch.from_numpy(sl).permute(2, 0, 1).unsqueeze(0) for sl in chunks[i:i + BATCH_SIZE]]
                x = torch.cat(batch, dim=0).to(self._device)
                x = F.interpolate(x, size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                                  mode="bilinear", align_corners=False)
                x = (x - mean.to(x.device)) / std.to(x.device)
                feats.append(self._model(x).cpu().numpy())
        return np.concatenate(feats, axis=0).astype(np.float32) if feats \
            else np.zeros((0, 2048), dtype=np.float32)

    def _pool(self, frames: np.ndarray) -> np.ndarray:
        return temporal_pyramid_pool(frames, levels=PYRAMID_LEVELS)
