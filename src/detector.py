"""Object detection backend supporting both PyTorch (YOLOv8) and ONNX Runtime."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Sequence

import cv2
import numpy as np

from src.config import (
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    ONNX_WEIGHTS,
    YOLO_WEIGHTS,
)


class Backend(Enum):
    PYTORCH = auto()
    ONNX = auto()


@dataclass(frozen=True, slots=True)
class Detection:
    """Single detected object."""
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    class_id: int
    confidence: float


class Detector:
    """Unified detector that wraps either Ultralytics YOLOv8 or ONNX Runtime.

    Usage::

        det = Detector(backend=Backend.ONNX)
        detections, inference_ms = det.detect(frame)
    """

    def __init__(
        self,
        backend: Backend = Backend.ONNX,
        conf: float = CONFIDENCE_THRESHOLD,
        iou: float = IOU_THRESHOLD,
        device: str = "cpu",
    ) -> None:
        self.backend = backend
        self.conf = conf
        self.iou = iou
        self._model = None
        self._session = None

        if backend is Backend.PYTORCH:
            self._init_pytorch(device)
        else:
            self._init_onnx()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_pytorch(self, device: str) -> None:
        from ultralytics import YOLO

        self._model = YOLO(str(YOLO_WEIGHTS))
        self._model.to(device)

    def _init_onnx(self) -> None:
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        self._session = ort.InferenceSession(str(ONNX_WEIGHTS), providers=providers)
        meta = self._session.get_inputs()[0]
        self._input_name = meta.name
        _, _, self._input_h, self._input_w = meta.shape

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> tuple[list[Detection], float]:
        """Run detection on a BGR frame.

        Returns:
            (detections, inference_time_ms)
        """
        if self.backend is Backend.PYTORCH:
            return self._detect_pytorch(frame)
        return self._detect_onnx(frame)

    # ------------------------------------------------------------------
    # PyTorch path
    # ------------------------------------------------------------------

    def _detect_pytorch(self, frame: np.ndarray) -> tuple[list[Detection], float]:
        t0 = time.perf_counter()
        results = self._model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        detections: list[Detection] = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            detections.append(Detection(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                class_id=int(box.cls[0]),
                confidence=float(box.conf[0]),
            ))
        return detections, elapsed

    # ------------------------------------------------------------------
    # ONNX path
    # ------------------------------------------------------------------

    def _detect_onnx(self, frame: np.ndarray) -> tuple[list[Detection], float]:
        blob, scale, pad_w, pad_h = self._preprocess(frame)

        t0 = time.perf_counter()
        outputs = self._session.run(None, {self._input_name: blob})
        elapsed = (time.perf_counter() - t0) * 1000

        detections = self._postprocess(outputs[0], scale, pad_w, pad_h, frame.shape)
        return detections, elapsed

    def _preprocess(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, float, int, int]:
        """Letterbox-resize and normalise to [1, 3, H, W] float32."""
        h, w = frame.shape[:2]
        scale = min(self._input_w / w, self._input_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        pad_w, pad_h = (self._input_w - new_w) // 2, (self._input_h - new_h) // 2

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full(
            (self._input_h, self._input_w, 3), 114, dtype=np.uint8
        )
        canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        blob = canvas.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # HWC -> 1CHW
        return blob, scale, pad_w, pad_h

    def _postprocess(
        self,
        output: np.ndarray,
        scale: float,
        pad_w: int,
        pad_h: int,
        orig_shape: tuple[int, ...],
    ) -> list[Detection]:
        """Decode raw ONNX output (1, 84, 8400) into Detections."""
        # output shape: (1, 4 + num_classes, num_anchors)
        preds = output[0].T  # (8400, 84)
        boxes_xywh = preds[:, :4]
        scores = preds[:, 4:]

        class_ids = scores.argmax(axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]

        mask = confidences >= self.conf
        boxes_xywh = boxes_xywh[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        if len(boxes_xywh) == 0:
            return []

        # xywh centre -> xyxy
        x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

        # undo letterbox padding + scale
        x1 = ((x1 - pad_w) / scale).clip(0)
        y1 = ((y1 - pad_h) / scale).clip(0)
        x2 = ((x2 - pad_w) / scale).clip(0, orig_shape[1])
        y2 = ((y2 - pad_h) / scale).clip(0, orig_shape[0])

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # NMS
        indices = self._nms(boxes_xyxy, confidences)

        detections: list[Detection] = []
        for i in indices:
            detections.append(Detection(
                bbox=(int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])),
                class_id=int(class_ids[i]),
                confidence=float(confidences[i]),
            ))
        return detections

    def _nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
    ) -> list[int]:
        """Class-agnostic greedy NMS."""
        order = scores.argsort()[::-1]
        keep: list[int] = []

        while order.size > 0:
            i = order[0]
            keep.append(int(i))

            if order.size == 1:
                break

            rest = order[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
            iou = inter / (area_i + area_rest - inter + 1e-6)

            order = rest[iou <= self.iou]

        return keep
