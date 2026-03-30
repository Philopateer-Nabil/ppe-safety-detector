"""Violation timing, visual alerting, and CSV logging.

The :class:`ViolationTracker` maintains per-person timers so that a missing-PPE
observation must persist for :data:`VIOLATION_HOLD_SEC` consecutive seconds
before being escalated into a logged violation.
"""

from __future__ import annotations

import csv
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src.config import (
    BORDER_THICKNESS,
    COLOR_COMPLIANT,
    COLOR_TEXT_BG,
    COLOR_VIOLATION,
    FONT_SCALE,
    VIOLATION_HOLD_SEC,
    VIOLATION_LOG,
)
from src.tracker import PersonStatus


class ViolationTracker:
    """Track how long each person has been missing required PPE.

    Persons are keyed by a spatial hash of their bounding-box centre so that
    small movements between frames don't reset the timer.
    """

    def __init__(self, grid_size: int = 80) -> None:
        self._grid = grid_size
        # key: spatial bucket  ->  timestamp when non-compliance first seen
        self._timers: dict[tuple[int, int], float] = {}
        self._logged: set[tuple[int, int]] = set()

    def _bucket(self, bbox: tuple[int, int, int, int]) -> tuple[int, int]:
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        return (cx // self._grid, cy // self._grid)

    def update(self, statuses: list[PersonStatus]) -> list[tuple[PersonStatus, bool]]:
        """Process a frame's worth of person statuses.

        Returns:
            List of (status, is_violation) tuples.  ``is_violation`` is True
            only when the hold timer has elapsed.
        """
        now = time.monotonic()
        active_buckets: set[tuple[int, int]] = set()
        results: list[tuple[PersonStatus, bool]] = []

        for status in statuses:
            bucket = self._bucket(status.person.bbox)
            active_buckets.add(bucket)

            if status.is_compliant:
                self._timers.pop(bucket, None)
                self._logged.discard(bucket)
                results.append((status, False))
                continue

            if bucket not in self._timers:
                self._timers[bucket] = now

            elapsed = now - self._timers[bucket]
            is_violation = elapsed >= VIOLATION_HOLD_SEC

            if is_violation and bucket not in self._logged:
                _log_violation(status)
                self._logged.add(bucket)

            results.append((status, is_violation))

        # purge stale buckets
        stale = set(self._timers) - active_buckets
        for b in stale:
            self._timers.pop(b, None)
            self._logged.discard(b)

        return results


def _log_violation(status: PersonStatus) -> None:
    """Append a violation record to the CSV log."""
    path = Path(VIOLATION_LOG)
    write_header = not path.exists()

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "bbox", "missing_ppe"])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            status.person.bbox,
            ", ".join(sorted(status.missing_ppe)),
        ])


# -----------------------------------------------------------------------
# Drawing helpers
# -----------------------------------------------------------------------

def draw_results(
    frame: np.ndarray,
    results: list[tuple[PersonStatus, bool]],
    fps: float,
) -> np.ndarray:
    """Draw bounding boxes, PPE status labels, and FPS overlay on *frame*."""
    overlay = frame.copy()

    for status, is_violation in results:
        color = COLOR_VIOLATION if is_violation else COLOR_COMPLIANT
        x1, y1, x2, y2 = status.person.bbox

        # bounding box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, BORDER_THICKNESS)

        # semi-transparent fill when violating
        if is_violation:
            roi = overlay[y1:y2, x1:x2]
            tint = np.full_like(roi, COLOR_VIOLATION)
            cv2.addWeighted(tint, 0.18, roi, 0.82, 0, roi)

        # label
        missing = sorted(status.missing_ppe) if status.missing_ppe else []
        if is_violation:
            label = f"VIOLATION: missing {', '.join(missing)}"
        elif missing:
            label = f"Warning: {', '.join(missing)}"
        else:
            present = sorted(status.present_ppe.keys())
            label = f"OK: {', '.join(present)}" if present else "OK"

        _draw_label(overlay, label, (x1, y1 - 8), color)

        # draw PPE item boxes
        for ppe_name, ppe_det in status.present_ppe.items():
            px1, py1, px2, py2 = ppe_det.bbox
            cv2.rectangle(overlay, (px1, py1), (px2, py2), COLOR_COMPLIANT, 2)
            _draw_label(overlay, ppe_name, (px1, py1 - 6), COLOR_COMPLIANT)

    # FPS counter
    _draw_label(overlay, f"FPS: {fps:.1f}", (10, 28), (255, 200, 0))

    return overlay


def draw_alert_banner(frame: np.ndarray, text: str) -> np.ndarray:
    """Draw a full-width red banner at the top of the frame."""
    h, w = frame.shape[:2]
    banner_h = 48
    cv2.rectangle(frame, (0, 0), (w, banner_h), COLOR_VIOLATION, -1)
    cv2.putText(
        frame,
        text,
        (12, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def _draw_label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    """Draw text with a dark background rectangle for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, FONT_SCALE, 1)
    x, y = origin
    y = max(y, th + 4)  # prevent drawing above frame
    cv2.rectangle(frame, (x, y - th - 4), (x + tw + 6, y + baseline + 2), COLOR_TEXT_BG, -1)
    cv2.putText(frame, text, (x + 3, y), font, FONT_SCALE, color, 1, cv2.LINE_AA)
