"""Lightweight IoU-based tracker that associates PPE items with detected persons.

This module does NOT attempt full multi-object tracking (no Kalman filters or
re-ID). Instead it performs per-frame spatial association:

1. Separate detections into *persons* and *PPE items*.
2. For each person, find which PPE items overlap with their bounding box.
3. Determine which required PPE items are present and which are missing.

A separate temporal layer (:class:`ViolationTimer` in ``alert.py``) decides
when a missing-PPE observation becomes a reportable violation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config import (
    PERSON_CLASS_ID,
    PPE_ASSOCIATION_IOU,
    PPE_CLASSES,
    REQUIRED_PPE,
)
from src.detector import Detection


@dataclass(slots=True)
class PersonStatus:
    """PPE compliance status for a single detected person."""
    person: Detection
    present_ppe: dict[str, Detection] = field(default_factory=dict)
    missing_ppe: set[str] = field(default_factory=set)

    @property
    def is_compliant(self) -> bool:
        return len(self.missing_ppe) == 0


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Compute Intersection-over-Union between two (x1, y1, x2, y2) boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def _contained_ratio(
    ppe_box: tuple[int, int, int, int],
    person_box: tuple[int, int, int, int],
) -> float:
    """Fraction of the PPE box area that falls inside the person box.

    This is more suitable than IoU for associating small PPE items
    (helmets, gloves) with larger person boxes.
    """
    x1 = max(ppe_box[0], person_box[0])
    y1 = max(ppe_box[1], person_box[1])
    x2 = min(ppe_box[2], person_box[2])
    y2 = min(ppe_box[3], person_box[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ppe_area = (ppe_box[2] - ppe_box[0]) * (ppe_box[3] - ppe_box[1])
    if ppe_area == 0:
        return 0.0
    return inter / ppe_area


def associate(detections: list[Detection]) -> list[PersonStatus]:
    """Associate PPE items with persons detected in a single frame.

    Returns one :class:`PersonStatus` per detected person.
    """
    persons = [d for d in detections if d.class_id == PERSON_CLASS_ID]
    ppe_items = [d for d in detections if d.class_id in PPE_CLASSES]

    results: list[PersonStatus] = []

    for person in persons:
        status = PersonStatus(person=person)

        for ppe in ppe_items:
            overlap = max(
                _iou(ppe.bbox, person.bbox),
                _contained_ratio(ppe.bbox, person.bbox),
            )
            if overlap < PPE_ASSOCIATION_IOU:
                continue

            label = PPE_CLASSES[ppe.class_id]
            # keep the highest-confidence detection per PPE type
            if label not in status.present_ppe or ppe.confidence > status.present_ppe[label].confidence:
                status.present_ppe[label] = ppe

        status.missing_ppe = REQUIRED_PPE - set(status.present_ppe.keys())
        results.append(status)

    return results
