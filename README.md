# PPE Safety Detector

A model-agnostic, real-time PPE (Personal Protective Equipment) compliance detection pipeline built with YOLOv8, ONNX Runtime, and Streamlit.

This project focuses on the **inference pipeline, alerting logic, and system architecture** — not the model itself. It ships with a pretrained YOLOv8n (COCO) for person detection out of the box, and is designed to drop in any custom-trained YOLOv8 PPE model with a single config change.

## What This Project Demonstrates

- **Dual inference backend** — seamlessly switch between PyTorch and ONNX Runtime, with hand-written ONNX preprocessing (letterbox resize, NMS) that doesn't depend on Ultralytics at inference time
- **Spatial PPE association** — detected PPE items are matched to the nearest person using both IoU and containment ratio, handling the common case where a small helmet box sits entirely within a larger person bounding box
- **Timed violation tracking** — missing PPE must persist for 2+ consecutive seconds before triggering an alert, using spatial bucketing so small person movements between frames don't reset the timer
- **Real-time Streamlit + WebRTC frontend** — live webcam feed with adjustable confidence threshold and backend selection
- **Automated violation logging** — violations are written to CSV with UTC timestamps, deduplicated per incident

## Architecture

```
Webcam → Detector (YOLOv8 / ONNX) → Tracker (person ↔ PPE association) → Alert System → Annotated Frame
                                                                              ↓
                                                                         violations.csv
```

```
ppe-safety-detector/
├── src/
│   ├── config.py       # All tunable parameters and class mappings
│   ├── detector.py     # YOLOv8 + ONNX Runtime inference backends
│   ├── tracker.py      # Per-frame person ↔ PPE spatial association
│   ├── alert.py        # Violation timing, drawing, CSV logging
│   └── utils.py        # FPS counter
├── app.py              # Streamlit + WebRTC frontend
├── export_onnx.py      # One-command .pt → .onnx export
├── benchmark.py        # PyTorch vs ONNX latency comparison
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the base model and export to ONNX
mkdir -p models
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').save('models/yolov8n.pt')"
python export_onnx.py

# 4. Launch the app
streamlit run app.py
```

## Current Model & How to Plug In Your Own

Out of the box, this project uses **YOLOv8n pretrained on COCO** which reliably detects persons but does not detect PPE-specific classes (helmets, vests, gloves). The pipeline is fully functional — it just needs a model that outputs PPE classes.

### Plugging in a custom PPE model

1. Train YOLOv8 on a PPE dataset. Recommended public datasets:
   - [Safety-Helmet-Wearing-Dataset](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset)
   - [PPE Detection on Roboflow](https://universe.roboflow.com/search?q=ppe%20detection)
   - Training takes ~30 min on a free Google Colab T4 GPU
2. Place the resulting `best.pt` in `models/`
3. Update **three lines** in `src/config.py`:
   ```python
   PERSON_CLASS_ID = 3          # adjust to match your dataset's class index
   PPE_CLASSES = {0: "helmet", 1: "vest", 2: "gloves"}
   REQUIRED_PPE = {"helmet", "vest"}
   ```
4. Re-export to ONNX:
   ```bash
   python export_onnx.py --weights models/best.pt
   ```

No other code changes required — the detection, tracking, alerting, and UI all work with any class mapping.

## Key Design Decisions

| Decision | Why |
|----------|-----|
| Hand-written ONNX pre/postprocessing | Eliminates Ultralytics dependency at inference time; shows understanding of the model's input/output contract |
| IoU + containment ratio for association | Pure IoU fails when a small PPE box is fully inside a large person box — containment ratio catches this |
| Spatial bucketing for violation timers | A person shifting a few pixels between frames shouldn't reset the 2-second violation timer |
| Single-frame association, not full MOT | Keeps complexity proportional to the problem — Kalman filters and re-ID are overkill for stationary camera PPE monitoring |
| CSV logging deduplicated per incident | One log entry per violation, not one per frame — prevents log bloat |

## Benchmark Results

Run benchmarks with:

```bash
python benchmark.py --iterations 300
```

| Backend  | Mean (ms) | Median (ms) | P95 (ms) | FPS   |
|----------|-----------|-------------|----------|-------|
| PyTorch  | 113.3     | 113.8       | 124.0    | 8.8   |
| ONNX     | 47.2      | 48.7        | 55.1     | 21.2  |

> Tested on Intel Core i7-10750H (CPU). ONNX Runtime provides a **2.4x speedup** over PyTorch.

## Configuration

All tuneable parameters live in [`src/config.py`](src/config.py):

| Parameter             | Default      | Description                                    |
|-----------------------|--------------|------------------------------------------------|
| `CONFIDENCE_THRESHOLD`| 0.45         | Minimum detection confidence                   |
| `IOU_THRESHOLD`       | 0.50         | NMS IoU threshold                              |
| `PPE_ASSOCIATION_IOU` | 0.30         | Minimum overlap to associate PPE with a person  |
| `VIOLATION_HOLD_SEC`  | 2.0          | Seconds of missing PPE before alert triggers    |
| `REQUIRED_PPE`        | helmet, vest | PPE items that must be present                  |

## Tech Stack

- **YOLOv8** (Ultralytics) — object detection
- **ONNX Runtime** — optimized inference
- **OpenCV** — image preprocessing and annotation
- **Streamlit + WebRTC** — real-time browser frontend
- **Python 3.10+**

## License

MIT
