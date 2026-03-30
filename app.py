"""Streamlit + WebRTC frontend for real-time PPE detection.

Launch with::

    streamlit run app.py
"""

from __future__ import annotations

import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from src.alert import ViolationTracker, draw_alert_banner, draw_results
from src.config import ONNX_WEIGHTS, REQUIRED_PPE
from src.detector import Backend, Detector
from src.tracker import associate
from src.utils import FPSCounter

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="PPE Safety Detector", layout="wide")
st.title("PPE Safety Detector")
st.caption("Real-time personal protective equipment compliance monitoring")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    backend_choice = st.radio(
        "Inference backend",
        ["ONNX", "PyTorch"],
        index=0,
        help="ONNX is recommended for lower latency on CPU.",
    )

    conf_threshold = st.slider(
        "Confidence threshold",
        0.1, 1.0, 0.45, 0.05,
    )

    st.markdown("---")
    st.subheader("Required PPE")
    st.write(", ".join(sorted(REQUIRED_PPE)))

    st.markdown("---")
    st.markdown(
        "Violations are logged to `violations.csv` in the project root."
    )


# ---------------------------------------------------------------------------
# Shared state — detector, tracker, FPS counter
# ---------------------------------------------------------------------------
@st.cache_resource
def get_detector(backend_name: str, conf: float) -> Detector:
    backend = Backend.ONNX if backend_name == "ONNX" else Backend.PYTORCH

    if backend is Backend.ONNX and not ONNX_WEIGHTS.exists():
        st.error(
            f"ONNX weights not found at `{ONNX_WEIGHTS}`. "
            "Run `python export_onnx.py` first, or switch to PyTorch backend."
        )
        st.stop()

    return Detector(backend=backend, conf=conf)


detector = get_detector(backend_choice, conf_threshold)
tracker = ViolationTracker()
fps_counter = FPSCounter()

# ---------------------------------------------------------------------------
# WebRTC video callback
# ---------------------------------------------------------------------------

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    detections, _ = detector.detect(img)
    statuses = associate(detections)
    results = tracker.update(statuses)
    fps = fps_counter.tick()

    annotated = draw_results(img, results, fps)

    # add top banner if any active violation
    violations = [r for r in results if r[1]]
    if violations:
        count = len(violations)
        annotated = draw_alert_banner(
            annotated,
            f"  {count} PPE VIOLATION{'S' if count > 1 else ''} DETECTED",
        )

    return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ---------------------------------------------------------------------------
# Stream
# ---------------------------------------------------------------------------
ctx = webrtc_streamer(
    key="ppe-detector",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)

if not ctx.state.playing:
    st.info("Click **START** above to begin webcam detection.")
