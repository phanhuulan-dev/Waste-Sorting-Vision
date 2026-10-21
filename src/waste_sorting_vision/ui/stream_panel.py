from __future__ import annotations

from collections import Counter
from datetime import timedelta
import os
import sys
from typing import Any, Mapping

import cv2
import numpy as np
import streamlit as st

from ..io.camera import format_camera_label, probe_camera_indices
from ..io.image_loader import load_uploaded_image
from ..live import LatestFrameReader, LiveProcessor
from ..pipeline import predict_frame, resize_frame_for_inference

try:  # optional: real-time video from the viewer's browser (works on phones and Streamlit Cloud)
    import av
    from streamlit_webrtc import WebRtcMode, webrtc_streamer
except ImportError:  # pragma: no cover - depends on the environment
    webrtc_streamer = None

_MODE_KEY = "live_stream_source_mode"
_PROCESSOR_KEY = "live_stream_processor"
_PROCESSOR_CONFIG_KEY = "live_stream_processor_config"
_READER_KEY = "live_stream_reader"
_ACTIVE_KEY = "live_stream_active"
_CAMERAS_KEY = "live_stream_available_cameras"
_CAMERA_ERROR_KEY = "live_stream_camera_error"
_SNAPSHOT_ID_KEY = "live_stream_browser_photo_id"
_SNAPSHOT_RESULT_KEY = "live_stream_snapshot_result"
_SNAPSHOT_COUNTS_KEY = "live_stream_snapshot_counts"

_MODE_LABELS = {
    "webrtc": "Browser live video (recommended)",
    "snapshot": "Browser snapshot",
    "opencv": "Local camera (this computer)",
}


# --------------------------------------------------------------------------- helpers


def _rows_to_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"Class": row["class_name"], "Confidence": row["confidence"], "Box (x1, y1, x2, y2)": row["bbox_xyxy"]}
        for row in rows
    ]


def _render_counts(title: str, counts: Mapping[str, int], empty_text: str) -> None:
    st.markdown(f"**{title}**")
    if not counts:
        st.caption(empty_text)
        return
    st.dataframe(
        [{"Class": name, "Count": count} for name, count in counts.items()],
        width="stretch",
        hide_index=True,
    )


def _render_live_stats(processor: LiveProcessor, show_detection_details: bool) -> None:
    stats = processor.stats()
    st.caption(
        f"{stats['fps']:.1f} FPS · inference {stats['inference_ms']:.0f} ms on {stats['device']} · "
        f"{stats['current_objects']} object(s) in view · frame {stats['frames']}"
    )
    if processor.track:
        _render_counts(
            "Unique objects seen (tracked)",
            stats["unique_counts"],
            "No object has been tracked long enough to be counted yet.",
        )
    if show_detection_details and stats["current_rows"]:
        with st.expander("Detections in the current frame", expanded=False):
            st.dataframe(_rows_to_table(stats["current_rows"]), width="stretch", hide_index=True)


def _get_processor(
    detector: Any,
    class_names: Mapping[int, str],
    confidence: float,
    imgsz: int,
    max_width: int,
    inference_stride: int,
    track: bool,
) -> LiveProcessor:
    """One processor per session; rebuilt (counts reset) when a setting changes."""
    config = (id(detector), tuple(sorted(class_names.items())), confidence, imgsz, max_width, inference_stride, track)
    processor = st.session_state.get(_PROCESSOR_KEY)
    if processor is None or st.session_state.get(_PROCESSOR_CONFIG_KEY) != config:
        processor = LiveProcessor(
            detector, class_names, confidence, imgsz, max_width, inference_stride=inference_stride, track=track
        )
        st.session_state[_PROCESSOR_KEY] = processor
        st.session_state[_PROCESSOR_CONFIG_KEY] = config
    return processor


def _rtc_configuration() -> dict[str, Any]:
    """Public STUN, plus a TURN server when configured.

    Viewers behind strict NAT (common on mobile networks and when the app runs
    on Streamlit Community Cloud) need TURN. Configure it with the environment
    variables WSV_TURN_URLS (comma separated), WSV_TURN_USERNAME and
    WSV_TURN_CREDENTIAL, or the same keys under [turn] in .streamlit/secrets.toml.
    """
    servers: list[dict[str, Any]] = [{"urls": ["stun:stun.l.google.com:19302"]}]
    turn: Mapping[str, Any] = {}
    try:
        turn = st.secrets.get("turn", {})
    except Exception:  # no secrets file
        turn = {}
    urls = os.environ.get("WSV_TURN_URLS") or turn.get("urls")
    if urls:
        servers.append({
            "urls": urls.split(",") if isinstance(urls, str) else list(urls),
            "username": os.environ.get("WSV_TURN_USERNAME") or turn.get("username", ""),
            "credential": os.environ.get("WSV_TURN_CREDENTIAL") or turn.get("credential", ""),
        })
    return {"iceServers": servers}


# --------------------------------------------------------------------------- local OpenCV camera


def _open_capture(camera_index: int, buffer_size: int) -> cv2.VideoCapture | None:
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    capture = cv2.VideoCapture(camera_index, backend)
    if not capture.isOpened():
        capture.release()
        return None
    capture.set(cv2.CAP_PROP_BUFFERSIZE, max(1, int(buffer_size)))
    return capture


def _stop_reader() -> None:
    reader = st.session_state.pop(_READER_KEY, None)
    if reader is not None:
        reader.release()
    st.session_state[_ACTIVE_KEY] = False


def _available_cameras(rescan: bool = False) -> list[int]:
    """Probe cameras once per session (probing opens every device, which is slow)."""
    if rescan or _CAMERAS_KEY not in st.session_state:
        st.session_state[_CAMERAS_KEY] = probe_camera_indices()
    return list(st.session_state[_CAMERAS_KEY])


def _render_opencv_stream(
    processor: LiveProcessor,
    camera_index: int,
    target_fps: float,
    camera_buffer_size: int,
    show_detection_details: bool,
) -> None:
    st.caption(
        "Reads a camera attached to the machine running the app, so it only works when the app "
        "runs locally. On macOS, allow camera access for your terminal or editor in "
        "System Settings → Privacy & Security → Camera."
    )
    is_streaming = bool(st.session_state.get(_ACTIVE_KEY))

    scan_col, camera_col = st.columns([1, 2])
    with scan_col:
        rescan = st.button("Scan cameras", disabled=is_streaming, width="stretch", key="live_stream_scan")
    cameras = _available_cameras(rescan=rescan)
    if rescan:
        st.session_state.pop(_CAMERA_ERROR_KEY, None)
    with camera_col:
        if cameras:
            default = cameras.index(camera_index) if camera_index in cameras else 0
            selected = st.selectbox(
                "Camera", options=cameras, index=default, format_func=format_camera_label,
                disabled=is_streaming, key="live_stream_camera_index",
            )
        else:
            selected = None
            st.warning("No camera detected. Check camera permissions, then press **Scan cameras**.")

    display_fps = st.slider(
        "Display FPS", min_value=5, max_value=30, value=int(min(30, max(5, round(target_fps)))),
        disabled=is_streaming, key="live_stream_display_fps",
    )

    start_col, stop_col, reset_col = st.columns(3)
    with start_col:
        if st.button("Start", type="primary", disabled=is_streaming or selected is None,
                     width="stretch", key="live_stream_start"):
            capture = _open_capture(int(selected), camera_buffer_size)
            if capture is None:
                st.session_state[_CAMERA_ERROR_KEY] = f"Cannot open {format_camera_label(int(selected))}."
            else:
                st.session_state.pop(_CAMERA_ERROR_KEY, None)
                st.session_state[_READER_KEY] = LatestFrameReader(capture)
                st.session_state[_ACTIVE_KEY] = True
                processor.reset_counts()
            st.rerun()
    with stop_col:
        if st.button("Stop", disabled=not is_streaming, width="stretch", key="live_stream_stop"):
            _stop_reader()
            st.rerun()
    with reset_col:
        if st.button("Reset counts", width="stretch", key="live_stream_reset"):
            processor.reset_counts()

    if st.session_state.get(_CAMERA_ERROR_KEY):
        st.error(st.session_state[_CAMERA_ERROR_KEY])

    frame_placeholder = st.empty()

    @st.fragment(run_every=timedelta(seconds=1.0 / display_fps) if is_streaming else None)
    def _loop() -> None:
        reader: LatestFrameReader | None = st.session_state.get(_READER_KEY)
        if reader is None:
            frame_placeholder.info("Press **Start** to begin live detection from the selected camera.")
            _render_live_stats(processor, show_detection_details)
            return
        if not reader.healthy:
            _stop_reader()
            st.session_state[_CAMERA_ERROR_KEY] = "The camera stopped delivering frames."
            st.rerun()
        _sequence, frame = reader.latest()
        if frame is None:
            frame_placeholder.info("Waiting for the first camera frame…")
            return
        annotated_bgr = processor.process(frame)
        frame_placeholder.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), width="stretch")
        _render_live_stats(processor, show_detection_details)

    _loop()


# --------------------------------------------------------------------------- browser video (WebRTC)


def _render_webrtc_stream(processor: LiveProcessor, show_detection_details: bool) -> None:
    st.caption(
        "Streams video from this browser's camera to the app and back with detections. "
        "Works on phones and on Streamlit Cloud; allow camera access when the browser asks."
    )
    facing = st.radio(
        "Camera", options=("environment", "user"), horizontal=True, key="live_stream_facing",
        format_func=lambda value: "Back (phones)" if value == "environment" else "Front / webcam",
    )

    def on_frame(frame: "av.VideoFrame") -> "av.VideoFrame":
        # runs on streamlit-webrtc's worker thread: no Streamlit calls in here
        annotated = processor.process(frame.to_ndarray(format="bgr24"))
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    context = webrtc_streamer(
        key=f"live-webrtc-{facing}",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=_rtc_configuration(),
        media_stream_constraints={
            "video": {"facingMode": {"ideal": facing}, "width": {"ideal": 1280}},
            "audio": False,
        },
        video_frame_callback=on_frame,
        async_processing=True,  # drop frames instead of queueing them when inference is slow
    )

    if st.button("Reset counts", key="live_stream_webrtc_reset"):
        processor.reset_counts()

    playing = bool(context.state.playing)

    @st.fragment(run_every=timedelta(seconds=1) if playing else None)
    def _stats() -> None:
        _render_live_stats(processor, show_detection_details)

    _stats()


# --------------------------------------------------------------------------- browser snapshot


def _render_snapshot(
    detector: Any,
    class_names: Mapping[int, str],
    confidence: float,
    imgsz: int,
    max_width: int,
    show_detection_details: bool,
) -> None:
    st.caption("Takes single photos with the browser camera. Use it when live video cannot connect.")
    photo = st.camera_input("Browser webcam", key="live_stream_browser_camera")
    if photo is None:
        return

    token = f"{photo.name}:{photo.size}"
    if token != st.session_state.get(_SNAPSHOT_ID_KEY):
        frame_bgr = cv2.cvtColor(np.asarray(load_uploaded_image(photo).convert("RGB")), cv2.COLOR_RGB2BGR)
        result = predict_frame(
            detector, resize_frame_for_inference(frame_bgr, max_width), confidence, class_names,
            imgsz=imgsz, as_pil=False,
        )
        st.session_state[_SNAPSHOT_ID_KEY] = token
        st.session_state[_SNAPSHOT_RESULT_KEY] = result
        totals = Counter(st.session_state.get(_SNAPSHOT_COUNTS_KEY, {}))
        totals.update(result.class_counts)
        st.session_state[_SNAPSHOT_COUNTS_KEY] = dict(totals.most_common())

    result = st.session_state[_SNAPSHOT_RESULT_KEY]
    st.image(result.annotated_image, caption=f"{len(result.detection_rows)} object(s) detected", width="stretch")
    _render_counts("Detections across snapshots", st.session_state.get(_SNAPSHOT_COUNTS_KEY, {}),
                   "No objects detected yet.")
    if show_detection_details and result.detection_rows:
        with st.expander("Detections in this snapshot", expanded=False):
            st.dataframe(_rows_to_table(result.detection_rows), width="stretch", hide_index=True)


# --------------------------------------------------------------------------- entry point


def render_stream_panel(
    detector: Any,
    class_names: Mapping[int, str],
    confidence: float,
    camera_index: int = 0,
    target_fps: float = 15.0,
    inference_stride: int = 1,
    inference_imgsz: int = 416,
    max_frame_width: int = 640,
    camera_buffer_size: int = 1,
    stats_update_stride: int = 5,  # kept for config compatibility; stats now refresh once per second
    show_detection_details: bool = True,
    default_source_mode: str = "webrtc",
    enable_tracking: bool = True,
) -> None:
    st.subheader("Live Stream")

    modes = [mode for mode in _MODE_LABELS if mode != "webrtc" or webrtc_streamer is not None]
    if default_source_mode == "browser":  # older configs
        default_source_mode = "webrtc" if "webrtc" in modes else "snapshot"
    source_mode = st.radio(
        "Camera source", options=modes, format_func=_MODE_LABELS.get, horizontal=True,
        index=modes.index(default_source_mode) if default_source_mode in modes else 0,
        disabled=bool(st.session_state.get(_ACTIVE_KEY)), key=_MODE_KEY,
    )
    if webrtc_streamer is None:
        st.caption("Install `streamlit-webrtc` to enable live video in the browser.")
    if source_mode != "opencv" and st.session_state.get(_READER_KEY) is not None:
        _stop_reader()

    size_col, width_col, track_col = st.columns(3)
    with size_col:
        sizes = (320, 416, 640)
        imgsz = st.selectbox(
            "Inference size", options=sizes, index=sizes.index(inference_imgsz) if inference_imgsz in sizes else 1,
            help="Smaller is faster; larger finds smaller objects.", key="live_stream_inference_imgsz",
        )
    with width_col:
        widths = (480, 640, 960)
        max_width = st.selectbox(
            "Max frame width", options=widths, index=widths.index(max_frame_width) if max_frame_width in widths else 1,
            key="live_stream_max_frame_width",
        )
    with track_col:
        track = st.toggle(
            "Count unique objects", value=enable_tracking, disabled=source_mode == "snapshot",
            help="Track objects across frames so each one is counted once.", key="live_stream_track",
        )

    if source_mode == "snapshot":
        _render_snapshot(detector, class_names, confidence, int(imgsz), int(max_width), show_detection_details)
        return

    processor = _get_processor(
        detector, class_names, confidence, int(imgsz), int(max_width), int(inference_stride), bool(track)
    )
    if source_mode == "webrtc":
        _render_webrtc_stream(processor, show_detection_details)
    else:
        _render_opencv_stream(processor, camera_index, target_fps, camera_buffer_size, show_detection_details)
