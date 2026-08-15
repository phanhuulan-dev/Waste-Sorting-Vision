from __future__ import annotations

from collections import Counter
from datetime import timedelta
import sys
import time
from typing import Any, Mapping

import cv2
import numpy as np
import streamlit as st

from ..io.camera import format_camera_label, probe_camera_indices
from ..io.image_loader import load_uploaded_image
from ..pipeline import predict_frame, resize_frame_for_inference

_STREAM_ACTIVE_KEY = "live_stream_active"
_STREAM_CAPTURE_KEY = "live_stream_capture"
_STREAM_COUNTS_KEY = "live_stream_class_counts"
_STREAM_FRAME_INDEX_KEY = "live_stream_frame_index"
_STREAM_INFER_INDEX_KEY = "live_stream_infer_index"
_STREAM_LAST_RESULT_KEY = "live_stream_last_result"
_STREAM_LAST_FRAME_KEY = "live_stream_last_frame"
_STREAM_STATS_TICK_KEY = "live_stream_stats_tick"
_STREAM_AVAILABLE_CAMERAS_KEY = "live_stream_available_cameras"
_STREAM_CAMERA_ERROR_KEY = "live_stream_camera_error"
_STREAM_BROWSER_PHOTO_ID_KEY = "live_stream_browser_photo_id"
_STREAM_SOURCE_MODE_KEY = "live_stream_source_mode"


def _counts_to_rows(class_counts: Mapping[str, int]) -> list[dict[str, int | str]]:
    return [
        {"Class": class_name, "Count": count}
        for class_name, count in class_counts.items()
    ]


def _merge_class_counts(
    existing: Mapping[str, int],
    new: Mapping[str, int],
) -> dict[str, int]:
    merged = Counter(existing)
    merged.update(new)
    return dict(sorted(merged.items(), key=lambda item: (-item[1], item[0])))


def _release_capture() -> None:
    capture = st.session_state.pop(_STREAM_CAPTURE_KEY, None)
    if capture is not None:
        capture.release()


def _configure_capture(capture: cv2.VideoCapture, buffer_size: int) -> None:
    capture.set(cv2.CAP_PROP_BUFFERSIZE, max(1, int(buffer_size)))


def _open_capture(camera_index: int, buffer_size: int) -> cv2.VideoCapture | None:
    if sys.platform == "darwin":
        capture = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    else:
        capture = cv2.VideoCapture(camera_index)

    if not capture.isOpened():
        capture.release()
        return None

    _configure_capture(capture, buffer_size)
    return capture


def _scan_cameras() -> list[int]:
    return probe_camera_indices()


def _get_available_cameras() -> list[int]:
    cached = st.session_state.get(_STREAM_AVAILABLE_CAMERAS_KEY)
    if isinstance(cached, list) and cached:
        return [int(index) for index in cached]
    return _scan_cameras()


def _get_capture(camera_index: int, buffer_size: int) -> cv2.VideoCapture | None:
    if st.session_state.get(_STREAM_CAMERA_ERROR_KEY):
        return None

    capture = st.session_state.get(_STREAM_CAPTURE_KEY)
    bound_index = st.session_state.get("live_stream_bound_camera_index")
    if (
        capture is not None
        and capture.isOpened()
        and bound_index == camera_index
    ):
        return capture

    _release_capture()
    capture = _open_capture(camera_index, buffer_size)
    if capture is None:
        available = _get_available_cameras()
        available_text = ", ".join(str(index) for index in available) or "none"
        st.session_state[_STREAM_CAMERA_ERROR_KEY] = (
            f"Cannot open camera index {camera_index}. "
            f"Detected device(s): {available_text}."
        )
        return None

    st.session_state[_STREAM_CAPTURE_KEY] = capture
    st.session_state["live_stream_bound_camera_index"] = camera_index
    return capture


def _read_latest_frame(capture: cv2.VideoCapture) -> tuple[bool, np.ndarray | None]:
    return capture.read()


def _reset_stream_state() -> None:
    _release_capture()
    st.session_state[_STREAM_ACTIVE_KEY] = False
    st.session_state[_STREAM_COUNTS_KEY] = {}
    st.session_state[_STREAM_FRAME_INDEX_KEY] = 0
    st.session_state[_STREAM_INFER_INDEX_KEY] = 0
    st.session_state.pop(_STREAM_LAST_RESULT_KEY, None)
    st.session_state.pop(_STREAM_LAST_FRAME_KEY, None)
    st.session_state[_STREAM_STATS_TICK_KEY] = 0
    st.session_state.pop(_STREAM_CAMERA_ERROR_KEY, None)
    st.session_state.pop("live_stream_bound_camera_index", None)


def _render_stats_table(
    placeholder: Any,
    class_counts: Mapping[str, int],
) -> None:
    with placeholder.container():
        st.markdown("**Session totals (cumulative detections)**")
        if not class_counts:
            st.info("No objects detected yet in this session.")
            return
        st.dataframe(
            _counts_to_rows(class_counts),
            width="stretch",
            hide_index=True,
        )


def _render_detection_expander(last_result: Any, show_detection_details: bool) -> None:
    if not show_detection_details:
        return

    with st.expander("Current frame detections", expanded=False):
        if last_result is None or not last_result.detection_rows:
            st.write("No detections in the latest frame yet.")
            return
        st.dataframe(
            [
                {
                    "Class ID": row["class_id"],
                    "Class": row["class_name"],
                    "Confidence": row["confidence"],
                    "Bounding Box (x1, y1, x2, y2)": row["bbox_xyxy"],
                }
                for row in last_result.detection_rows
            ],
            width="stretch",
            hide_index=True,
        )


def _run_inference_on_frame(
    detector: Any,
    frame_bgr: np.ndarray,
    confidence: float,
    class_names: Mapping[int, str],
    imgsz: int,
    max_width: int,
) -> Any:
    resized_frame = resize_frame_for_inference(frame_bgr, int(max_width))
    return predict_frame(
        detector=detector,
        frame_bgr=resized_frame,
        confidence=confidence,
        class_names=class_names,
        imgsz=int(imgsz),
        as_pil=False,
    )


def _render_browser_stream(
    detector: Any,
    class_names: Mapping[int, str],
    confidence: float,
    imgsz: int,
    max_width: int,
    show_detection_details: bool,
    frame_placeholder: Any,
    status_placeholder: Any,
    stats_placeholder: Any,
) -> None:
    st.info(
        "Browser camera uses your webcam through Chrome/Safari. "
        "Take snapshots for detection — more stable on Mac than OpenCV."
    )
    photo = st.camera_input("Browser webcam", key="live_stream_browser_camera")
    if photo is None:
        frame_placeholder.info("Allow camera access, then capture a frame to run detection.")
        return

    photo_token = f"{photo.name}:{photo.size}"
    if photo_token == st.session_state.get(_STREAM_BROWSER_PHOTO_ID_KEY):
        last_result = st.session_state.get(_STREAM_LAST_RESULT_KEY)
        if last_result is not None:
            frame_placeholder.image(
                last_result.annotated_image,
                caption="Last annotated capture",
                width="stretch",
            )
        return

    image = load_uploaded_image(photo)
    frame_bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)

    try:
        result = _run_inference_on_frame(
            detector=detector,
            frame_bgr=frame_bgr,
            confidence=confidence,
            class_names=class_names,
            imgsz=imgsz,
            max_width=max_width,
        )
    except Exception as exc:
        status_placeholder.error("Browser camera inference failed.")
        st.exception(exc)
        return

    st.session_state[_STREAM_BROWSER_PHOTO_ID_KEY] = photo_token
    st.session_state[_STREAM_LAST_RESULT_KEY] = result
    st.session_state[_STREAM_COUNTS_KEY] = _merge_class_counts(
        st.session_state.get(_STREAM_COUNTS_KEY, {}),
        result.class_counts,
    )
    status_placeholder.caption(
        f"Captured frame · {len(result.detection_rows)} object(s) detected"
    )
    frame_placeholder.image(
        result.annotated_image,
        caption="Annotated capture",
        width="stretch",
    )
    _render_stats_table(stats_placeholder, st.session_state.get(_STREAM_COUNTS_KEY, {}))
    _render_detection_expander(result, show_detection_details)


def render_stream_panel(
    detector: Any,
    class_names: Mapping[int, str],
    confidence: float,
    camera_index: int = 0,
    target_fps: float = 15.0,
    inference_stride: int = 2,
    inference_imgsz: int = 416,
    max_frame_width: int = 640,
    camera_buffer_size: int = 1,
    stats_update_stride: int = 5,
    show_detection_details: bool = True,
    default_source_mode: str = "browser",
) -> None:
    st.subheader("Live Stream")
    st.caption(
        "Use **Browser camera** on Mac if OpenCV reports invalid camera indices."
    )

    if _STREAM_ACTIVE_KEY not in st.session_state:
        _reset_stream_state()

    is_streaming = bool(st.session_state.get(_STREAM_ACTIVE_KEY))
    available_cameras = _get_available_cameras()

    source_mode = st.radio(
        "Camera source",
        options=("browser", "opencv"),
        format_func=lambda value: (
            "Browser camera (recommended)"
            if value == "browser"
            else "Local OpenCV stream"
        ),
        horizontal=True,
        disabled=is_streaming,
        index=0 if default_source_mode == "browser" else 1,
        key=_STREAM_SOURCE_MODE_KEY,
    )

    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    stats_placeholder = st.empty()

    tuning_col1, tuning_col2 = st.columns(2)
    with tuning_col1:
        imgsz = st.selectbox(
            "Inference size",
            options=(320, 416, 640),
            index=(320, 416, 640).index(inference_imgsz)
            if inference_imgsz in (320, 416, 640)
            else 1,
            disabled=is_streaming and source_mode == "opencv",
            key="live_stream_inference_imgsz",
        )
    with tuning_col2:
        max_width = st.selectbox(
            "Max frame width",
            options=(480, 640, 960),
            index=(480, 640, 960).index(max_frame_width)
            if max_frame_width in (480, 640, 960)
            else 1,
            disabled=is_streaming and source_mode == "opencv",
            key="live_stream_max_frame_width",
        )

    if source_mode == "browser":
        _render_browser_stream(
            detector=detector,
            class_names=class_names,
            confidence=confidence,
            imgsz=int(imgsz),
            max_width=int(max_width),
            show_detection_details=show_detection_details,
            frame_placeholder=frame_placeholder,
            status_placeholder=status_placeholder,
            stats_placeholder=stats_placeholder,
        )
        return

    scan_col, camera_col = st.columns([1, 2])
    with scan_col:
        if st.button(
            "Scan cameras",
            disabled=is_streaming,
            width="stretch",
            key="live_stream_scan_cameras",
        ):
            st.session_state[_STREAM_AVAILABLE_CAMERAS_KEY] = _scan_cameras()
            st.session_state.pop(_STREAM_CAMERA_ERROR_KEY, None)

    default_index = camera_index if camera_index in available_cameras else available_cameras[0]
    with camera_col:
        selected_camera = st.selectbox(
            "Available cameras",
            options=available_cameras,
            index=available_cameras.index(default_index),
            format_func=format_camera_label,
            disabled=is_streaming,
            help="Only detected indices are listed. Your Mac log showed devices 0-0 only.",
            key="live_stream_camera_index",
        )

    if not available_cameras:
        st.warning(
            "No OpenCV cameras were detected. Switch to **Browser camera** or check "
            "System Settings → Privacy → Camera."
        )

    camera_error = st.session_state.get(_STREAM_CAMERA_ERROR_KEY)
    if camera_error:
        st.error(str(camera_error))

    settings_col1, settings_col2 = st.columns(2)
    with settings_col1:
        display_fps = st.slider(
            "Display FPS",
            min_value=5,
            max_value=30,
            value=int(min(30, max(5, round(target_fps)))),
            disabled=is_streaming,
            key="live_stream_display_fps",
        )
    with settings_col2:
        effective_stride = st.slider(
            "Inference stride",
            min_value=1,
            max_value=6,
            value=max(1, int(inference_stride)),
            disabled=is_streaming,
            key="live_stream_inference_stride",
        )

    control_col1, control_col2, control_col3 = st.columns(3)
    with control_col1:
        if st.button(
            "Start stream",
            type="primary",
            disabled=is_streaming or not available_cameras,
            width="stretch",
            key="live_stream_start",
        ):
            _release_capture()
            st.session_state.pop(_STREAM_CAMERA_ERROR_KEY, None)
            st.session_state[_STREAM_ACTIVE_KEY] = True
            st.session_state[_STREAM_COUNTS_KEY] = {}
            st.session_state[_STREAM_FRAME_INDEX_KEY] = 0
            st.session_state[_STREAM_INFER_INDEX_KEY] = 0
            st.session_state[_STREAM_STATS_TICK_KEY] = 0
            st.session_state.pop(_STREAM_LAST_RESULT_KEY, None)
            st.session_state.pop(_STREAM_LAST_FRAME_KEY, None)
    with control_col2:
        if st.button(
            "Stop stream",
            disabled=not is_streaming,
            width="stretch",
            key="live_stream_stop",
        ):
            _reset_stream_state()
    with control_col3:
        if st.button(
            "Reset counts",
            disabled=not is_streaming,
            width="stretch",
            key="live_stream_reset_counts",
        ):
            st.session_state[_STREAM_COUNTS_KEY] = {}
            st.session_state[_STREAM_FRAME_INDEX_KEY] = 0
            st.session_state[_STREAM_INFER_INDEX_KEY] = 0

    interval_seconds = 1.0 / max(int(display_fps), 1)
    run_every = (
        timedelta(seconds=interval_seconds)
        if st.session_state.get(_STREAM_ACTIVE_KEY)
        else None
    )

    @st.fragment(run_every=run_every)
    def _live_stream_loop() -> None:
        if not st.session_state.get(_STREAM_ACTIVE_KEY):
            frame_placeholder.info(
                "Press **Start stream** to begin live detection from your webcam."
            )
            return

        if st.session_state.get(_STREAM_CAMERA_ERROR_KEY):
            frame_placeholder.error(str(st.session_state[_STREAM_CAMERA_ERROR_KEY]))
            return

        loop_started = time.perf_counter()
        capture = _get_capture(int(selected_camera), camera_buffer_size)
        if capture is None:
            status_placeholder.error(
                st.session_state.get(
                    _STREAM_CAMERA_ERROR_KEY,
                    "Unable to open the selected camera.",
                )
            )
            st.session_state[_STREAM_ACTIVE_KEY] = False
            _release_capture()
            return

        try:
            success, frame = _read_latest_frame(capture)
        except cv2.error as exc:
            status_placeholder.error(f"Camera read failed: {exc}")
            st.session_state[_STREAM_CAMERA_ERROR_KEY] = str(exc)
            st.session_state[_STREAM_ACTIVE_KEY] = False
            _release_capture()
            return

        if not success or frame is None:
            status_placeholder.warning("Failed to read a frame from the camera.")
            return

        frame_index = int(st.session_state.get(_STREAM_FRAME_INDEX_KEY, 0)) + 1
        st.session_state[_STREAM_FRAME_INDEX_KEY] = frame_index

        infer_index = int(st.session_state.get(_STREAM_INFER_INDEX_KEY, 0))
        should_infer = (
            infer_index == 0 or frame_index % max(int(effective_stride), 1) == 0
        )

        if should_infer:
            infer_index += 1
            st.session_state[_STREAM_INFER_INDEX_KEY] = infer_index
            try:
                result = _run_inference_on_frame(
                    detector=detector,
                    frame_bgr=frame,
                    confidence=confidence,
                    class_names=class_names,
                    imgsz=int(imgsz),
                    max_width=int(max_width),
                )
            except Exception as exc:
                status_placeholder.error("Live inference failed.")
                st.exception(exc)
                _reset_stream_state()
                return

            st.session_state[_STREAM_LAST_RESULT_KEY] = result
            st.session_state[_STREAM_LAST_FRAME_KEY] = result.annotated_image
            st.session_state[_STREAM_COUNTS_KEY] = _merge_class_counts(
                st.session_state.get(_STREAM_COUNTS_KEY, {}),
                result.class_counts,
            )
            display_frame = result.annotated_image
            detection_count = len(result.detection_rows)
            infer_note = "inferred"
        else:
            display_frame = st.session_state.get(_STREAM_LAST_FRAME_KEY)
            if display_frame is None:
                display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection_count = len(
                getattr(st.session_state.get(_STREAM_LAST_RESULT_KEY), "detection_rows", [])
            )
            infer_note = "preview"

        elapsed_ms = (time.perf_counter() - loop_started) * 1000
        status_placeholder.caption(
            "Streaming · "
            f"camera {selected_camera} · "
            f"frame {frame_index} · "
            f"{detection_count} object(s) · "
            f"{infer_note} · "
            f"loop {elapsed_ms:.0f} ms"
        )
        frame_placeholder.image(
            display_frame,
            caption="Live annotated feed",
            width="stretch",
        )

        stats_tick = int(st.session_state.get(_STREAM_STATS_TICK_KEY, 0)) + 1
        st.session_state[_STREAM_STATS_TICK_KEY] = stats_tick
        if should_infer and stats_tick % max(int(stats_update_stride), 1) == 0:
            _render_stats_table(
                stats_placeholder,
                st.session_state.get(_STREAM_COUNTS_KEY, {}),
            )

    _live_stream_loop()

    if not is_streaming:
        _render_stats_table(
            stats_placeholder,
            st.session_state.get(_STREAM_COUNTS_KEY, {}),
        )

    _render_detection_expander(
        st.session_state.get(_STREAM_LAST_RESULT_KEY),
        show_detection_details,
    )
