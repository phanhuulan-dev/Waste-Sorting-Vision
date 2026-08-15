from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import streamlit as st

from ..config import get_demo_video_paths
from ..io.video_loader import load_video_bytes, save_uploaded_video
from ..pipeline import VideoInferenceResult, run_video_inference

_VIDEO_RESULT_STATE_KEY = "video_inference_result"
_VIDEO_SOURCE_TOKEN_STATE_KEY = "video_inference_source_token"


def _counts_to_rows(class_counts: Mapping[str, int]) -> list[dict[str, int | str]]:
    return [
        {"Class": class_name, "Count": count}
        for class_name, count in class_counts.items()
    ]


def _render_class_counts(class_counts: Mapping[str, int]) -> None:
    st.markdown("**Detected Objects Across Sampled Frames**")
    if not class_counts:
        st.info("No objects were detected in the sampled video frames.")
        return

    st.dataframe(
        _counts_to_rows(class_counts),
        width="stretch",
        hide_index=True,
    )


def _resolve_video_source(source_mode: str, uploaded_video: Any) -> tuple[Path | None, str | None]:
    if source_mode == "Demo video":
        demo_videos = get_demo_video_paths()
        if not demo_videos:
            return None, None

        selected_label = st.selectbox(
            "Choose a demo video",
            options=list(demo_videos.keys()),
            key="demo_video_select",
        )
        selected_path = demo_videos[selected_label]
        if not selected_path.exists():
            st.warning(f"Configured demo video is missing: {selected_path}")
            return None, None

        st.video(load_video_bytes(selected_path))
        return selected_path, None

    if uploaded_video is None:
        return None, None

    st.video(uploaded_video.getvalue())
    return None, "uploaded"


def _build_video_source_token(
    source_mode: str,
    selected_path: Path | None,
    uploaded_video: Any,
) -> str:
    if source_mode == "Demo video":
        return f"demo:{selected_path}" if selected_path is not None else "demo:none"
    if uploaded_video is not None:
        return (
            f"upload:{getattr(uploaded_video, 'name', 'unknown')}:"
            f"{getattr(uploaded_video, 'size', 'unknown')}"
        )
    return "upload:none"


def render_video_panel(
    detector: Any,
    class_names: Mapping[int, str],
    confidence: float,
    frame_stride: int,
    max_frames: int | None = None,
) -> None:
    st.subheader("Video Inference")
    source_mode = st.radio(
        "Video source",
        options=("Demo video", "Upload video"),
        horizontal=True,
        key="video_source_mode",
    )
    uploaded_video = None
    if source_mode == "Upload video":
        uploaded_video = st.file_uploader(
            "Upload a video",
            type=("mp4", "mov", "avi", "mkv"),
            key="video_uploader",
        )

    selected_path, selected_kind = _resolve_video_source(source_mode, uploaded_video)
    effective_frame_stride = st.slider(
        "Frame stride",
        min_value=1,
        max_value=24,
        value=max(1, int(frame_stride)),
        step=1,
        help="Lower values update the annotated preview more smoothly but require more computation.",
        key="video_frame_stride",
    )
    source_token = _build_video_source_token(source_mode, selected_path, uploaded_video)
    if st.session_state.get(_VIDEO_SOURCE_TOKEN_STATE_KEY) != source_token:
        st.session_state[_VIDEO_SOURCE_TOKEN_STATE_KEY] = source_token
        st.session_state.pop(_VIDEO_RESULT_STATE_KEY, None)

    run_inference = st.button(
        "Run video inference",
        type="primary",
        width="stretch",
        disabled=(selected_path is None and uploaded_video is None),
        key="run_video_inference",
    )

    temporary_video_path: Path | None = None

    if run_inference:
        try:
            video_path = selected_path
            if selected_kind == "uploaded" and uploaded_video is not None:
                temporary_video_path = save_uploaded_video(uploaded_video)
                video_path = temporary_video_path

            if video_path is None:
                st.warning("Choose a video source before running inference.")
            else:
                progress_placeholder = st.empty()
                frame_placeholder = st.empty()

                def _render_live_frame(
                    annotated_frame,
                    processed_frame_count: int,
                    sampled_frame_count: int,
                ) -> None:
                    progress_placeholder.caption(
                        "Running video inference: "
                        f"processed {processed_frame_count} frames, "
                        f"sampled {sampled_frame_count} frames."
                    )
                    frame_placeholder.image(
                        annotated_frame,
                        caption="Live annotated frame",
                        width="stretch",
                    )

                try:
                    with st.spinner("Running video inference..."):
                        st.session_state[_VIDEO_RESULT_STATE_KEY] = run_video_inference(
                            detector=detector,
                            video_path=video_path,
                            confidence=confidence,
                            class_names=class_names,
                            frame_stride=effective_frame_stride,
                            max_frames=max_frames,
                            on_annotated_frame=_render_live_frame,
                        )
                except Exception as exc:
                    st.session_state.pop(_VIDEO_RESULT_STATE_KEY, None)
                    st.error("Video inference failed.")
                    st.exception(exc)
        finally:
            if temporary_video_path and temporary_video_path.exists():
                temporary_video_path.unlink()

    inference_result = st.session_state.get(_VIDEO_RESULT_STATE_KEY)

    if inference_result is not None:
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Processed frames", inference_result.processed_frames)
        metric_col2.metric("Sampled frames", inference_result.sampled_frames)
        if inference_result.last_annotated_frame is not None:
            st.image(
                inference_result.last_annotated_frame,
                caption="Last annotated sampled frame",
                width="stretch",
            )
        _render_class_counts(inference_result.class_counts)
