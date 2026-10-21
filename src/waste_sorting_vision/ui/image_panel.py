from __future__ import annotations

from typing import Any, Mapping

import streamlit as st

from ..config import get_default_detected_image_path, get_default_image_path
from ..io.image_loader import image_to_png_bytes, load_image, load_uploaded_image
from ..pipeline import ImageInferenceResult, predict_image

_IMAGE_RESULT_STATE_KEY = "image_inference_result"
_IMAGE_SOURCE_TOKEN_STATE_KEY = "image_inference_source_token"


def _counts_to_rows(class_counts: Mapping[str, int]) -> list[dict[str, int | str]]:
    return [
        {"Class": class_name, "Count": count}
        for class_name, count in class_counts.items()
    ]


def _render_class_counts(class_counts: Mapping[str, int]) -> None:
    st.markdown("**Detected Objects**")
    if not class_counts:
        st.info("No objects were detected in the current image.")
        return

    st.dataframe(
        _counts_to_rows(class_counts),
        width="stretch",
        hide_index=True,
    )


def _render_detection_rows(result: ImageInferenceResult) -> None:
    with st.expander("Detection details", expanded=False):
        if not result.detection_rows:
            st.write("No bounding boxes were returned.")
            return
        st.dataframe(
            [
                {
                    "Class ID": row["class_id"],
                    "Class": row["class_name"],
                    "Confidence": row["confidence"],
                    "Bounding Box (x1, y1, x2, y2)": row["bbox_xyxy"],
                }
                for row in result.detection_rows
            ],
            width="stretch",
            hide_index=True,
        )


def _load_preview_image(uploaded_image: Any) -> tuple[Any | None, str]:
    if uploaded_image is not None:
        return load_uploaded_image(uploaded_image), "Uploaded image"

    default_image_path = get_default_image_path()
    if default_image_path and default_image_path.exists():
        return load_image(default_image_path), "Demo image"

    return None, "No image available"


def _load_reference_detected_image() -> Any | None:
    detected_image_path = get_default_detected_image_path()
    if detected_image_path and detected_image_path.exists():
        return load_image(detected_image_path)
    return None


def _build_image_source_token(uploaded_image: Any) -> str:
    if uploaded_image is not None:
        return (
            f"upload:{getattr(uploaded_image, 'name', 'unknown')}:"
            f"{getattr(uploaded_image, 'size', 'unknown')}"
        )

    default_image_path = get_default_image_path()
    if default_image_path is not None:
        return f"default:{default_image_path}"
    return "none"


def render_image_panel(
    detector: Any,
    class_names: Mapping[int, str],
    confidence: float,
    enable_download: bool = True,
) -> None:
    st.subheader("Image Inference")
    uploaded_image = st.file_uploader(
        "Upload an image",
        type=("jpg", "jpeg", "png", "bmp", "webp"),
        key="image_uploader",
    )

    preview_image, preview_label = _load_preview_image(uploaded_image)
    source_token = _build_image_source_token(uploaded_image)
    if st.session_state.get(_IMAGE_SOURCE_TOKEN_STATE_KEY) != source_token:
        st.session_state[_IMAGE_SOURCE_TOKEN_STATE_KEY] = source_token
        st.session_state.pop(_IMAGE_RESULT_STATE_KEY, None)

    run_inference = st.button(
        "Run image inference",
        type="primary",
        width="stretch",
        disabled=preview_image is None,
        key="run_image_inference",
    )

    if run_inference and preview_image is not None:
        try:
            with st.spinner("Running image inference..."):
                st.session_state[_IMAGE_RESULT_STATE_KEY] = predict_image(
                    detector=detector,
                    image=preview_image,
                    confidence=confidence,
                    class_names=class_names,
                )
        except Exception as exc:
            st.session_state.pop(_IMAGE_RESULT_STATE_KEY, None)
            st.error("Image inference failed.")
            st.exception(exc)

    inference_result = st.session_state.get(_IMAGE_RESULT_STATE_KEY)

    col1, col2 = st.columns(2)
    with col1:
        if preview_image is None:
            st.info("Upload an image or keep a demo image under `assets/demo_images/`.")
        else:
            st.image(preview_image, caption=preview_label, width="stretch")

    with col2:
        if inference_result is not None:
            st.image(
                inference_result.annotated_image,
                caption="Detected image",
                width="stretch",
            )
            if enable_download:
                st.download_button(
                    "Download annotated image",
                    data=image_to_png_bytes(inference_result.annotated_image),
                    file_name="waste-sorting-vision_prediction.png",
                    mime="image/png",
                    width="stretch",
                )
        else:
            reference_detected_image = _load_reference_detected_image()
            if reference_detected_image is not None:
                st.image(
                    reference_detected_image,
                    caption="Demo detected example",
                    width="stretch",
                )
            else:
                st.info("Run inference to view the annotated result.")

    if inference_result is not None:
        _render_class_counts(inference_result.class_counts)
        _render_detection_rows(inference_result)
