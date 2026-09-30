# Waste Sorting Vision

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://waste-sorting-vision.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-111111)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-5C3EE8?logo=opencv&logoColor=white)

YOLOv8-based household waste detection with a modular Streamlit inference app for image and video analysis.

[Live demo](https://waste-sorting-vision.streamlit.app/) • Tested locally with Homebrew Python `3.14`

## Overview

Waste Sorting Vision is a computer vision project for recognising common household waste categories.
It combines model inference, a lightweight Streamlit interface, and a concise set of project documents covering the main experimental results.

This public release is designed to be straightforward to run and easy to review.
The focus is on the working application, the class set used by the app, and the project outcomes that are most relevant in a portfolio context.

## App Preview

Sample detection output from the current application:

![Sample detection output](assets/demo_images/default_detected.jpg)

## Highlights

- image and video inference through a Streamlit app
- configurable checkpoint selection for local evaluation
- bundled 15-class and 16-class checkpoints for comparison
- annotated image download for quick result review
- project figures, result summaries, and workflow notes

## Results Snapshot

- the recorded training history covers `8`, `11`, `15`, and `16`-class settings
- the highest `mAP@50` listed in the summary table is `0.957` from a `yolov8s`, `640`, `38,000` image, `8`-class experiment
- the default app checkpoint is `waste_v5.pt`, a `yolov8m` model with a `9`-class label set (Paper, Paper Cup, Vinyl, Plastic, Glass, Can, PET, Styrofoam, Battery) trained on TACO (official and community annotations) plus Roboflow Universe datasets, after a label-quality audit that removed mislabelled sources and images with missing labels
- on the `waste_v5` validation set (3,484 images) it reaches `mAP@50` `0.768` / `mAP@50-95` `0.644`, versus `0.699` for `waste_v3.pt` and `0.677` for `waste_v2.pt` (both `yolov8n`)
- on real-world TACO photos it reaches `0.381` on the official validation split (`waste_v2.pt`: `0.362`) and `0.268` on official plus community validation photos (`waste_v2.pt`: `0.210`); Glass and Styrofoam remain weak on such photos, so real-world results should be treated as experimental
- `waste_v5.pt` is about 4x slower than the `yolov8n` checkpoints on CPU; `waste_v3.pt` and `waste_v2.pt` stay selectable for faster video and webcam use
- the older `best5.pt` (`16` classes) and `best.pt` (`15` classes) checkpoints remain selectable for comparison

Representative project figure:

![P-R Curve](assets/figures/pr_curve_archived.png)

Additional tables and figures are summarised in [docs/modeling_report.md](docs/modeling_report.md) and [docs/experiment_history.md](docs/experiment_history.md).

## Project Structure

- a thin Streamlit entry point in `app/streamlit_app.py`
- modular application logic under `src/waste_sorting_vision/`
- configurable class names, demo assets, and checkpoint locations in `configs/`
- project figures under `assets/figures/`
- supporting project notes under `docs/`

## Repository Layout

```text
waste-sorting-vision/
├─ app/
├─ assets/
├─ configs/
├─ docs/
├─ models/
├─ requirements/
├─ src/
└─ tests/
```

## Running The App

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/app.txt
streamlit run app/streamlit_app.py
```

The commands above assume that `python` points to the interpreter you want to use for this project environment.
If you manage multiple Python installations locally, create the virtual environment with your preferred interpreter first and then activate `.venv`.

For the closest match to the tested local environment, install from `requirements/full-lock.txt` instead.

## Streamlit Deployment

For Streamlit Community Cloud deployment, use `app/streamlit_app.py` as the entry point.
This repository also includes a root `requirements.txt` so the deployment environment can resolve the application dependencies automatically.
It also includes a root `packages.txt` for the Linux system packages required by OpenCV in Community Cloud.

For deployment stability, choose Python `3.13` or `3.14` in the deployment settings when those options are available.

## Running Tests

```bash
source .venv/bin/activate
pip install -r requirements/dev.txt
pytest -p no:cacheprovider tests
```

## Checkpoint Configuration

The app exposes these model keys:

- `waste_v5` (default)
- `waste_v3`
- `waste_v2`
- `best`
- `best5`

Resolution order is:

1. matching environment variable
2. `models/` path configured in `configs/model_sources.yaml`

Environment variables:

- `WSV_MODEL_WASTE_V5`
- `WSV_MODEL_WASTE_V3`
- `WSV_MODEL_WASTE_V2`
- `WSV_MODEL_BEST`
- `WSV_MODEL_BEST5`

If you later decide to move the checkpoints outside the repository, point those environment variables to release assets, Git LFS files, or another artefact location.

## Key Docs

- [Project scope note](docs/reproducibility_note.md)
- [Environment reference](docs/environment_reference.md)
- [Class labels](docs/class_taxonomy_reconstruction.md)
- [Results report for the 9-class models (Vietnamese)](docs/bao_cao_ket_qua.md)
- [Modelling report](docs/modeling_report.md)
- [Experiment history](docs/experiment_history.md)
- [Preprocessing summary](docs/preprocessing_summary.md)
- [Checkpoint storage strategy](docs/checkpoint_storage_strategy.md)

## References

- AI-Hub Household Waste Dataset: https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71385
- TACO (Trash Annotations in Context): https://github.com/pedropro/TACO
- Roboflow Universe datasets used for training: listed with licences in [docs/bao_cao_ket_qua.md](docs/bao_cao_ket_qua.md)
- Ultralytics YOLO: https://github.com/ultralytics/ultralytics
- Streamlit: https://streamlit.io/

## Attribution

This project builds on [Waste Sorting Vision](https://github.com/J-Y00N/Waste-Sorting-Vision) by J. Yoon, released under the MIT License, which is kept unchanged in [LICENSE](LICENSE).

From the original project:

- the Streamlit application and its modular code structure
- the `best.pt` (15-class) and `best5.pt` (16-class) checkpoints trained on AI-Hub data, with their experiment history and documentation

Added in this fork:

- a 9-class label set suited to publicly available data, and the `waste_v2`, `waste_v3` and `waste_v5` YOLOv8 checkpoints (`waste_v5` is the default)
- a data pipeline that merges TACO (official and community annotations) with Roboflow Universe datasets, removes near-duplicate images and fixes EXIF-rotated images
- a model-assisted label audit that excluded mislabelled sources and images with missing labels
- a Colab training notebook, evaluation scripts, and a results report with error analysis and limitations
- fixes to the app's image inference (colour channel order and duplicate labels across classes)
