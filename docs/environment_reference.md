# Environment Reference

## Tested Local Environment

- Python `3.14.4`
- Streamlit `1.56.0`
- Ultralytics `8.4.37`
- Torch `2.11.0`
- Torchvision `0.26.0`
- OpenCV Headless `4.13.0.92`
- NumPy `2.4.4`
- Pillow `12.2.0`
- PyYAML `6.0.3`

## Installation Options

### Option A: Pinned Direct Dependencies

Use this when you want a concise setup that matches the verified direct dependency versions in this repository.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/app.txt
```

### Option B: Exact Tested Environment

Use this when you want the closest available match to the local environment used for verification in this repository.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/full-lock.txt
```

## Verification Performed

- `pytest -p no:cacheprovider tests`
- Streamlit app import and local execution checks

## Scope

This environment reference improves reproducibility for the current app and test setup.
It does not turn the repository into a full retraining package for the original project stage.
