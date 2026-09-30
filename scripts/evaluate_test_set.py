"""Evaluate 9-class checkpoints on an independent test set (Roboflow YOLOv8 export).

The export may use any split folders (train/valid/test); every image found is
treated as test data. Class names in the export's data.yaml are matched to the
app's 9 classes by name (case-insensitive), so the export's class order does
not matter. Test images that are near-duplicates of training images are
reported, because they would make the test set less independent.

Usage:
  python scripts/evaluate_test_set.py path/to/roboflow_export
  python scripts/evaluate_test_set.py path/to/roboflow_export --models waste_v4 waste_v3
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waste_sorting_vision.class_names import load_class_names  # noqa: E402
from waste_sorting_vision.config import get_model_catalog  # noqa: E402
from waste_sorting_vision.detector import resolve_model_path  # noqa: E402

TRAIN_IMAGES = ROOT / "datasets" / "waste_v1" / "images" / "train"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NEAR_DUPLICATE_BITS = 10


def nine_class_models() -> list[str]:
    reference = load_class_names("waste_v3")
    return [key for key in get_model_catalog() if load_class_names(key) == reference]


def dhash(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(image, (17, 16), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()


def build_test_dataset(export: Path, stage: Path, class_names: dict[int, str]) -> list[Path]:
    export_names = yaml.safe_load((export / "data.yaml").read_text())["names"]
    if isinstance(export_names, dict):
        export_names = [export_names[k] for k in sorted(export_names)]
    by_name = {name.lower(): class_id for class_id, name in class_names.items()}
    unknown = [name for name in export_names if name.lower() not in by_name]
    if unknown:
        raise SystemExit(f"Unknown class names in export: {unknown}. Expected: {list(class_names.values())}")
    remap = {index: by_name[name.lower()] for index, name in enumerate(export_names)}

    (stage / "images").mkdir(parents=True)
    (stage / "labels").mkdir()
    images = []
    for image in sorted(export.glob("*/images/*")):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = image.parent.parent / "labels" / (image.stem + ".txt")
        lines = []
        if label.is_file():
            for line in label.read_text().splitlines():
                parts = line.split()
                if len(parts) == 5:
                    lines.append(" ".join([str(remap[int(parts[0])]), *parts[1:]]))
        (stage / "images" / image.name).symlink_to(image.resolve())
        (stage / "labels" / (image.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
        images.append(image)

    data = {"path": str(stage), "train": "images", "val": "images", "names": class_names}
    (stage / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return images


def report_train_overlap(images: list[Path]) -> None:
    train = [p for p in TRAIN_IMAGES.iterdir()] if TRAIN_IMAGES.is_dir() else []
    if not train:
        print("(training images not found locally; skipped duplicate check)")
        return
    train_hashes = np.stack([dhash(p) for p in train])
    duplicates = [p.name for p in images if int((train_hashes != dhash(p)).sum(1).min()) <= NEAR_DUPLICATE_BITS]
    print(f"Test images that duplicate a training image: {len(duplicates)}/{len(images)}")
    for name in duplicates[:10]:
        print(f"  - {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export", type=Path, help="Roboflow YOLOv8 export folder containing data.yaml")
    parser.add_argument("--models", nargs="+", help="model keys from configs/model_sources.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="e.g. mps, cpu, 0")
    args = parser.parse_args()

    from ultralytics import YOLO

    models = args.models or nine_class_models()
    class_names = load_class_names(models[0])

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "test"
        images = build_test_dataset(args.export, stage, class_names)
        boxes = sum(1 for p in (stage / "labels").glob("*.txt") for line in p.read_text().splitlines() if line)
        print(f"Test set: {len(images)} images, {boxes} boxes")
        report_train_overlap(images)

        results = {}
        for key in models:
            metrics = YOLO(str(resolve_model_path(key))).val(
                data=str(stage / "data.yaml"), imgsz=args.imgsz, batch=8, device=args.device,
                plots=False, verbose=False, project=str(Path(tmp) / "runs"), name=key,
            )
            per_class = {int(c): float(metrics.box.ap50[i]) for i, c in enumerate(metrics.box.ap_class_index)}
            counts = {int(c): int(metrics.nt_per_class[int(c)]) for c in metrics.box.ap_class_index}
            results[key] = (metrics.box.map50, metrics.box.map, metrics.box.mp, metrics.box.mr, per_class, counts)

    print(f"\n{'model':<12}{'mAP50':>8}{'mAP50-95':>10}{'P':>7}{'R':>7}")
    for key, (m50, m95, precision, recall, _, _) in results.items():
        print(f"{key:<12}{m50:>8.3f}{m95:>10.3f}{precision:>7.3f}{recall:>7.3f}")

    counts = next(iter(results.values()))[5]
    print(f"\nmAP50 per class\n{'class':<12}{'boxes':>6}" + "".join(f"{key:>11}" for key in results))
    for class_id, name in class_names.items():
        if class_id not in counts:
            continue
        row = "".join(f"{results[key][4].get(class_id, 0.0):>11.3f}" for key in results)
        print(f"{name:<12}{counts[class_id]:>6}{row}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
