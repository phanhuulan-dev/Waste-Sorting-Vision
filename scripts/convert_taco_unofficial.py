"""Convert TACO's community ("unofficial") annotations into a YOLO source folder.

Output mirrors a Roboflow export so merge_yolo_datasets.py can merge it like
any other source; class names stay TACO's own and are remapped there:

  datasets/taco_unofficial/data.yaml
  datasets/taco_unofficial/{train,valid}/images/*.jpg   (symlinks)
  datasets/taco_unofficial/{train,valid}/labels/*.txt

Images must be downloaded first (scripts/download_taco.py
annotations_unofficial.json). Before converting, every image is checked
against its annotation size:
  - some photos carry an EXIF rotation flag although they were annotated in
    the raw (unrotated) frame; they are rewritten in the raw frame without the
    flag, otherwise Ultralytics would rotate them and misplace every box
  - images larger than MAX_SIDE are downscaled (labels are normalised)
  - images whose aspect ratio matches neither frame are skipped
The check is idempotent, so re-running it is safe.
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TACO_DATA_DIR = ROOT / "datasets" / "taco_repo" / "data"
OUTPUT_DIR = ROOT / "datasets" / "taco_unofficial"
VAL_FRACTION = 0.15
SEED = 42
MAX_SIDE = 1280
ROTATED = {5, 6, 7, 8}  # EXIF orientations that swap width and height


def normalise_image(path: Path, ann_width: int, ann_height: int) -> bool:
    """Make the file match its annotation frame; return False if it cannot."""
    with Image.open(path) as image:
        width, height = image.size
        orientation = (image.getexif() or {}).get(274, 1)
    ann_ratio = ann_width / ann_height
    raw_matches = abs(width / height - ann_ratio) < 0.01
    rotated_matches = abs(height / width - ann_ratio) < 0.01

    if orientation in ROTATED and raw_matches:
        flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION  # keep the annotated raw frame
    elif (orientation in ROTATED and rotated_matches) or (orientation not in ROTATED and raw_matches):
        flags = cv2.IMREAD_COLOR  # decoder applies EXIF, same as Ultralytics
    else:
        return False

    if orientation in (0, 1) and max(width, height) <= MAX_SIDE:
        return True
    pixels = cv2.imread(str(path), flags)
    if pixels is None:
        return False
    h, w = pixels.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        pixels = cv2.resize(pixels, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.imwrite(str(path), pixels, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> int:
    annotations = json.loads((TACO_DATA_DIR / "annotations_unofficial.json").read_text())
    excluded = set()
    for image in annotations["images"]:
        path = TACO_DATA_DIR / image["file_name"]
        if path.is_file() and path.stat().st_size > 0 and not normalise_image(path, image["width"], image["height"]):
            excluded.add(image["file_name"])
    if excluded:
        print(f"skipping {len(excluded)} images whose size does not match their annotation")

    categories = sorted(annotations["categories"], key=lambda c: c["id"])
    class_index = {c["id"]: index for index, c in enumerate(categories)}
    images = {image["id"]: image for image in annotations["images"]}

    labels: dict[int, list[str]] = {}
    for ann in annotations["annotations"]:
        image = images[ann["image_id"]]
        width, height = float(image["width"]), float(image["height"])
        x, y, w, h = ann["bbox"]
        cx = min(max((x + w / 2) / width, 0.0), 1.0)
        cy = min(max((y + h / 2) / height, 0.0), 1.0)
        nw, nh = min(max(w / width, 1e-6), 1.0), min(max(h / height, 1e-6), 1.0)
        labels.setdefault(ann["image_id"], []).append(
            f"{class_index[ann['category_id']]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
        )

    usable = [
        image_id for image_id in labels
        if images[image_id]["file_name"] not in excluded
        and (TACO_DATA_DIR / images[image_id]["file_name"]).is_file()
        and (TACO_DATA_DIR / images[image_id]["file_name"]).stat().st_size > 0
    ]
    random.Random(SEED).shuffle(usable)
    val_count = int(len(usable) * VAL_FRACTION)
    splits = {"valid": usable[:val_count], "train": usable[val_count:]}

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for split, ids in splits.items():
        (OUTPUT_DIR / split / "images").mkdir(parents=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True)
        for image_id in ids:
            source = TACO_DATA_DIR / images[image_id]["file_name"]
            name = Path(images[image_id]["file_name"]).name
            (OUTPUT_DIR / split / "images" / name).symlink_to(source.resolve())
            (OUTPUT_DIR / split / "labels" / (Path(name).stem + ".txt")).write_text(
                "\n".join(labels[image_id]) + "\n"
            )

    names = "\n".join(f"- {c['name']}" for c in categories)
    (OUTPUT_DIR / "data.yaml").write_text(f"nc: {len(categories)}\nnames:\n{names}\n")
    print(f"images train/valid: {len(splits['train'])}/{len(splits['valid'])} "
          f"(skipped {len(labels) - len(usable)} missing or excluded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
