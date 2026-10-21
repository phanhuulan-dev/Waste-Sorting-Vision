"""Convert TACO (COCO format) into a YOLO detection dataset with remapped classes.

TACO's 60 litter categories are remapped onto the 9-class taxonomy used by the
new Waste Sorting Vision training run. Categories that do not fit any target
class are skipped. Images whose annotations are all skipped are excluded.

Output layout (created under datasets/waste_v1/):
  images/{train,val}/*.jpg
  labels/{train,val}/*.txt
  data.yaml
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TACO_DATA_DIR = ROOT / "datasets" / "taco_repo" / "data"
OUTPUT_DIR = ROOT / "datasets" / "waste_v1"
VAL_FRACTION = 0.15
SEED = 42

CLASS_NAMES = [
    "Paper",       # 0
    "Paper Cup",   # 1
    "Vinyl",       # 2
    "Plastic",     # 3
    "Glass",       # 4
    "Can",         # 5
    "PET",         # 6
    "Styrofoam",   # 7
    "Battery",     # 8
]

# TACO category name (lowercase) -> target class id. Unlisted categories are skipped.
TACO_TO_TARGET = {
    # Paper / cartons
    "normal paper": 0,
    "magazine paper": 0,
    "wrapping paper": 0,
    "paper bag": 0,
    "toilet tube": 0,
    "other carton": 0,
    "egg carton": 0,
    "drink carton": 0,
    "corrugated carton": 0,
    "meal carton": 0,
    "pizza box": 0,
    # Paper cup
    "paper cup": 1,
    # Vinyl (films, bags, wrappers)
    "plastic film": 2,
    "six pack rings": 2,
    "garbage bag": 2,
    "other plastic wrapper": 2,
    "single-use carrier bag": 2,
    "polypropylene bag": 2,
    "crisp packet": 2,
    "plastified paper bag": 2,
    # Rigid plastic
    "other plastic bottle": 3,
    "plastic bottle cap": 3,
    "disposable plastic cup": 3,
    "other plastic cup": 3,
    "plastic lid": 3,
    "other plastic": 3,
    "spread tub": 3,
    "tupperware": 3,
    "disposable food container": 3,
    "other plastic container": 3,
    "plastic glooves": 3,
    "plastic gloves": 3,
    "plastic utensils": 3,
    "squeezable tube": 3,
    "plastic straw": 3,
    # Glass
    "glass bottle": 4,
    "broken glass": 4,
    "glass jar": 4,
    "glass cup": 4,
    # Metal cans
    "food can": 5,
    "aerosol": 5,
    "drink can": 5,
    # PET
    "clear plastic bottle": 6,
    # Styrofoam
    "styrofoam piece": 7,
    "foam cup": 7,
    "foam food container": 7,
    # Battery
    "battery": 8,
}


def main() -> int:
    annotations = json.loads((TACO_DATA_DIR / "annotations.json").read_text())

    category_map: dict[int, int] = {}
    for category in annotations["categories"]:
        target = TACO_TO_TARGET.get(category["name"].strip().lower())
        if target is not None:
            category_map[category["id"]] = target

    images_by_id = {image["id"]: image for image in annotations["images"]}
    labels_by_image: dict[int, list[str]] = {}

    kept_annotations = 0
    skipped_annotations = 0
    for annotation in annotations["annotations"]:
        target = category_map.get(annotation["category_id"])
        if target is None:
            skipped_annotations += 1
            continue

        image = images_by_id[annotation["image_id"]]
        width, height = float(image["width"]), float(image["height"])
        x, y, w, h = annotation["bbox"]

        cx = (x + w / 2) / width
        cy = (y + h / 2) / height
        nw = w / width
        nh = h / height
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        nw, nh = min(max(nw, 1e-6), 1.0), min(max(nh, 1e-6), 1.0)

        labels_by_image.setdefault(annotation["image_id"], []).append(
            f"{target} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
        )
        kept_annotations += 1

    usable: list[tuple[int, Path]] = []
    missing_files = 0
    for image_id, lines in labels_by_image.items():
        source = TACO_DATA_DIR / images_by_id[image_id]["file_name"]
        if source.is_file() and source.stat().st_size > 0:
            usable.append((image_id, source))
        else:
            missing_files += 1

    random.Random(SEED).shuffle(usable)
    val_count = max(1, int(len(usable) * VAL_FRACTION))
    splits = {"val": usable[:val_count], "train": usable[val_count:]}

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for split, items in splits.items():
        image_dir = OUTPUT_DIR / "images" / split
        label_dir = OUTPUT_DIR / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for image_id, source in items:
            flat_name = images_by_id[image_id]["file_name"].replace("/", "_")
            shutil.copy2(source, image_dir / flat_name)
            (label_dir / (Path(flat_name).stem + ".txt")).write_text(
                "\n".join(labels_by_image[image_id]) + "\n"
            )

    names_block = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    (OUTPUT_DIR / "data.yaml").write_text(
        f"path: {OUTPUT_DIR}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n{names_block}\n"
    )

    per_class = {name: 0 for name in CLASS_NAMES}
    for lines in labels_by_image.values():
        for line in lines:
            per_class[CLASS_NAMES[int(line.split()[0])]] += 1

    print(f"kept annotations:    {kept_annotations}")
    print(f"skipped annotations: {skipped_annotations}")
    print(f"images train/val:    {len(splits['train'])}/{len(splits['val'])}")
    print(f"missing image files: {missing_files}")
    print("per-class boxes:")
    for name, count in per_class.items():
        print(f"  {name}: {count}")
    print(f"data.yaml: {OUTPUT_DIR / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
