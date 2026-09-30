"""Merge extra YOLO datasets (e.g. Roboflow downloads) into datasets/waste_v1.

Each source dataset needs a remap table from its own class names to the unified
9-class taxonomy (see CLASS_NAMES in convert_taco_to_yolo.py). Keys are
lowercase class names or fnmatch patterns (e.g. "bottle_*"). Boxes whose class
matches no key are dropped; images left without boxes are skipped. Map a class
to None when it is not waste we track at all (e.g. "branch"): its boxes are
dropped without counting as unmapped for ``drop_unmapped``.

Per-source options:
  dedupe_augmented   keep one copy per original image. Roboflow exports often
                     contain several augmented copies (``<stem>.rf.<hash>.jpg``).
  val_fraction       move this share of originals to val (for sources without a
                     val/test split).
  drop_unmapped      skip images that contain any box of an unmapped class, so
                     unlabeled objects of our classes do not become background.
  max_images         cap the number of originals taken from the source.
  max_boxes          skip images with more labelled boxes than this.
  max_box_area       skip images with a box covering more than this share of
                     the image (e.g. a whole pile boxed as one object).
  near_dedupe        drop images that are near-duplicates (difference hash) of a
                     TACO image or of an image merged earlier in this run, and
                     move val images that are near-duplicates of the source's own
                     train images to train. Needed for sources that re-host
                     photos from other datasets or contain burst shots.

Images are grouped by original stem across splits, so copies of one original
never end up in both train and val. Polygon labels are converted to boxes.

Images listed in configs/audit_exclude.txt (written from an audit of label
quality) are skipped.

The script is idempotent: files previously merged from a source (same filename
prefix) are removed before that source is merged again.

Usage:
  python scripts/merge_yolo_datasets.py
"""

from __future__ import annotations

import fnmatch
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

from convert_taco_to_yolo import TACO_TO_TARGET

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "datasets" / "waste_v1"
SEED = 42
NEAR_DUPLICATE_BITS = 10  # max differing bits (of 256) for two images to count as the same photo
# images flagged by scripts/audit_dataset.py as missing labels; one merged filename per line
AUDIT_EXCLUDE = ROOT / "configs" / "audit_exclude.txt"


def taco_remap() -> dict[str, int | None]:
    """TACO category -> class id; untracked categories are ignored, except
    "unlabeled litter", which could be any class and so drops its image."""
    names = yaml.safe_load((ROOT / "datasets" / "taco_unofficial" / "data.yaml").read_text())["names"] \
        if (ROOT / "datasets" / "taco_unofficial" / "data.yaml").is_file() else []
    return {
        name.lower(): TACO_TO_TARGET.get(name.strip().lower())
        for name in names
        if name.strip().lower() != "unlabeled litter"
    }

# Excluded after review (see scripts/audit_dataset.py):
#   trash_detection2      Roboflow re-export of TACO (duplicates, val leakage)
#   garbage_cls3          bookshelves as Paper, leaves labelled Glass
#   aluminium_cans        piles of cans with only a few labelled
#   plastic_litter        drone shots, objects of 2-5 px
#   beverage_containers   drinking glasses on dinner tables, cutout artefacts
#   vn_drinks, soda_bottles, waste_in_water, yolo_waste   incomplete or wrong labels
SOURCES: dict[str, dict] = {
    # community-annotated TACO photos: real-world litter, same taxonomy as waste_v1
    "datasets/taco_unofficial": {
        "remap": taco_remap(),  # "unlabeled litter" stays unmapped -> image dropped
        "drop_unmapped": True,
        "near_dedupe": True,
    },
    "datasets/roboflow/battery_detection": {
        "remap": {"battery": 8},
        "dedupe_augmented": True,
        "val_fraction": 0.15,
    },
    "datasets/roboflow/pet_bottle_type": {
        # bottle_* are PET bottles; cap_* classes are loose caps and are dropped
        "remap": {"bottle_*": 6},
    },
    "datasets/roboflow/paper_cup": {
        "remap": {"paper cup": 1},
        "dedupe_augmented": True,
        "max_images": 400,  # ~7 cups per image; about half land in val
    },
    "datasets/roboflow/styrofoam": {
        "remap": {"styrofoam": 7},
        "val_fraction": 0.15,
    },
    "datasets/roboflow/soda_can": {
        "remap": {"coke": 5, "pepsi": 5, "sprite": 5},
        "max_images": 100,  # ~6 cans per image, only three brands
    },
    "datasets/roboflow/soda_can_small": {
        "remap": {"soda-can": 5},
        "dedupe_augmented": True,
    },
    "datasets/roboflow/waste_mixed": {
        "remap": {"paper": 0, "cardboard": 0, "glass": 4},
        "drop_unmapped": True,
    },
    "datasets/roboflow/plastic_bag": {
        "remap": {"plastic-bag": 2},
        "dedupe_augmented": True,
        "max_images": 350,  # ~5 bags per image
    },
    "datasets/roboflow/plastic_bags_small": {
        "remap": {"plastic-bags": 2},
    },
    "datasets/roboflow/food_wrappers": {
        "remap": {"food-wrapper": 2},
        "dedupe_augmented": True,
        "val_fraction": 0.15,
        "max_images": 250,
    },
    # --- real-world scenes (waste_v3) ---
    "datasets/roboflow/litr_drinking": {
        # crushed / dirty bottles on streets, grass and decks; re-hosts some TACO photos
        "remap": {"clear plastic bottle": 6},
        "dedupe_augmented": True,
        "near_dedupe": True,
        "max_images": 1000,
    },
    "datasets/roboflow/trash_plastic_bottle": {
        "remap": {"plastic-bottles": 6},
        "dedupe_augmented": True,
        "near_dedupe": True,
    },
    "datasets/roboflow/plastic_bottles": {
        # bottle piles and crushed bottles; piles boxed as one object are dropped
        "remap": {"plastic bottle": 6},
        "max_box_area": 0.6,
        "max_images": 300,
    },
    "datasets/roboflow/river_trash": {
        "remap": {
            "plastic-bottle": 6,
            "metal-can": 5,
            "plastic-container": 3,
            "styrofoam": 7,
            "cardboard": 0,
            "plastic-bag": 2,
            # not waste we track
            "branch": None,
            "oil-spil": None,
            "clustered-trash": None,
            # "cup" is ambiguous (paper or plastic) -> unmapped, image dropped
        },
        "dedupe_augmented": True,
        "drop_unmapped": True,
        "max_boxes": 5,  # dump-site scenes label only a few of hundreds of objects
        "max_images": 600,
    },
    "datasets/roboflow/beach_litter": {
        # "plastic" is ambiguous -> unmapped, image dropped
        "remap": {"plastic bottle": 6, "fishing net": None},
        "dedupe_augmented": True,
        "drop_unmapped": True,
        "near_dedupe": True,
    },
    # --- printed-label bottles vs cans, glass bottles (waste_v4) ---
    "datasets/roboflow/bottle_can_pack": {
        # same backdrop and angle for labelled plastic bottles, cans and cartons
        "remap": {"plasticbottle_*": 6, "can_*": 5, "tetrapack_*": 0},
        "dedupe_augmented": True,
        "near_dedupe": True,
    },
    "datasets/roboflow/recycling_glass": {
        # single glass bottles and jars; likely overlaps waste_mixed, hence near_dedupe
        "remap": {"glass": 4, "paper": 0, "cardboard": 0},
        "dedupe_augmented": True,
        "drop_unmapped": True,
        "near_dedupe": True,
    },
    "datasets/roboflow/bottles_and_cans": {
        # crushed cans and bottles outdoors; same decking as litr_drinking
        "remap": {"bottle": 6, "can": 5},
        "dedupe_augmented": True,
        "near_dedupe": True,
        "max_images": 800,
    },
    "datasets/roboflow/plastic_waste_mgmt": {
        # same split as the TACO mapping: bags -> Vinyl, drink bottles -> PET,
        # cups/containers/utensils/straws -> rigid Plastic
        "remap": {
            "plastic bag": 2,
            "plastic bottle": 6,
            "plastic container": 3,
            "plastic cup": 3,
            "plastic utensil": 3,
            "plastic straw": 3,
        },
        "dedupe_augmented": True,
    },
}


def load_names(data_yaml: Path) -> dict[int, str]:
    data = yaml.safe_load(data_yaml.read_text())
    names = data.get("names", {})
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    return {int(key): str(value) for key, value in names.items()}


IGNORED = -1  # class is not waste we track; drop its boxes but keep the image


def match_class(name: str, remap: dict[str, int | None]) -> int | None:
    for pattern, target_id in remap.items():
        if fnmatch.fnmatchcase(name.lower(), pattern):
            return IGNORED if target_id is None else target_id
    return None


def original_stem(path: Path) -> str:
    return path.stem.split(".rf.")[0]


def find_image(image_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = image_dir / (stem + ext)
        if candidate.is_file():
            return candidate
    return None


def to_box(coords: list[str]) -> list[str] | None:
    """Return ``[cx, cy, w, h]`` for a YOLO box or polygon, or None if malformed."""
    if len(coords) == 4:
        return coords
    if len(coords) < 6 or len(coords) % 2:
        return None
    values = [float(value) for value in coords]
    xs, ys = values[0::2], values[1::2]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return [f"{(x0 + x1) / 2:.6f}", f"{(y0 + y1) / 2:.6f}", f"{x1 - x0:.6f}", f"{y1 - y0:.6f}"]


def dhash(path: Path) -> np.ndarray:
    """256-bit difference hash; robust to resizing and re-encoding."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(image, (17, 16), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()


def is_near_duplicate(image_hash: np.ndarray, hashes: np.ndarray) -> bool:
    return len(hashes) > 0 and int((hashes != image_hash).sum(1).min()) <= NEAR_DUPLICATE_BITS


def taco_hashes() -> np.ndarray:
    paths = [p for split in ("train", "val") for p in (TARGET / "images" / split).glob("batch_*")]
    return np.stack([dhash(p) for p in paths]) if paths else np.zeros((0, 256), dtype=bool)


def count_classes(split: str) -> Counter:
    counts: Counter = Counter()
    for label_file in (TARGET / "labels" / split).glob("*.txt"):
        for line in label_file.read_text().splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1
    return counts


def main() -> int:
    if not TARGET.exists():
        print("Run convert_taco_to_yolo.py first: datasets/waste_v1 not found.")
        return 1

    target_names = load_names(TARGET / "data.yaml")
    rng = random.Random(SEED)
    total_added = 0
    # near_dedupe reference: TACO plus every image written earlier in this run,
    # hashed lazily so runs without near_dedupe sources stay fast
    reference_hashes: np.ndarray | None = None
    written: list[Path] = []
    hashed_upto = 0
    audit_excluded = set(AUDIT_EXCLUDE.read_text().split()) if AUDIT_EXCLUDE.is_file() else set()
    if audit_excluded:
        print(f"skipping {len(audit_excluded)} images listed in {AUDIT_EXCLUDE.name}")

    for source_rel, config in SOURCES.items():
        source = ROOT / source_rel
        data_yaml = source / "data.yaml"
        if not data_yaml.is_file():
            print(f"skip {source_rel}: no data.yaml (not downloaded)")
            continue

        prefix = source_rel.split("/")[-1]
        for kind in ("images", "labels"):
            for split in ("train", "val"):
                for old in (TARGET / kind / split).glob(f"{prefix}_*"):
                    old.unlink()

        names = load_names(data_yaml)
        id_map = {class_id: match_class(name, config["remap"]) for class_id, name in names.items()}
        unmatched = sorted(name for class_id, name in names.items() if id_map[class_id] is None)
        if unmatched:
            print(f"{source_rel}: dropping unmapped classes {unmatched}")

        # (image, kept label lines, source split) grouped by original stem
        groups: dict[str, list[tuple[Path, list[str], str]]] = {}
        for split in ("train", "valid", "val", "test"):
            image_dir = source / split / "images"
            label_dir = source / split / "labels"
            if not label_dir.is_dir():
                continue
            in_split = "val" if split in ("valid", "val", "test") else "train"

            for label_file in sorted(label_dir.glob("*.txt")):
                kept_lines = []
                has_unmapped = False
                for line in label_file.read_text().splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    mapped = id_map.get(int(parts[0]))
                    if mapped == IGNORED:
                        continue
                    box = to_box(parts[1:])
                    if mapped is None or box is None:
                        has_unmapped = True
                        continue
                    kept_lines.append(" ".join([str(mapped), *box]))
                if not kept_lines or (has_unmapped and config.get("drop_unmapped")):
                    continue
                if len(kept_lines) > config.get("max_boxes", len(kept_lines)):
                    continue
                if "max_box_area" in config and any(
                    float(line.split()[3]) * float(line.split()[4]) > config["max_box_area"]
                    for line in kept_lines
                ):
                    continue

                image_file = find_image(image_dir, label_file.stem)
                if image_file is None or f"{prefix}_{image_file.name}" in audit_excluded:
                    continue
                groups.setdefault(original_stem(image_file), []).append(
                    (image_file, kept_lines, in_split)
                )

        group_hashes: dict[str, np.ndarray] = {}
        if config.get("near_dedupe"):
            if reference_hashes is None:
                reference_hashes = taco_hashes()
            if hashed_upto < len(written):
                new_hashes = np.stack([dhash(path) for path in written[hashed_upto:]])
                reference_hashes = np.concatenate([reference_hashes, new_hashes])
                hashed_upto = len(written)
            group_hashes = {key: dhash(members[0][0]) for key, members in groups.items()}
            copies = [key for key, h in group_hashes.items() if is_near_duplicate(h, reference_hashes)]
            for key in copies:
                del groups[key]
            if copies:
                print(f"{source_rel}: dropped {len(copies)} near-duplicates of images already merged")

        group_keys = sorted(groups)
        rng.shuffle(group_keys)
        if "max_images" in config:
            group_keys = group_keys[: config["max_images"]]
        forced_val = set(group_keys[: round(len(group_keys) * config.get("val_fraction", 0.0))])

        selected: list[tuple[str, list[tuple[Path, list[str], str]], str]] = []
        for key in group_keys:
            members = groups[key]
            if config.get("dedupe_augmented"):
                # prefer the val copy so the original is scored on, not trained on
                members = sorted(members, key=lambda member: member[2] != "val")[:1]
            in_val = key in forced_val or any(member[2] == "val" for member in members)
            selected.append((key, members, "val" if in_val else "train"))

        if group_hashes:
            train_hashes = [group_hashes[key] for key, _, split in selected if split == "train"]
            train_hashes = np.stack(train_hashes) if train_hashes else np.zeros((0, 256), dtype=bool)
            moved = 0
            for index, (key, members, split) in enumerate(selected):
                if split == "val" and is_near_duplicate(group_hashes[key], train_hashes):
                    selected[index] = (key, members, "train")
                    moved += 1
            if moved:
                print(f"{source_rel}: moved {moved} val images that duplicate train images to train")

        added = Counter()
        for key, members, out_split in selected:
            for image_file, kept_lines, _ in members:
                out_name = f"{prefix}_{image_file.name}"
                out_image = TARGET / "images" / out_split / out_name
                shutil.copy2(image_file, out_image)
                written.append(out_image)
                (TARGET / "labels" / out_split / (Path(out_name).stem + ".txt")).write_text(
                    "\n".join(kept_lines) + "\n"
                )
                added[out_split] += 1

        print(f"{source_rel}: +{added['train']} train, +{added['val']} val images")
        total_added += sum(added.values())

    for cache in (TARGET / "labels").glob("*.cache"):
        cache.unlink()

    print(f"total added: {total_added}\n")
    train_counts, val_counts = count_classes("train"), count_classes("val")
    print(f"{'class':<14}{'train':>8}{'val':>8}")
    for class_id, name in sorted(target_names.items()):
        print(f"{class_id} {name:<12}{train_counts[class_id]:>8}{val_counts[class_id]:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
