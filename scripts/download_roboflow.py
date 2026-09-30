"""Download supplementary waste datasets from Roboflow Universe (YOLOv8 format).

Requires a free Roboflow API key:
  export ROBOFLOW_API_KEY=your_key
  python scripts/download_roboflow.py

Each downloaded dataset keeps its own class list; merge into the unified
9-class taxonomy afterwards with scripts/merge_yolo_datasets.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "datasets" / "roboflow"

# (workspace, project, version, folder name)
DATASETS = [
    ("yolov8-trashdetection", "trash-detection2-kqt3w", 1, "trash_detection2"),
    ("waste-seregation", "battery-detection-m7onv", 1, "battery_detection"),
    ("project-yysbf", "pet-bottle-type", 12, "pet_bottle_type"),
    ("project-zq0wz", "paper-cup-xhrgq", 3, "paper_cup"),
    ("a-3zezt", "styrofoam-wfuot", 1, "styrofoam"),
    ("personal-g5mzf", "soda-can-object-detection", 3, "soda_can"),
    ("cyberwarriorstemcamp", "soda-can-qr4c4", 2, "soda_can_small"),
    ("waste-classification-3zmlr", "waste-hsysm-xnrsr", 1, "waste_mixed"),
    ("ecosorter-drvgm", "plastic-bag-dataset", 6, "plastic_bag"),
    ("dataset-t7hz7", "plastic-bags-0qzjp", 3, "plastic_bags_small"),
    ("aia-assignment-4xyx1", "food-wrappers-dpjkz-fer1g", 1, "food_wrappers"),
    ("yolov5-6agzx", "plastic-waste-management-6p8kw", 15, "plastic_waste_mgmt"),
    # real-world scenes: litter in the wild, rivers, beaches, bottle piles
    ("k-s", "litr-drinking-waste", 2, "litr_drinking"),
    ("trisha-lingat-m9vjl", "river-trash-final-k9997", 2, "river_trash"),
    ("waste-rq8p9", "plastic-bottles-uu8v9", 1, "plastic_bottles"),
    ("ros", "trash-plastic-bottle-detection", 2, "trash_plastic_bottle"),
    ("beach-litter-jgn6d", "beach-litter-wt1od", 1, "beach_litter"),
    # printed-label bottles vs cans (PET mistaken for Can), plus glass bottles
    ("recyclorobloai-intern", "can-bottle-and-pack-detection", 1, "bottle_can_pack"),
    ("raj-sangani-ygq8p", "bottles-and-cans", 1, "bottles_and_cans"),
    ("recyclestuff", "updated-recycling-dataset", 9, "recycling_glass"),
]


def main() -> int:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("ROBOFLOW_API_KEY is not set. Get a free key at app.roboflow.com -> Settings -> API.")
        return 1

    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    for workspace, project_id, version, folder in DATASETS:
        location = OUTPUT_ROOT / folder
        print(f"Downloading {workspace}/{project_id}/{version} -> {location}")
        try:
            project = rf.workspace(workspace).project(project_id)
            project.version(version).download(
                model_format="yolov8",
                location=str(location),
                overwrite=False,
            )
        except Exception as exc:  # noqa: BLE001 - continue with remaining datasets
            print(f"  FAILED: {exc}")
        else:
            print("  OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
