"""Audit label quality of datasets/waste_v1, per source, with a trained model.

For every image the model's predictions are compared with the labels:
  missing  confident prediction (conf >= --conf-high) that overlaps no label
           (IoU < 0.3): probably an object someone forgot to label
  conflict label and confident prediction overlap (IoU >= 0.5) but disagree on
           the class: a mislabel or a taxonomy mismatch between sources
  undetected label that no prediction (conf >= 0.25) overlaps (IoU >= 0.5):
           a hard object, or a wrong / sloppy box
Geometry checks flag tiny boxes, boxes covering almost the whole image and
duplicated boxes. Training images are included on purpose: after training, a
model that still disagrees with a training label is a strong hint the label is
wrong. Results are per source (filename prefix) and per split.

Writes <out>/audit_summary.csv, <out>/audit_images.csv and contact sheets of
the worst images of each source.

Usage:
  python scripts/audit_dataset.py --model models/waste_v3.pt --out audit
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "waste_v1"
NAMES = ["Paper", "Paper Cup", "Vinyl", "Plastic", "Glass", "Can", "PET", "Styrofoam", "Battery"]


def source_of(name: str) -> str:
    if name.startswith("batch_"):
        return "taco"
    # merged files are "<source>_<original name>"; sources are known folder names
    for source in SOURCES:
        if name.startswith(source + "_"):
            return source
    return "unknown"


SOURCES = sorted(
    (p.name for p in (ROOT / "datasets" / "roboflow").iterdir() if p.is_dir()),
    key=len,
    reverse=True,  # longest first so "soda_can_small" wins over "soda_can"
) if (ROOT / "datasets" / "roboflow").is_dir() else []


def read_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []
    if not rows:
        return np.zeros(0, dtype=int), np.zeros((0, 4))
    arr = np.array([[float(v) for v in r[:5]] for r in rows])
    return arr[:, 0].astype(int), arr[:, 1:]


def xywhn_to_xyxy(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    cx, cy, bw, bh = boxes.T
    return np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], 1)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    from ultralytics.utils.metrics import box_iou

    return box_iou(torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)).numpy()


def audit(model_path: Path, conf_high: float, device: str | None) -> tuple[list[dict], dict]:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    images = [p for split in ("train", "val") for p in sorted((DATASET / "images" / split).iterdir())]
    rows: list[dict] = []
    conflicts: dict[str, Counter] = defaultdict(Counter)
    for start in range(0, len(images), 32):
        batch = images[start:start + 32]
        results = model.predict([str(p) for p in batch], imgsz=640, conf=0.25, agnostic_nms=True,
                                device=device, verbose=False)
        for image, result in zip(batch, results):
            split = image.parent.name
            h, w = result.orig_shape
            gt_cls, gt_xywh = read_labels(DATASET / "labels" / split / (image.stem + ".txt"))
            gt = xywhn_to_xyxy(gt_xywh, w, h)
            pred = result.boxes.xyxy.cpu().numpy()
            pred_cls = result.boxes.cls.cpu().numpy().astype(int)
            pred_conf = result.boxes.conf.cpu().numpy()
            iou = iou_matrix(gt, pred)

            confident = pred_conf >= conf_high
            missing = int(sum(1 for j in np.where(confident)[0] if not len(gt) or iou[:, j].max() < 0.3))
            undetected = int(sum(1 for i in range(len(gt)) if not len(pred) or iou[i].max() < 0.5))
            conflict = 0
            for i in range(len(gt)):
                if not len(pred):
                    continue
                j = int(iou[i].argmax())
                if iou[i, j] >= 0.5 and confident[j] and pred_cls[j] != gt_cls[i]:
                    conflict += 1
                    conflicts[source_of(image.name)][(NAMES[gt_cls[i]], NAMES[pred_cls[j]])] += 1

            tiny = int(((gt_xywh[:, 2] * w < 8) | (gt_xywh[:, 3] * h < 8)).sum()) if len(gt) else 0
            huge = int((gt_xywh[:, 2] * gt_xywh[:, 3] > 0.9).sum()) if len(gt) else 0
            dup_iou = iou_matrix(gt, gt)
            np.fill_diagonal(dup_iou, 0) if len(gt) else None
            duplicate = int((np.triu(dup_iou) > 0.9).sum()) if len(gt) else 0

            rows.append({
                "image": image.name, "split": split, "source": source_of(image.name),
                "boxes": len(gt), "classes": " ".join(sorted({NAMES[c] for c in gt_cls})),
                "missing": missing, "conflict": conflict, "undetected": undetected,
                "tiny": tiny, "huge": huge, "duplicate": duplicate,
            })
        print(f"\r{start + len(batch)}/{len(images)}", end="", flush=True)
    print()
    return rows, conflicts


def summarise(rows: list[dict], conflicts: dict, out: Path) -> None:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)

    header = ["source", "images", "boxes", "img_missing%", "box_conflict%", "box_undetected%",
              "tiny", "huge", "duplicate", "top_conflicts"]
    table = []
    for source, items in sorted(by_source.items()):
        boxes = sum(r["boxes"] for r in items) or 1
        table.append({
            "source": source, "images": len(items), "boxes": sum(r["boxes"] for r in items),
            "img_missing%": round(100 * sum(r["missing"] > 0 for r in items) / len(items), 1),
            "box_conflict%": round(100 * sum(r["conflict"] for r in items) / boxes, 1),
            "box_undetected%": round(100 * sum(r["undetected"] for r in items) / boxes, 1),
            "tiny": sum(r["tiny"] for r in items), "huge": sum(r["huge"] for r in items),
            "duplicate": sum(r["duplicate"] for r in items),
            "top_conflicts": "; ".join(f"{a}->{b}:{n}" for (a, b), n in conflicts[source].most_common(3)),
        })
    with (out / "audit_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(table)
    with (out / "audit_images.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'source':<22}{'imgs':>6}{'boxes':>7}{'miss%':>7}{'confl%':>8}{'undet%':>8}{'tiny':>6}{'huge':>6}{'dup':>5}  top conflicts")
    for t in sorted(table, key=lambda t: -t["img_missing%"]):
        print(f"{t['source']:<22}{t['images']:>6}{t['boxes']:>7}{t['img_missing%']:>7}{t['box_conflict%']:>8}"
              f"{t['box_undetected%']:>8}{t['tiny']:>6}{t['huge']:>6}{t['duplicate']:>5}  {t['top_conflicts']}")


def contact_sheets(rows: list[dict], out: Path, per_source: int = 8) -> None:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    for source, items in by_source.items():
        worst = sorted(items, key=lambda r: -(r["missing"] * 2 + r["conflict"] * 2 + r["undetected"]))[:per_source]
        tiles = []
        for r in worst:
            image = cv2.imread(str(DATASET / "images" / r["split"] / r["image"]))
            h, w = image.shape[:2]
            cls, xywh = read_labels(DATASET / "labels" / r["split"] / (Path(r["image"]).stem + ".txt"))
            for c, (x0, y0, x1, y1) in zip(cls, xywhn_to_xyxy(xywh, w, h)):
                cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), max(2, w // 200))
                cv2.putText(image, NAMES[c], (int(x0), int(y0) + max(15, h // 25)), 0, max(0.6, w / 800), (0, 255, 255), 2)
            tile = cv2.resize(image, (320, 260))
            cv2.putText(tile, f"m{r['missing']} c{r['conflict']} u{r['undetected']}", (4, 252), 0, 0.6, (255, 0, 255), 2)
            tiles.append(tile)
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        cv2.imwrite(str(out / f"worst_{source}.jpg"), sheet)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "waste_v3.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "audit")
    parser.add_argument("--conf-high", type=float, default=0.6)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows, conflicts = audit(args.model, args.conf_high, args.device)
    summarise(rows, conflicts, args.out)
    contact_sheets(rows, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
