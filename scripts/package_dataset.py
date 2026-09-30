"""Package datasets/waste_v1 into a resized zip for Colab training.

Images are downscaled so the longest side is at most MAX_SIDE (YOLO labels are
normalised, so they stay valid) and re-encoded as JPEG. data.yaml inside the
zip points at /content/<name>, where the Colab notebook unzips it.

Training images that contain a class in REPEAT_CLASSES are written
REPEAT_TIMES times (repeat-factor sampling), so rare classes are seen more
often per epoch; the val split is never repeated.

Usage:
  python scripts/package_dataset.py waste_v2
  -> ./waste_v2.zip containing waste_v2/{images,labels}/{train,val}, data.yaml
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets" / "waste_v1"
MAX_SIDE = 1280
JPEG_QUALITY = 90
REPEAT_CLASSES = {4, 7}  # Glass, Styrofoam: the rarest classes after the label audit
REPEAT_TIMES = 2


def resize_image(src: Path, dst: Path) -> bool:
    image = cv2.imread(str(src))  # applies EXIF orientation, same as Ultralytics
    if image is None:
        return False
    height, width = image.shape[:2]
    scale = MAX_SIDE / max(height, width)
    if scale < 1:
        image = cv2.resize(
            image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
        )
    return cv2.imwrite(str(dst.with_suffix(".jpg")), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "waste_v1_merged"
    names = yaml.safe_load((SOURCE / "data.yaml").read_text())["names"]

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / name
        jobs = []
        for split in ("train", "val"):
            (stage / "images" / split).mkdir(parents=True)
            shutil.copytree(SOURCE / "labels" / split, stage / "labels" / split,
                            ignore=shutil.ignore_patterns("*.cache"))
            for image in (SOURCE / "images" / split).iterdir():
                jobs.append((image, stage / "images" / split / image.name))

        with ThreadPoolExecutor(8) as pool:
            failed = [src for (src, _), ok in zip(jobs, pool.map(lambda job: resize_image(*job), jobs)) if not ok]
        if failed:
            print(f"failed to convert {len(failed)} images, e.g. {failed[:3]}")
            return 1

        for split in ("train", "val"):
            image_stems = {path.stem for path in (stage / "images" / split).iterdir()}
            label_stems = {path.stem for path in (stage / "labels" / split).glob("*.txt")}
            if image_stems != label_stems:
                print(f"{split}: {len(image_stems ^ label_stems)} images/labels do not pair up")
                return 1
            print(f"{split}: {len(image_stems)} images")

        repeated = 0
        for label in list((stage / "labels" / "train").glob("*.txt")):
            classes = {int(line.split()[0]) for line in label.read_text().splitlines() if line.strip()}
            if classes & REPEAT_CLASSES:
                image = stage / "images" / "train" / (label.stem + ".jpg")
                for copy in range(1, REPEAT_TIMES):
                    shutil.copy2(image, image.with_name(f"{label.stem}_rep{copy}.jpg"))
                    shutil.copy2(label, label.with_name(f"{label.stem}_rep{copy}.txt"))
                    repeated += 1
        print(f"train: +{repeated} repeated images for classes {sorted(REPEAT_CLASSES)}")

        data ={"path": f"/content/{name}", "train": "images/train", "val": "images/val",
                "nc": len(names), "names": names}
        (stage / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))

        archive = shutil.make_archive(str(ROOT / name), "zip", root_dir=tmp, base_dir=name)
    print(f"wrote {archive} ({Path(archive).stat().st_size / 2**20:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
