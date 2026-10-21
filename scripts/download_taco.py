"""Parallel TACO image downloader.

Downloads the 640px Flickr renditions referenced in TACO's annotations.json
(falls back to the original URL when no 640px rendition exists). Safe to
re-run: existing files are skipped.

Usage:
  python scripts/download_taco.py                              # official set
  python scripts/download_taco.py annotations_unofficial.json  # community set
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

TACO_DATA_DIR = Path(__file__).resolve().parents[1] / "datasets" / "taco_repo" / "data"
ANNOTATIONS = TACO_DATA_DIR / "annotations.json"
WORKERS = 16
TIMEOUT_S = 30


def download_one(image: dict) -> tuple[str, bool, str]:
    file_name = image["file_name"]
    target = TACO_DATA_DIR / file_name
    if target.is_file() and target.stat().st_size > 0:
        return file_name, True, "cached"

    target.parent.mkdir(parents=True, exist_ok=True)
    url = image.get("flickr_640_url") or image.get("flickr_url")
    if not url:
        return file_name, False, "no url"

    try:
        response = requests.get(url, timeout=TIMEOUT_S)
        response.raise_for_status()
        target.write_bytes(response.content)
        return file_name, True, "downloaded"
    except Exception as exc:  # noqa: BLE001 - report and continue
        return file_name, False, str(exc)


def main() -> int:
    annotations = TACO_DATA_DIR / sys.argv[1] if len(sys.argv) > 1 else ANNOTATIONS
    images = json.loads(annotations.read_text())["images"]
    print(f"Total images: {len(images)}", flush=True)

    ok = 0
    failed: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(download_one, image) for image in images]
        for done, future in enumerate(as_completed(futures), start=1):
            name, success, note = future.result()
            if success:
                ok += 1
            else:
                failed.append((name, note))
            if done % 100 == 0 or done == len(images):
                print(f"progress {done}/{len(images)} ok={ok} failed={len(failed)}", flush=True)

    if failed:
        print("Failed downloads:")
        for name, note in failed[:30]:
            print(f"  {name}: {note}")
    print(f"DONE ok={ok} failed={len(failed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
