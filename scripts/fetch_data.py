"""Download the COCO val2017 subset (Phase A) and the NUS clips (Phase B + benchmark)."""

import sys

from gimbal.bench.datasets import fetch_coco, fetch_nus

what = sys.argv[1] if len(sys.argv) > 1 else "all"

if what in ("all", "coco"):
    print("fetching COCO val2017 subset...", flush=True)
    d = fetch_coco(limit=5000)
    n = len(list(d.glob("*.jpg")))
    print(f"COCO ready: {n} images in {d}", flush=True)

if what in ("all", "nus"):
    print("fetching NUS clips...", flush=True)
    d = fetch_nus()
    clips = sum(1 for _ in d.rglob("*.avi"))
    print(f"NUS ready: {clips} clips under {d}", flush=True)

print("DONE", flush=True)
