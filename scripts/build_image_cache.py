#!/usr/bin/env python3
"""Losslessly cache the deterministic PIL Resize(256, 256) preprocessing prefix."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from goal.data import DEFAULT_IMAGE_CACHE, IMAGE_CACHE_VERSION, cache_entry_matches_source, image_cache_resize_spec, read_list


def save_manifest(manifest, path):
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    os.replace(str(temporary), str(path))


def build_one(source, output, resize, existing=None):
    source = Path(source).resolve()
    before = source.stat()
    if existing and cache_entry_matches_source(existing, before):
        cached = output / existing["cached_path"]
        try:
            cached_stat = cached.stat()
            if cached_stat.st_size == existing["cached_file_size"] and cached_stat.st_mtime_ns == existing["cached_mtime_ns"]:
                return str(source), existing, False
        except (OSError, KeyError):
            pass
    with Image.open(source) as handle:
        original = handle.convert("RGB")
    original_dimensions = list(original.size)
    resized = resize(original)
    pixel_hash = hashlib.sha256(resized.tobytes()).hexdigest()
    key = hashlib.sha256(str(source).encode()).hexdigest()
    relative = Path("images") / key[:2] / (key + ".png")
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(destination.parent), suffix=".png", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        resized.save(temporary, format="PNG", compress_level=3)
        # Each generated PNG is decoded and verified before publication.
        with Image.open(temporary) as handle:
            restored = handle.convert("RGB")
        if restored.size != (256, 256) or hashlib.sha256(restored.tobytes()).hexdigest() != pixel_hash:
            raise ValueError("PNG round-trip changed resized pixels: " + str(source))
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Source changed while building cache: " + str(source))
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            temporary.unlink()
    cached_stat = destination.stat()
    entry = {"cached_path": relative.as_posix(), "original_size": before.st_size,
             "original_mtime_ns": before.st_mtime_ns, "original_dimensions": original_dimensions,
             "resized_dimensions": [256, 256], "pixel_sha256": pixel_hash,
             "cached_file_size": cached_stat.st_size, "cached_mtime_ns": cached_stat.st_mtime_ns}
    return str(source), entry, True


def verify_one(item, output):
    source, entry = item
    source_stat = Path(source).stat()
    if not cache_entry_matches_source(entry, source_stat):
        raise ValueError("Stale source in cache: " + source)
    with Image.open(output / entry["cached_path"]) as handle:
        image = handle.convert("RGB")
    if image.size != (256, 256) or hashlib.sha256(image.tobytes()).hexdigest() != entry["pixel_sha256"]:
        raise ValueError("Cached pixel hash mismatch: " + source)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root")
    parser.add_argument("--lists-root", default=str(ROOT / "data" / "officehome"))
    parser.add_argument("--domains", default="Real_World,Art,Clipart,Product", help="List stems, in build priority order")
    parser.add_argument("--list-files", nargs="+", help="Explicit image lists instead of --domains")
    parser.add_argument("--output", default=str(DEFAULT_IMAGE_CACHE))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--verify", action="store_true", help="Verify every existing source signature and cached pixel hash")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".build.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = output / "manifest.json"
    spec = image_cache_resize_spec()
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {"version": IMAGE_CACHE_VERSION, "resize": spec, "images": {}}
    if manifest.get("version") != IMAGE_CACHE_VERSION or manifest.get("resize") != spec:
        raise ValueError("Existing cache uses a different resize/version; choose another output directory")
    started = time.monotonic()
    if args.verify:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(lambda item: verify_one(item, output), manifest["images"].items()))
        print(json.dumps({"event": "verified", "images": len(manifest["images"]), "seconds": time.monotonic() - started}), flush=True)
        return
    if not args.data_root:
        parser.error("--data-root is required when building a cache")
    from torchvision import transforms
    resize = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR)
    lists = [Path(path) for path in args.list_files] if args.list_files else [Path(args.lists_root) / (domain.strip() + ".txt") for domain in args.domains.split(",")]
    for list_path in lists:
        samples = read_list(list_path, args.data_root)
        rebuilt, completed = 0, 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(build_one, source, output, resize, manifest["images"].get(source)) for source, _ in samples]
            for future in as_completed(futures):
                source, entry, changed = future.result()
                manifest["images"][source] = entry
                rebuilt += int(changed)
                completed += 1
                if completed % 256 == 0:
                    print(json.dumps({"event": "progress", "domain": list_path.stem, "images": completed,
                                      "total": len(samples), "seconds": round(time.monotonic() - started, 2)}), flush=True)
        save_manifest(manifest, manifest_path)
        print(json.dumps({"event": "domain_complete", "domain": list_path.stem, "images": completed,
                          "rebuilt": rebuilt, "manifest_images": len(manifest["images"]),
                          "seconds": round(time.monotonic() - started, 2)}), flush=True)
    print(json.dumps({"event": "complete", "images": len(manifest["images"]),
                      "png_bytes": sum(entry["cached_file_size"] for entry in manifest["images"].values()),
                      "manifest": str(manifest_path), "seconds": round(time.monotonic() - started, 2)}), flush=True)


if __name__ == "__main__":
    main()
