#!/usr/bin/env python3
"""Build benchmark lists from an existing class-directory dataset."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goal.data import BENCHMARKS, build_benchmark_lists


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmark", choices=list(BENCHMARKS), default="office-home")
    parser.add_argument("--strict-counts", action="store_true")
    parser.add_argument("--class-map", help="JSON class_name -> integer mapping used by an existing source checkpoint")
    args = parser.parse_args()
    class_map = None
    if args.class_map:
        class_map = json.loads(Path(args.class_map).read_text())
        if "class_to_idx" in class_map:
            class_map = class_map["class_to_idx"]
    manifest = build_benchmark_lists(args.data_root, args.output, args.benchmark, args.strict_counts, class_map)
    if args.class_map:
        manifest["class_map_file"] = str(Path(args.class_map).resolve())
        manifest["class_map_sha256"] = hashlib.sha256(Path(args.class_map).read_bytes()).hexdigest()
        (Path(args.output) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"benchmark": args.benchmark, "classes": len(manifest["classes"]),
                      "domains": {name: {key: value for key, value in info.items() if key != "per_class"}
                                  for name, info in manifest["domains"].items()}}, indent=2))


if __name__ == "__main__":
    main()
