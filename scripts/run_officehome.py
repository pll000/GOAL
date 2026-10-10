#!/usr/bin/env python3
"""Run an Office-Home, Office-31 or ImageCLEF matrix on physical GPUs.

The scheduler uses only the standard library. GPU indices are resolved through
nvidia-smi to UUIDs, avoiding CUDA ordinal changes when a device is unavailable.
Existing completed runs are skipped; --resume permits epoch-boundary recovery.
"""

import argparse
from collections import deque
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DOMAIN_MAPS = {
    "officehome": {"A": "Art", "C": "Clipart", "P": "Product", "R": "Real_World"},
    "office31": {"A": "amazon", "D": "dslr", "W": "webcam"},
    "imageclef": {"C": "c", "I": "i", "P": "p"},
}
DOMAINS = DOMAIN_MAPS["officehome"]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def comma_list(value):
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("Expected a nonempty comma-separated list with no duplicates")
    return result


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def save_json(value, path):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(str(temporary), str(path))


def gpu_mapping(gpus):
    mapping = []
    for device in gpus:
        if device.startswith("GPU-"):
            query_device = device
        elif device.isdigit():
            query_device = str(int(device))
        else:
            raise ValueError("--gpus accepts physical nvidia-smi indices or GPU UUIDs: " + device)
        result = subprocess.run(["nvidia-smi", "-i", query_device, "--query-gpu=index,uuid", "--format=csv,noheader"],
                                check=True, capture_output=True, text=True, timeout=15)
        lines = result.stdout.strip().splitlines()
        if len(lines) != 1:
            raise ValueError("nvidia-smi did not identify exactly one GPU for " + device)
        index, uuid = (field.strip() for field in lines[0].split(",", 1))
        if not index.isdigit() or not uuid.startswith("GPU-"):
            raise ValueError("Unexpected nvidia-smi output: " + lines[0])
        mapping.append({"physical_index": int(index), "uuid": uuid})
    if len({item["uuid"] for item in mapping}) != len(mapping):
        raise ValueError("Two --gpus entries resolve to the same physical GPU")
    return mapping


def expected_config(args, preset, seed):
    path = Path(args.config_root) / (args.dataset + "_" + preset + ".json")
    config = json.loads(path.read_text())
    for key in ("epochs", "batch_size", "workers", "lr", "max_steps"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    config["seed"] = seed
    return path.resolve(), config


def checkpoint_digest(path):
    digest = hashlib.sha256()
    if path.is_dir():
        for file in sorted(path.glob("source_*.pt")):
            digest.update(file.name.encode())
            digest.update(checkpoint_digest(file).encode())
    else:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def make_jobs(args):
    domains = DOMAIN_MAPS[args.dataset]
    lists_root = Path(args.lists_root) if args.lists_root else ROOT / "data" / args.dataset
    pairs = [a + "2" + b for a in domains for b in domains if a != b] if args.pairs.lower() == "all" else comma_list(args.pairs.upper())
    valid_pairs = {a + "2" + b for a in domains for b in domains if a != b}
    if not set(pairs).issubset(valid_pairs):
        raise ValueError("Pairs for {} must be one of {}".format(args.dataset, ",".join(sorted(valid_pairs))))
    seeds = [int(value) for value in comma_list(args.seeds)]
    jobs, source_digests = [], {}
    # Pair first: compare all presets/seeds before moving to the next transfer.
    for pair in pairs:
        source, target = pair.split("2")
        target_list = lists_root.resolve() / (domains[target] + ".txt")
        source_key = source if args.source_layout == "domain" else pair.lower()
        source_path = Path(args.source_root).resolve() / source_key
        if not source_path.is_dir():
            alternate_key = source.lower() if args.source_layout == "domain" else pair.upper()
            alternate = Path(args.source_root).resolve() / alternate_key
            if alternate.is_dir():
                source_path = alternate
        if not target_list.is_file():
            raise FileNotFoundError("Missing target image list: " + str(target_list))
        if not ((source_path / "source_best.pt").is_file() or
                ((source_path / "source_F.pt").is_file() and (source_path / "source_C.pt").is_file())):
            raise FileNotFoundError("No structured or legacy source checkpoint in " + str(source_path))
        if source_path not in source_digests:
            source_digests[source_path] = checkpoint_digest(source_path)
        for preset in args.presets:
            for seed in seeds:
                config_path, config = expected_config(args, preset, seed)
                output = Path(args.output_root).resolve() / preset / ("seed" + str(seed)) / pair
                command = [args.python, "-u", "-m", "goal.train", "--config", str(config_path),
                           "--target-list", str(target_list), "--data-root", str(Path(args.data_root).resolve()),
                           "--source", str(source_path), "--source-domain", source, "--target-domain", target,
                           "--output", str(output), "--device", "cuda:0", "--seed", str(seed),
                           "--eval-every", str(args.eval_every), "--save-every", str(args.save_every),
                           "--cpu-threads", str(args.cpu_threads)]
                for key in ("epochs", "batch_size", "workers", "lr", "max_steps"):
                    if getattr(args, key) is not None:
                        command.extend(["--" + key.replace("_", "-"), str(getattr(args, key))])
                if args.check_gradients:
                    command.append("--check-gradients")
                jobs.append({"id": "{}/seed{}/{}".format(preset, seed, pair), "pair": pair,
                             "preset": preset, "seed": seed, "config": config, "output": output,
                             "source": source_path, "target_list": target_list, "command": command})
                jobs[-1]["source_sha256"] = source_digests[source_path]
    return jobs


def classify_job(job, args):
    output = job["output"]
    summary, previous = read_json(output / "summary.json"), read_json(output / "run.json")
    if summary and summary.get("config") != job["config"]:
        return "blocked", "existing summary has different configuration; choose a new output root"
    if previous:
        expected = {"config": job["config"], "source": str(job["source"]),
                    "source_sha256": job["source_sha256"],
                    "target_list": str(job["target_list"]), "data_root": str(Path(args.data_root).resolve()),
                    "target_list_sha256": hashlib.sha256(job["target_list"].read_bytes()).hexdigest(),
                    "eval_every": args.eval_every}
        for key, value in expected.items():
            if previous.get(key) != value:
                return "blocked", "existing run differs in {}; choose a new output root".format(key)
    if summary and summary.get("status") == "completed":
        return "skipped", "already completed with matching configuration and data paths"
    if summary and summary.get("status") == "smoke_test":
        return "blocked", "smoke-test output cannot be resumed; choose a new output root"
    last = output / "last.pt"
    existing = (output / "metrics.jsonl").exists() or (output / "run.json").exists() or last.exists()
    if existing:
        if not args.resume:
            return "blocked", "unfinished output exists; pass --resume for an epoch checkpoint or choose a new root"
        if job["config"].get("max_steps"):
            return "blocked", "smoke tests cannot be resumed"
        if not last.is_file():
            return "blocked", "unfinished output has no last.pt to resume"
        if "--resume" not in job["command"]:
            job["command"].extend(["--resume", str(last)])
        return "pending", "resume last.pt; trainer validates epoch boundary and checkpoint provenance"
    return "pending", "new run"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DOMAIN_MAPS), default="officehome")
    parser.add_argument("--source-layout", choices=("domain", "pair"), default="domain",
                        help="Source subdirectories: A/C/P/R (domain) or a2d/i2p etc. (pair)")
    parser.add_argument("--gpus", type=comma_list, required=True, help="Physical nvidia-smi indices, e.g. 5,6,7")
    parser.add_argument("--presets", type=comma_list, default=["goal_v2"])
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--seeds", default="2021")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-root", required=True, help="Parent directory containing source checkpoint subdirectories")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--lists-root", help="Defaults to GOAL/data/<dataset>")
    parser.add_argument("--config-root", default=str(ROOT / "configs"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--check-gradients", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    args.python = shutil.which(args.python) or str(Path(args.python).resolve())
    if args.eval_every < 1 or args.save_every < 1 or args.cpu_threads < 1:
        raise ValueError("evaluation, save and thread settings must be positive")
    mapping = gpu_mapping(args.gpus)
    jobs = make_jobs(args)
    print(json.dumps({"event": "gpu_mapping", "gpus": mapping}), flush=True)
    if args.dry_run:
        for index, job in enumerate(jobs):
            status, reason = classify_job(job, args)
            print(json.dumps({"job": job["id"], "status": status, "reason": reason,
                              "gpu": mapping[index % len(mapping)], "command": shlex.join(job["command"]),
                              "cwd": str(ROOT), "output": str(job["output"])}), flush=True)
        return 0
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    queue_log = (output_root / "queue.jsonl").open("a", buffering=1)
    results, active = [], {}
    pending = deque(jobs)

    def report(event):
        event = dict(event, timestamp=stamp())
        line = json.dumps(event)
        queue_log.write(line + "\n")
        print(line, flush=True)

    def record_status(job, status, **extra):
        value = dict(job=job["id"], status=status, timestamp=stamp(), **extra)
        save_json(value, job["output"] / "queue_status.json")
        return value

    def finish(slot, returncode):
        item = active.pop(slot)
        item["log"].close()
        summary = read_json(item["job"]["output"] / "summary.json")
        success = returncode == 0 and summary and summary.get("status") in ("completed", "smoke_test")
        status = summary["status"] if success else "failed"
        record = record_status(item["job"], status, exit_code=returncode, gpu=item["gpu"],
                               pid=item["process"].pid, seconds=round(time.monotonic() - item["started"], 2))
        results.append(record)
        report(dict(record, event="finished"))
        item["lock"].close()

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    report({"event": "queue_start", "jobs": len(jobs), "gpus": mapping, "pid": os.getpid()})
    try:
        while pending or active:
            for slot, gpu in enumerate(mapping):
                if slot in active:
                    continue
                while pending:
                    job = pending.popleft()
                    job["output"].mkdir(parents=True, exist_ok=True)
                    lock = (job["output"] / ".queue.lock").open("a")
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        lock.close()
                        record = {"job": job["id"], "status": "blocked", "reason": "another scheduler owns this output"}
                        results.append(record)
                        report(record)
                        continue
                    status, reason = classify_job(job, args)
                    if status != "pending":
                        lock.close()
                        record = {"job": job["id"], "status": status, "reason": reason}
                        results.append(record)
                        report(record)
                        continue
                    environment = os.environ.copy()
                    environment.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=gpu["uuid"],
                                       PYTHONUNBUFFERED="1", OMP_NUM_THREADS=str(args.cpu_threads),
                                       PYTHONPATH=str(ROOT) + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""))
                    log = (job["output"] / "stdout.log").open("a", buffering=1)
                    log.write(json.dumps({"event": "launch", "timestamp": stamp(), "gpu": gpu,
                                          "command": job["command"], "reason": reason}) + "\n")
                    log.flush()
                    try:
                        process = subprocess.Popen(job["command"], cwd=str(ROOT), env=environment,
                                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    except OSError as error:
                        log.close()
                        lock.close()
                        record = record_status(job, "failed", exit_code=127, reason=str(error), gpu=gpu)
                        results.append(record)
                        report(record)
                        continue
                    active[slot] = {"job": job, "process": process, "log": log, "lock": lock,
                                    "gpu": gpu, "started": time.monotonic()}
                    record = record_status(job, "running", pid=process.pid, gpu=gpu, command=job["command"])
                    report(dict(record, event="launched"))
                    break
            for slot in list(active):
                returncode = active[slot]["process"].poll()
                if returncode is not None:
                    finish(slot, returncode)
            if active:
                time.sleep(2)
    except KeyboardInterrupt:
        report({"event": "interrupted", "reason": "Stopping only subprocess groups launched by this scheduler"})
        for item in active.values():
            if item["process"].poll() is None:
                os.killpg(item["process"].pid, signal.SIGTERM)
        deadline = time.monotonic() + 10
        for slot in list(active):
            process = active[slot]["process"]
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            finish(slot, process.returncode)
        report({"event": "queue_stopped", "not_started": len(pending)})
        return 130
    finally:
        queue_log.close()
    result = {"timestamp": stamp(), "gpus": mapping, "jobs": results,
              "counts": {status: sum(record["status"] == status for record in results)
                         for status in ("completed", "smoke_test", "skipped", "failed", "blocked")}}
    save_json(result, output_root / "queue_summary.json")
    print(json.dumps(result["counts"]), flush=True)
    return 1 if any(record["status"] in ("failed", "blocked") for record in results) else 0


if __name__ == "__main__":
    sys.exit(main())
