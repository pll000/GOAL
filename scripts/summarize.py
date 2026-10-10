#!/usr/bin/env python3
"""Summarize completed SFUDA runs without mixing smoke tests or model choices.

Example: python scripts/summarize.py --runs runs --json results.json --csv results.csv
Run metadata should contain task="A2C" (or source_domain/target_domain), preset,
and config. Ambiguous tasks and duplicate task/seed results are excluded rather
than selecting the best score. Full benchmark means require every task.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


PAPER = {
    "officehome": {
        "tasks": dict(zip(("A2C", "A2P", "A2R", "C2A", "C2P", "C2R", "P2A", "P2C", "P2R", "R2A", "R2C", "R2P"),
                          (59.2, 81.4, 82.7, 70.8, 82.1, 79.9, 69.3, 59.4, 82.8, 73.2, 61.2, 86.8))),
        "mean": 74.1, "table": 1,
        "domains": {"a": "A", "art": "A", "c": "C", "clipart": "C", "p": "P", "product": "P", "r": "R", "realworld": "R"},
    },
    "office31": {
        "tasks": dict(zip(("A2D", "A2W", "D2A", "D2W", "W2A", "W2D"), (95.0, 92.6, 76.8, 98.8, 76.9, 99.8))),
        "mean": 90.0, "table": 2,
        "domains": {"a": "A", "amazon": "A", "d": "D", "dslr": "D", "w": "W", "webcam": "W"},
    },
    "imageclef": {
        "tasks": dict(zip(("C2I", "C2P", "I2C", "I2P", "P2C", "P2I"), (94.3, 77.8, 96.8, 77.9, 96.3, 93.2))),
        "mean": 89.4, "table": 3,
        "domains": {"c": "C", "caltech": "C", "caltech256": "C", "i": "I", "imagenet": "I", "p": "P", "pascal": "P", "pascalvoc": "P"},
    },
}


def compact(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def first(sources, *keys):
    for source in sources:
        for key in keys:
            if source.get(key) is not None:
                return source[key]
    return None


def parse_task(value, dataset):
    if value is None:
        return None
    fields = re.split(r"(?:->|→|_to_|\bto\b)", str(value), flags=re.IGNORECASE)
    if len(fields) == 1:
        fields = str(value).split("2")
    if len(fields) != 2:
        return None
    domains = PAPER[dataset]["domains"]
    source, target = (domains.get(compact(field)) for field in fields)
    task = "{}2{}".format(source, target)
    return task if task in PAPER[dataset]["tasks"] else None


def resolve_task(sources, dataset, run_path):
    domains = PAPER[dataset]["domains"]
    source = first(sources, "source_domain")
    target = first(sources, "target_domain")
    if source is not None and target is not None:
        task = "{}2{}".format(domains.get(compact(source)), domains.get(compact(target)))
        if task not in PAPER[dataset]["tasks"]:
            raise ValueError("invalid explicit source_domain/target_domain")
        return task, "explicit_domains"
    named_task = first(sources, "task", "domain_pair", "pair")
    if named_task is not None:
        task = parse_task(named_task, dataset)
        if task is None:
            raise ValueError("invalid explicit task: {}".format(named_task))
        return task, "explicit_task"
    # Legacy run directories commonly use .../goal_repaired_A2C_seed2021/.
    candidates = {match.upper() for match in re.findall(r"(?<![a-z0-9])([acrdwip]2[acrdwip])(?![a-z0-9])", str(run_path).lower())}
    candidates &= set(PAPER[dataset]["tasks"])
    if len(candidates) == 1:
        return candidates.pop(), "run_directory"
    # Only recognize unambiguous named path components, never numeric domains.
    if source is None:
        source_path = first(sources, "source")
        if source_path:
            source = next((domains[compact(part)] for part in reversed(Path(source_path).parent.parts)
                           if compact(part) in domains), None)
    target_list = first(sources, "target_list")
    if target is None and target_list:
        target_stem = Path(target_list).stem
        target = domains.get(compact(target_stem[:-5] if target_stem.endswith("_list") else target_stem))
    task = parse_task("{}2{}".format(source, target), dataset)
    if task:
        return task, "source_and_list_paths"
    raise ValueError("cannot identify domain pair; record task or source_domain/target_domain")


def preset_name(sources, config, dataset):
    explicit = first(sources, "preset", "config_path", "config_file")
    if explicit:
        return Path(str(explicit)).stem
    if config.get("method") == "aad":
        return dataset + "_aad_control"
    suffix = "literal" if config.get("topology_mask") == "literal" else "repaired"
    return dataset + "_goal_" + suffix


def config_signature(config):
    # Seed and task identities vary within an experiment; all method settings
    # stay in the signature so a tuned learning rate cannot silently be pooled.
    omitted = {"seed", "task", "domain_pair", "pair", "source_domain", "target_domain", "preset"}
    canonical = {key: value for key, value in config.items() if key not in omitted}
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def load_runs(root, task_map=None, dataset_filter=None, preset_filter=None):
    records, excluded = [], []
    for path in sorted(root.rglob("summary.json")):
        try:
            summary = json.loads(path.read_text())
            if summary.get("status") != "completed":
                raise ValueError("status is {}, not completed".format(summary.get("status", "missing")))
            total, completed = summary.get("total_steps"), summary.get("completed_steps")
            if not isinstance(total, int) or total <= 0 or completed != total:
                raise ValueError("incomplete/invalid step counts")
            if summary.get("selection_rule") != "final":
                raise ValueError("selection_rule must be final, never oracle best")
            run_file = path.with_name("run.json")
            run = json.loads(run_file.read_text()) if run_file.exists() else {}
            config = summary.get("config", run.get("config"))
            if not isinstance(config, dict):
                raise ValueError("missing configuration")
            if config.get("max_steps"):
                raise ValueError("max_steps is set (capped/smoke configuration)")
            override = (task_map or {}).get(str(path.parent.relative_to(root)), {})
            sources = [override, summary, summary.get("provenance", {}), run, config]
            if first(sources, "smoke_test", "synthetic") is True or first(sources, "run_kind") in ("smoke", "smoke_test", "synthetic"):
                raise ValueError("explicit smoke/synthetic run")
            dataset = compact(first(sources, "dataset", "benchmark"))
            if dataset not in PAPER:
                raise ValueError("unknown dataset: " + dataset)
            if dataset_filter and dataset != compact(dataset_filter):
                continue
            preset = preset_name(sources, config, dataset)
            if preset_filter and preset != preset_filter:
                continue
            task, task_origin = resolve_task(sources, dataset, path.parent)
            seed = config.get("seed")
            if not isinstance(seed, int):
                raise ValueError("missing/integer seed required")
            metrics = summary.get("metrics", {})
            if metrics.get("step", completed) != completed:
                raise ValueError("metrics are not from the final step")
            if metrics.get("epoch", config.get("epochs")) != config.get("epochs"):
                raise ValueError("metrics are not from the final epoch")
            count = 0
            for model in ("student", "teacher"):
                values = metrics.get(model)
                if values is None:
                    continue
                accuracy = values.get("accuracy")
                if not isinstance(accuracy, (int, float)) or not math.isfinite(accuracy) or not 0 <= accuracy <= 100:
                    excluded.append({"path": str(path), "model": model, "reason": "no valid labeled evaluation accuracy"})
                    continue
                if values.get("evaluated_labels", 1) <= 0:
                    excluded.append({"path": str(path), "model": model, "reason": "no evaluated labels"})
                    continue
                paper = PAPER[dataset]["tasks"][task]
                records.append({"dataset": dataset, "preset": preset, "config_signature": config_signature(config),
                                "task": task, "task_origin": task_origin, "seed": seed, "model": model,
                                "accuracy": float(accuracy), "paper_goal_accuracy": paper, "delta_pp": accuracy-paper,
                                "completed_steps": completed, "summary_path": str(path.resolve())})
                count += 1
            if not count:
                raise ValueError("no valid model results")
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            excluded.append({"path": str(path), "reason": str(error)})
    # Multiple directories for the same seed are not independent repetitions.
    by_identity = defaultdict(list)
    for row in records:
        key = tuple(row[field] for field in ("dataset", "preset", "config_signature", "task", "seed", "model"))
        by_identity[key].append(row)
    unique = []
    for rows in by_identity.values():
        if len(rows) == 1:
            unique.extend(rows)
        else:
            excluded.extend({"path": row["summary_path"], "model": row["model"],
                             "reason": "duplicate task/seed/model/config; no automatic best-run selection"} for row in rows)
    return unique, excluded


def aggregate(records, excluded):
    groups = defaultdict(list)
    for row in records:
        key = tuple(row[field] for field in ("dataset", "preset", "config_signature", "model"))
        groups[key].append(row)
    task_rows, benchmarks = [], []
    for key, rows in sorted(groups.items()):
        dataset, preset, signature, model = key
        identity = dict(dataset=dataset, preset=preset, config_signature=signature, model=model)
        by_task = defaultdict(list)
        by_seed = defaultdict(dict)
        for row in rows:
            by_task[row["task"]].append(row)
            by_seed[row["seed"]][row["task"]] = row["accuracy"]
        expected = PAPER[dataset]["tasks"]
        task_means = {}
        for task, paper in expected.items():
            task_results = by_task.get(task, [])
            scores = [row["accuracy"] for row in task_results]
            mean = statistics.mean(scores) if scores else None
            if scores:
                task_means[task] = mean
            task_rows.append(dict(identity, task=task, seeds=sorted(row["seed"] for row in task_results),
                                  seed_count=len(scores), mean_accuracy=mean,
                                  std_accuracy=statistics.stdev(scores) if len(scores) > 1 else None,
                                  paper_goal_accuracy=paper, delta_pp=mean-paper if scores else None))
        missing = [task for task in expected if task not in by_task]
        complete_seed_means = {str(seed): statistics.mean(values.values()) for seed, values in sorted(by_seed.items())
                               if set(values) == set(expected)}
        macro = statistics.mean(task_means.values()) if not missing else None
        benchmarks.append(dict(identity, completed_tasks=len(task_means), expected_tasks=len(expected),
                               missing_tasks=missing, complete=not missing, macro_accuracy=macro,
                               paper_goal_macro_accuracy=PAPER[dataset]["mean"], paper_table=PAPER[dataset]["table"],
                               delta_pp=macro-PAPER[dataset]["mean"] if macro is not None else None,
                               complete_seed_macro_accuracies=complete_seed_means,
                               seed_coverage={task: sorted(row["seed"] for row in by_task[task]) for task in expected}))
    return {"protocol": "final completed runs only; each model separate; seed mean per task then equal-task macro; missing tasks imply null macro",
            "runs": records, "task_aggregates": task_rows, "benchmarks": benchmarks, "excluded": excluded}


def write_csv(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["row_type", "dataset", "preset", "config_signature", "model", "task", "seed", "accuracy", "std_accuracy",
              "seed_count", "paper_goal_accuracy", "delta_pp", "completed_tasks", "expected_tasks", "summary_path"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        for row in report["runs"]:
            writer.writerow(dict(row, row_type="run"))
        for row in report["task_aggregates"]:
            writer.writerow(dict(row, row_type="task_seed_mean", accuracy=row["mean_accuracy"]))
        for row in report["benchmarks"]:
            writer.writerow(dict(row, row_type="benchmark_macro", accuracy=row["macro_accuracy"],
                                 paper_goal_accuracy=row["paper_goal_macro_accuracy"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path(__file__).resolve().parents[1] / "runs")
    parser.add_argument("--json", type=Path, help="write complete structured report")
    parser.add_argument("--csv", type=Path, help="write run/task/macro rows, missing results are empty")
    parser.add_argument("--task-map", type=Path, help="optional JSON keyed by relative run directory; supplies missing metadata")
    parser.add_argument("--dataset", help="optional dataset filter")
    parser.add_argument("--preset", help="optional exact preset filter")
    args = parser.parse_args()
    if not args.runs.is_dir():
        parser.error("runs directory does not exist: {}".format(args.runs))
    mapping = json.loads(args.task_map.read_text()) if args.task_map else None
    records, excluded = load_runs(args.runs, mapping, args.dataset, args.preset)
    report = aggregate(records, excluded)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if args.csv:
        write_csv(report, args.csv)
    print("Accepted {} model results; excluded {} entries.".format(len(records), len(excluded)))
    for row in report["benchmarks"]:
        accuracy = "{:.2f}%".format(row["macro_accuracy"]) if row["complete"] else "unavailable (missing tasks)"
        print("{dataset} {preset} [{config_signature}] {model}: {completed_tasks}/{expected_tasks} tasks, macro=".format(**row) + accuracy)
    for row in report["task_aggregates"]:
        accuracy = "{:.2f}% ({:+.2f} pp)".format(row["mean_accuracy"], row["delta_pp"]) if row["seed_count"] else "missing"
        print("  {preset} {model} {task}: ".format(**row) + accuracy + " [GOAL {:.1f}%]".format(row["paper_goal_accuracy"]))


if __name__ == "__main__":
    main()
