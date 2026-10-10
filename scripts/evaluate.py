#!/usr/bin/env python3
"""Recompute accuracy from a trusted local target checkpoint, not summary.json.

Examples (from GOAL/):
  .venv/bin/python scripts/evaluate.py --run runs/<target-run> --device cpu
  .venv/bin/python scripts/evaluate.py --checkpoint runs/<target-run>/last.pt \
      --metadata runs/<target-run>/run.json --model teacher --output reevaluation.json
  .venv/bin/python scripts/evaluate.py --run runs/<target-run> --tta ten_crop \
      --batch-size 128 --device cpu --output reevaluation_tta.json

Only load trusted local checkpoints: PyTorch training files contain pickled
optimizer/RNG metadata and are loaded with weights_only=False. ResNet weights
are restored strictly from the selected student/teacher state; no download or
training is performed. Data overrides are recorded explicitly in the JSON.
Optional --tta flip/ten_crop evaluates the fixed student with probability
averaging and a separate inference protocol. --batch-size caps forward crops;
the TTA image batch is reduced so views do not multiply GPU memory use.
flip uses the original center crop and its horizontal flip. ten_crop uses
the four corners and center of the same 256x256 resize, plus their flips.
Each view is 224x224, with equal averaging of softmax probabilities, never
logits. At cap=128, ten_crop loads 12 images per batch (120 crops); caps below
the view count split one image's crops into multiple forwards. The extra cost
is 2x/10x image-view evaluations, not a promised wall-time ratio. TTA output
has a separate kind and must not replace the training center-crop summary.
"""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from goal.data import ImageList, eval_transform
from goal.models import GoalModel
from goal.train import evaluate, sha256
from goal.losses import entropy


PROTECTED_NAMES = {"summary.json", "run.json", "metrics.jsonl", "last.pt", "result.json",
                   "source_best.pt", "source_last.pt", "source_F.pt", "source_C.pt"}
TTA_VIEWS = {"none": 1, "flip": 2, "ten_crop": 10}


class InferenceViews:
    """Stack deterministic normalized crops after the cacheable Resize prefix."""
    def __init__(self, mode):
        if mode not in ("flip", "ten_crop"):
            raise ValueError("multiple views require flip or ten_crop")
        self.mode = mode

    def __call__(self, image):
        from torchvision.transforms import functional as TF
        crops = ([TF.center_crop(image, [224, 224])] if self.mode == "flip"
                 else list(TF.five_crop(image, [224, 224])))
        crops = crops + [TF.hflip(crop) for crop in crops]
        return torch.stack([TF.normalize(TF.to_tensor(crop), [0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225]) for crop in crops])


def inference_transform(mode):
    if mode == "none":
        return eval_transform()
    if mode not in TTA_VIEWS:
        raise ValueError("tta must be none, flip, or ten_crop")
    from torchvision import transforms
    return transforms.Compose([transforms.Resize((256, 256)), InferenceViews(mode)])


@torch.no_grad()
def mean_view_probabilities(model, images, device, forward_crop_cap):
    """Average softmax probabilities, forwarding at most cap crops at a time."""
    if images.ndim != 5 or images.shape[1] < 1 or forward_crop_cap < 1:
        raise ValueError("TTA expects [B,V,C,H,W] and a positive forward crop cap")
    count, views = images.shape[:2]
    crops = images.flatten(0, 1)
    probabilities = []
    for start in range(0, len(crops), forward_crop_cap):
        _, logits = model(crops[start:start + forward_crop_cap].to(device))
        probabilities.append(logits.softmax(-1))
    return torch.cat(probabilities).reshape(count, views, -1).mean(1)


@torch.no_grad()
def evaluate_tta(model, loader, device, forward_crop_cap):
    model.eval()
    correct, count, ent = 0, 0, 0.
    histogram = torch.zeros(model.classifier.fc.out_features, dtype=torch.long)
    for images, labels, _ in loader:
        probs = mean_view_probabilities(model, images, device, forward_crop_cap)
        prediction = probs.argmax(-1).cpu()
        valid = labels >= 0
        correct += int((prediction[valid] == labels[valid]).sum())
        count += int(valid.sum())
        ent += float(entropy(probs).sum())
        histogram += torch.bincount(prediction, minlength=len(histogram))
    return {"accuracy": 100 * correct / count if count else None,
            "entropy": ent / len(loader.dataset), "prediction_histogram": histogram.tolist(),
            "evaluated_labels": count}


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = result.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--run", type=Path, help="target run directory containing last.pt and run.json")
    inputs.add_argument("--checkpoint", type=Path, help="trusted local target checkpoint containing student/teacher")
    result.add_argument("--metadata", type=Path, help="run.json; defaults to the run/checkpoint directory")
    result.add_argument("--model", choices=("student", "teacher"), default="student")
    result.add_argument("--tta", choices=tuple(TTA_VIEWS), default="none",
                        help="Explicit student-only inference protocol: none=original center crop, flip=2 views, ten_crop=10 views")
    result.add_argument("--data-root", help="explicit data-root override; recorded in evaluation provenance")
    result.add_argument("--target-list", type=Path, help="explicit target-list override; may change the evaluation set")
    result.add_argument("--device", default="cpu", help="torch device; default cpu, e.g. cuda:0 when requested")
    result.add_argument("--batch-size", type=int,
                        help="maximum crops per forward; default 2 * training batch size. TTA image batch=max(1,batch_size/views)")
    result.add_argument("--workers", type=int, help="default training metadata worker count")
    result.add_argument("--cpu-threads", type=int, default=4)
    result.add_argument("--output", type=Path, help="optional independent JSON; cannot replace training artifacts")
    return result


def evaluate_checkpoint(args):
    tta = getattr(args, "tta", "none")
    if tta not in TTA_VIEWS:
        raise ValueError("tta must be none, flip, or ten_crop")
    if tta != "none" and args.model != "student":
        raise ValueError("TTA is defined for the fixed student checkpoint; use --model student")
    views = TTA_VIEWS[tta]
    checkpoint_path = (args.run / "last.pt" if args.run is not None else args.checkpoint).resolve()
    metadata_path = (args.metadata or checkpoint_path.parent / "run.json").resolve()
    if args.output:
        output_path = args.output.resolve()
        if args.output.name in PROTECTED_NAMES or output_path.name in PROTECTED_NAMES or output_path in (checkpoint_path, metadata_path):
            raise ValueError("evaluation output cannot overwrite a training summary, metadata, log or checkpoint")
    if not checkpoint_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("both a trusted checkpoint and run metadata JSON must exist")
    metadata = json.loads(metadata_path.read_text())
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise ValueError("run metadata must include a configuration dictionary")
    classes = int(config["classes"])
    bottleneck = int(config.get("bottleneck_dim", config.get("bottleneck", 256)))
    batch_size = args.batch_size if args.batch_size is not None else 2 * int(config["batch_size"])
    workers = args.workers if args.workers is not None else int(config.get("workers", 0))
    if classes < 2 or bottleneck < 1 or batch_size < 1 or workers < 0 or args.cpu_threads < 1:
        raise ValueError("invalid class/bottleneck/batch/worker/thread configuration")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    target_list = Path(args.target_list or metadata["target_list"]).resolve()
    if args.output and output_path == target_list:
        raise ValueError("evaluation output cannot overwrite the input target list")
    data_root = str(Path(args.data_root if args.data_root is not None else metadata.get("data_root", "")).resolve())
    target_hash = sha256(target_list)
    if args.target_list is None and metadata.get("target_list_sha256") not in (None, target_hash):
        raise ValueError("target-list hash differs from training; provide --target-list to explicitly evaluate changed data")

    # Load all checkpoint storage onto CPU even for CUDA evaluation, so unused
    # optimizer and bank state never occupies GPU memory.
    started = time.monotonic()
    # Hash and deserialize the same open inode: an atomic replacement of last.pt
    # by a still-running trainer must not mismatch the reported hash and weights.
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
        checkpoint_file.seek(0)
        state = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    checkpoint_hash = digest.hexdigest()
    if not isinstance(state, dict) or not isinstance(state.get(args.model), dict):
        raise ValueError("checkpoint does not contain a {} state dictionary".format(args.model))
    embedded = state.get("provenance", {})
    if embedded.get("config") != config:
        raise ValueError("checkpoint configuration differs from run metadata")
    for key in ("target_list_sha256", "source_sha256", "source_classifier_sha256", "data_root"):
        if embedded.get(key) != metadata.get(key):
            raise ValueError("checkpoint and run metadata disagree: " + key)
    model = GoalModel(classes, bottleneck_dim=bottleneck, pretrained="none")
    model.load_state_dict(state[args.model], strict=True)
    model.to(device)
    dataset = ImageList(str(target_list), data_root=data_root, transform=inference_transform(tta))
    image_batch_size = batch_size if tta == "none" else max(1, batch_size // views)
    loader = DataLoader(dataset, batch_size=image_batch_size, shuffle=False, num_workers=workers,
                        drop_last=False, pin_memory=device.type == "cuda")
    metrics = (evaluate(model, loader, device) if tta == "none"
               else evaluate_tta(model, loader, device, batch_size))

    expected_steps = int(config["epochs"]) * int(metadata.get("steps_per_epoch", 0))
    final = (expected_steps > 0 and state.get("step") == expected_steps
             and state.get("epoch") == config["epochs"] and not state.get("partial_epoch", False)
             and not config.get("max_steps"))
    overrides = {}
    if args.target_list is not None:
        overrides["target_list"] = {"training": metadata.get("target_list"), "evaluation": str(target_list)}
    if args.data_root is not None:
        overrides["data_root"] = {"training": metadata.get("data_root"), "evaluation": data_root}
    result = {
        "kind": "independent_checkpoint_evaluation" if tta == "none" else "independent_checkpoint_evaluation_tta",
        "inference_protocol": "center_crop" if tta == "none" else tta + "_probability_mean",
        "tta": tta,
        "views": views,
        "aggregation": "single_view_softmax" if tta == "none" else "equal_mean_of_softmax_probabilities",
        "comparable_to_training_summary_protocol": tta == "none",
        "model": args.model,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": state.get("epoch"),
        "checkpoint_step": state.get("step"),
        "checkpoint_is_final": final,
        "metadata": str(metadata_path),
        "metadata_sha256": sha256(metadata_path),
        "preset": metadata.get("preset"),
        "source_domain": metadata.get("source_domain"),
        "target_domain": metadata.get("target_domain"),
        "source_sha256": metadata.get("source_sha256"),
        "classes": classes,
        "bottleneck_dim": bottleneck,
        "target_list": str(target_list),
        "target_list_sha256": target_hash,
        "data_root": data_root,
        "overrides": overrides,
        "images": len(dataset),
        "metrics": metrics,
        "device": str(device),
        "batch_size": batch_size,
        "requested_batch_size": args.batch_size,
        "actual_image_batch_size": image_batch_size,
        "forward_crop_cap": batch_size,
        "inference_cost": {"image_view_evaluations": len(dataset) * views,
                           "relative_to_center_crop_image_views": views,
                           "extra_image_view_evaluations": len(dataset) * (views - 1),
                           "wall_time_ratio": None},
        "workers": workers,
        "cpu_threads": args.cpu_threads,
        "input_cache": getattr(dataset, "cache_info", None),
        "transform": ("goal.data.eval_transform: resize256/center_crop224/ImageNet normalization" if tta == "none"
                      else "resize(256,256)/" + ("center_crop224+horizontal_flip" if tta == "flip"
                            else "four_corners+center_crop224+each_crop_horizontal_flip") + "/ImageNet normalization"),
        "torch": torch.__version__,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "evaluated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
    }
    if args.output:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.name + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        os.replace(temporary, output_path)
    return result


def main():
    argument_parser = parser()
    args = argument_parser.parse_args()
    try:
        result = evaluate_checkpoint(args)
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        argument_parser.exit(1, "evaluation failed: {}\n".format(error))
    print(json.dumps({"model": result["model"], "accuracy": result["metrics"]["accuracy"],
                      "inference_protocol": result["inference_protocol"], "views": result["views"],
                      "transform": result["transform"], "inference_cost": result["inference_cost"],
                      "actual_image_batch_size": result["actual_image_batch_size"],
                      "forward_crop_cap": result["forward_crop_cap"],
                      "evaluated_labels": result["metrics"]["evaluated_labels"],
                      "images": result["images"], "checkpoint_is_final": result["checkpoint_is_final"],
                      "checkpoint_sha256": result["checkpoint_sha256"],
                      "target_list_sha256": result["target_list_sha256"],
                      "overrides": result["overrides"], "output": str(args.output.resolve()) if args.output else None},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
