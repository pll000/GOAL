"""Supervised source training with source-validation or fixed-final selection."""

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import ImageList, eval_transform, read_list, seed_everything, seed_worker, stratified_split, train_transform, write_list
from .models import GoalModel, _torch_load, legacy_feature_state


def label_smoothed_cross_entropy(logits, labels, smoothing=0.1):
    """AaD/SHOT smoothing: epsilon mass is spread across all C classes."""
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(1, labels[:, None]).squeeze(1)
    uniform = -log_probs.mean(dim=1)
    return ((1.0 - smoothing) * nll + smoothing * uniform).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, count, total_loss = 0, 0, 0.0
    per_class_correct = torch.zeros(model.classes, dtype=torch.long)
    per_class_count = torch.zeros(model.classes, dtype=torch.long)
    for batch in loader:
        inputs, labels = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        _, logits = model(inputs)
        predictions = logits.argmax(dim=1)
        matches = predictions.eq(labels)
        correct += matches.sum().item()
        count += labels.numel()
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        labels_cpu = labels.cpu()
        per_class_count += torch.bincount(labels_cpu, minlength=model.classes)
        per_class_correct += torch.bincount(labels_cpu[matches.cpu()], minlength=model.classes)
    if not count:
        raise ValueError("Evaluation loader has no samples")
    present = per_class_count > 0
    return {"accuracy": 100.0 * correct / count, "loss": total_loss / count, "count": count,
            "mean_class_accuracy": (100.0 * per_class_correct[present].float() / per_class_count[present]).mean().item()}


def atomic_save(checkpoint, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, str(temporary))
    os.replace(str(temporary), str(path))


def _split_fingerprint(train_samples, val_samples):
    text = repr((train_samples, val_samples)).encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--val-list", help="Explicit held-out SOURCE validation list")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--selection", choices=("source_validation", "final"), default="source_validation",
                        help="Select by held-out source accuracy or export the fixed final epoch")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--classes", type=int, default=65)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01, help="Bottleneck/classifier LR; backbone uses one tenth")
    parser.add_argument("--lr-schedule", choices=("constant", "poly"), default="constant")
    parser.add_argument("--lr-gamma", type=float, default=10.0)
    parser.add_argument("--lr-power", type=float, default=0.75)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pretrained", choices=("imagenet", "imagenet_v2", "none"), default="imagenet",
                        help="imagenet always means ImageNet1K V1; imagenet_v2 is an explicit alternative")
    parser.add_argument("--bottleneck-dim", type=int, default=256)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA")
    parser.add_argument("--resume", help="Resume our source_last.pt including optimizer and random state")
    parser.add_argument("--log-every", type=int, default=50)
    return parser


def train(args):
    if args.epochs < 1 or args.batch_size < 2:
        raise ValueError("epochs must be positive and batch_size must be at least 2")
    if not 0.0 <= args.smoothing <= 1.0:
        raise ValueError("smoothing must be in [0, 1]")
    if not 0.0 <= args.val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    if args.val_fraction == 0.0 and args.selection != "final":
        raise ValueError("val_fraction=0 is allowed only with --selection final")
    output = Path(args.output)
    artifacts = ("config.json", "train.jsonl", "source_best.pt", "source_last.pt", "source_F.pt", "source_C.pt", "result.json")
    if not args.resume and any((output / name).exists() for name in artifacts):
        raise FileExistsError("Output contains an existing source run; choose a new output or --resume")
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    output.mkdir(parents=True, exist_ok=True)
    full_samples = read_list(args.train_list, args.data_root)
    if args.val_list:
        train_samples, val_samples = full_samples, read_list(args.val_list, args.data_root)
        if {p for p, _ in train_samples} & {p for p, _ in val_samples}:
            raise ValueError("Source training and validation lists overlap")
    elif args.val_fraction > 0:
        train_samples, val_samples = stratified_split(full_samples, args.val_fraction, args.seed)
    else:
        train_samples, val_samples = full_samples, []
    if any(label < 0 or label >= args.classes for _, label in train_samples + val_samples):
        raise ValueError("Source image labels must satisfy 0 <= label < classes")
    if len(train_samples) < 2:
        raise ValueError("Training requires at least two source images")
    evaluation_kind = "source_validation" if val_samples else "source_train_monitoring"
    evaluation_samples = val_samples if val_samples else train_samples
    fingerprint = _split_fingerprint(train_samples, val_samples)
    config = dict(vars(args))
    config.update({"train_count": len(train_samples), "val_count": len(val_samples),
                   "evaluation_count": len(evaluation_samples), "evaluation_kind": evaluation_kind,
                   "split_sha256": fingerprint, "selection": args.selection,
                   "selection_rule": "fixed_final_epoch" if args.selection == "final" else "held_out_source_validation_accuracy",
                   "target_data_used": False, "drop_last": False,
                   "singleton_batch_rule": "skip only a final batch of size 1 because of BatchNorm1d",
                   "backbone_lr_multiplier": 0.1, "architecture": "resnet50_bn{}_weightnorm".format(args.bottleneck_dim),
                   "torch_version": torch.__version__, "source_protocol": "aad_20epochs_constant" if args.epochs == 20 and args.lr_schedule == "constant" else "custom"})
    if args.selection == "final" and not val_samples and args.epochs == 20 and args.lr_schedule == "constant":
        config["source_protocol"] = "aad_20epochs_full_source_final"
    generator = torch.Generator().manual_seed(args.seed)
    common = {"batch_size": args.batch_size, "num_workers": args.workers,
              "pin_memory": device.type == "cuda", "worker_init_fn": seed_worker}
    train_loader = DataLoader(ImageList(samples=train_samples, transform=train_transform()),
                              shuffle=True, generator=generator, drop_last=False, **common)
    val_loader = DataLoader(ImageList(samples=evaluation_samples, transform=eval_transform()), shuffle=False, **common)
    config["image_cache"] = {"train": getattr(train_loader.dataset, "cache_info", {}),
                             "evaluation": getattr(val_loader.dataset, "cache_info", {})}
    model = GoalModel(args.classes, args.bottleneck_dim, "none" if args.resume else args.pretrained).to(device)
    optimizer = torch.optim.SGD(model.parameter_groups(args.lr), momentum=args.momentum,
                                 weight_decay=args.weight_decay, nesterov=True)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    start_epoch, step, best_accuracy, best_epoch = 0, 0, -math.inf, 0
    selected_epoch, selected_accuracy = 0, None
    if args.resume:
        checkpoint = _torch_load(args.resume)
        if checkpoint["config"]["split_sha256"] != fingerprint:
            raise ValueError("Cannot resume with a changed source training/validation split")
        old_selection = checkpoint["config"].get("selection", "source_validation")
        if old_selection == "held_out_source_validation_accuracy":
            old_selection = "source_validation"
        if old_selection != args.selection:
            raise ValueError("Cannot resume with a different source model selection rule")
        for key in ("epochs", "batch_size", "lr", "lr_schedule", "momentum", "weight_decay", "smoothing",
                    "seed", "workers", "classes", "bottleneck_dim", "amp", "pretrained"):
            if checkpoint["config"].get(key) != config[key]:
                raise ValueError("Cannot resume with a changed source setting: " + key)
        if checkpoint["epoch"] >= args.epochs:
            raise ValueError("Completed source runs cannot be resumed")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch, step = checkpoint["epoch"], checkpoint["step"]
        if args.selection == "source_validation":
            best_accuracy, best_epoch = checkpoint["best_accuracy"], checkpoint["best_epoch"]
        selected_epoch = checkpoint.get("selected_epoch", best_epoch)
        selected_accuracy = checkpoint.get("selected_accuracy")
        rng = checkpoint.get("rng")
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            generator.set_state(rng["loader"])
            if device.type == "cuda" and rng["cuda"] is not None:
                torch.cuda.set_rng_state(rng["cuda"], device)
    write_list(train_samples, output / "source_train.txt")
    write_list(evaluation_samples, output / ("source_val.txt" if val_samples else "source_monitoring.txt"))
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    total_steps = args.epochs * (len(train_loader) - int(len(train_samples) % args.batch_size == 1))
    started = time.monotonic()
    with (output / "train.jsonl").open("a" if args.resume else "w", buffering=1) as log:
        def record(value):
            line = json.dumps(value, allow_nan=False)
            log.write(line + "\n")
            print(line, flush=True)

        record({"event": "start", "config": config, "start_epoch": start_epoch})
        for epoch in range(start_epoch, args.epochs):
            model.train()
            epoch_loss, epoch_correct, epoch_count = 0.0, 0, 0
            epoch_start = time.monotonic()
            for batch_index, (images, labels, _) in enumerate(train_loader):
                if images.shape[0] < 2:
                    continue  # BatchNorm1d cannot train on a singleton batch.
                decay = (1.0 + args.lr_gamma * step / max(1, total_steps)) ** (-args.lr_power) if args.lr_schedule == "poly" else 1.0
                for group in optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * decay
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                    _, logits = model(images)
                    loss = label_smoothed_cross_entropy(logits, labels, args.smoothing)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite source loss at step {}".format(step))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                step += 1
                epoch_loss += loss.item() * len(labels)
                epoch_correct += logits.detach().argmax(1).eq(labels).sum().item()
                epoch_count += len(labels)
                if args.log_every and (batch_index + 1) % args.log_every == 0:
                    record({"event": "train", "epoch": epoch + 1, "batch": batch_index + 1, "step": step,
                            "loss": epoch_loss / epoch_count, "accuracy": 100.0 * epoch_correct / epoch_count,
                            "head_lr": optimizer.param_groups[-1]["lr"]})
            evaluation = evaluate(model, val_loader, device)
            if args.selection == "source_validation":
                selected = evaluation["accuracy"] > best_accuracy
                if selected:
                    best_accuracy, best_epoch = evaluation["accuracy"], epoch + 1
            else:
                selected = epoch + 1 == args.epochs
            if selected:
                selected_epoch, selected_accuracy = epoch + 1, evaluation["accuracy"]
            metrics = {"event": "epoch", "epoch": epoch + 1, "step": step,
                       "train_loss": epoch_loss / epoch_count, "train_accuracy": 100.0 * epoch_correct / epoch_count,
                       "train_samples_seen": epoch_count, "skipped_singleton_samples": len(train_samples) - epoch_count,
                       evaluation_kind: evaluation, "evaluation_kind": evaluation_kind,
                       "selection": args.selection, "selected_epoch": selected_epoch,
                       "head_lr": optimizer.param_groups[-1]["lr"], "seconds": time.monotonic() - epoch_start}
            if args.selection == "source_validation":
                metrics.update(validation=evaluation, best_source_accuracy=best_accuracy, best_epoch=best_epoch)
            record(metrics)
            checkpoint = {"model": model.state_dict(), "config": config, "epoch": epoch + 1, "step": step,
                          "metrics": metrics, "selection": args.selection,
                          "selected_epoch": selected_epoch, "selected_accuracy": selected_accuracy,
                          "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                          "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                                  "torch": torch.get_rng_state(), "loader": generator.get_state(),
                                  "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}}
            if args.selection == "source_validation":
                checkpoint.update(best_accuracy=best_accuracy, best_epoch=best_epoch)
            atomic_save(checkpoint, output / "source_last.pt")
            if selected:
                # Serialize immediately; state_dict tensors alias live parameters.
                atomic_save({key: value for key, value in checkpoint.items() if key not in ("optimizer", "scaler", "rng")},
                            output / "source_best.pt")
        selected_checkpoint = _torch_load(output / "source_best.pt")
        model.load_state_dict(selected_checkpoint["model"])
        atomic_save(legacy_feature_state(model), output / "source_F.pt")
        atomic_save(model.classifier.state_dict(), output / "source_C.pt")
        result = {"event": "complete", "selection": args.selection, "selected_epoch": selected_epoch,
                  "selected_source_accuracy": selected_accuracy, "evaluation_kind": evaluation_kind,
                  "epochs": args.epochs, "seconds": time.monotonic() - started,
                  "checkpoint": str((output / "source_best.pt").resolve()), "config": config}
        if args.selection == "source_validation":
            result.update(best_source_accuracy=best_accuracy, best_epoch=best_epoch)
        record(result)
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
