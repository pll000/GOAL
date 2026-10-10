"""Source-free adaptation. Target labels are used only by evaluate()."""
import argparse
import hashlib
import json
import math
from numbers import Integral, Real
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ImageList, train_transform, eval_transform, seed_everything
from .models import GoalModel, load_checkpoint
from .losses import attraction, dispersion, entropy, neighbor_weights, topology_loss, neighbor_cross_entropy
from .memory import TargetMemory, update_teacher


def sha256(path):
    h = hashlib.sha256()
    if Path(path).is_dir():
        for file in sorted(Path(path).glob("source_*.pt")):
            h.update(file.name.encode())
            h.update(sha256(file).encode())
        return h.hexdigest()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_save(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, count, ent = 0, 0, 0.
    histogram = torch.zeros(model.classifier.fc.out_features, dtype=torch.long)
    for images, labels, _ in loader:
        _, logits = model(images.to(device))
        probs = logits.softmax(-1)
        prediction = probs.argmax(-1).cpu()
        # Unknown labels (-1) allow real unlabeled deployment without fake accuracy.
        valid = labels >= 0
        correct += int((prediction[valid] == labels[valid]).sum())
        count += int(valid.sum())
        ent += float(entropy(probs).sum())
        histogram += torch.bincount(prediction, minlength=len(histogram))
    return {"accuracy": 100*correct/count if count else None,
            "entropy": ent/len(loader.dataset), "prediction_histogram": histogram.tolist(),
            "evaluated_labels": count}


@torch.no_grad()
def reweight_reciprocal_neighbors(weights, factors, self_mask):
    """Preserve self exactly and redistribute only the non-self weight mass.

    All-zero original non-self rows remain zero. If a zero floor removes all
    non-self mass, fall back to the original neighbors instead of dividing by
    zero. Full-row renormalization is deliberately avoided: it would amplify
    self-predictions when most external neighbors are not reciprocal.
    """
    if weights.ndim != 2 or factors.shape != weights.shape or self_mask.shape != weights.shape:
        raise ValueError("reciprocal weights, factors, and self mask must match [B,K]")
    original = weights.detach()
    factors = factors.detach().to(original)
    self_mask = self_mask.detach().to(device=original.device, dtype=torch.bool)
    neighbors = original.masked_fill(self_mask, 0.)
    weighted = neighbors * factors
    old_sum = neighbors.sum(-1, keepdim=True)
    new_sum = weighted.sum(-1, keepdim=True)
    denominator = torch.where(new_sum != 0, new_sum, torch.ones_like(new_sum))
    normalized = weighted * (old_sum / denominator)
    normalized = torch.where(new_sum != 0, normalized, neighbors)
    normalized = torch.where(old_sum != 0, normalized, torch.zeros_like(normalized))
    return torch.where(self_mask, original, normalized)


def run(args):
    config = json.loads(Path(args.config).read_text())
    for key in ("epochs", "batch_size", "workers", "seed", "lr", "max_steps"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if config["epochs"] < 1 or config["batch_size"] < 2 or config["k"] < 1:
        raise ValueError("epochs/k must be positive and batch_size must be at least two")
    if config["workers"] < 0 or config["lr"] <= 0 or config["classes"] < 2:
        raise ValueError("invalid workers, learning rate, or class count")
    if config.get("max_steps") is not None and config["max_steps"] < 1:
        raise ValueError("max_steps must be positive")
    rho_source = config.get("rho_source", "prediction_ema")
    if rho_source not in ("prediction_ema", "teacher"):
        raise ValueError("rho_source must be 'prediction_ema' or 'teacher'")
    if rho_source == "teacher" and config["method"] == "aad":
        raise ValueError("rho_source='teacher' requires the GOAL teacher network")
    reciprocal_k = config.get("reciprocal_k", 0)
    reciprocal_floor = config.get("reciprocal_floor", 0.1)
    neighbor_ce_weight = config.get("neighbor_ce_weight", 0.)
    if isinstance(reciprocal_k, bool) or not isinstance(reciprocal_k, Integral) or reciprocal_k < 0:
        raise ValueError("reciprocal_k must be a nonnegative integer")
    if (isinstance(reciprocal_floor, bool) or not isinstance(reciprocal_floor, Real)
            or not math.isfinite(reciprocal_floor) or not 0 <= reciprocal_floor <= 1):
        raise ValueError("reciprocal_floor must be finite and in [0, 1]")
    if reciprocal_k and config["method"] == "aad":
        raise ValueError("reciprocal reweighting requires GOAL neighbor reliability weights")
    if (isinstance(neighbor_ce_weight, bool) or not isinstance(neighbor_ce_weight, Real)
            or not math.isfinite(neighbor_ce_weight) or neighbor_ce_weight < 0):
        raise ValueError("neighbor_ce_weight must be finite and nonnegative")
    if (reciprocal_k or neighbor_ce_weight) and config["entropy_mode"] == "cross_entropy":
        raise ValueError("reciprocal/neighbor CE options require nonnegative shannon or cross_entropy_clamped reliability")
    device = torch.device(args.device)
    seed_everything(config["seed"])
    torch.set_num_threads(args.cpu_threads)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.jsonl").exists() and not args.resume:
        raise FileExistsError("output already contains a run; choose another output or --resume")
    classes = config["classes"]
    dual = config["method"] != "aad"
    train_set = ImageList(args.target_list, data_root=args.data_root,
                          transform=train_transform(), two_views=dual)
    eval_set = ImageList(args.target_list, data_root=args.data_root, transform=eval_transform())
    if len(train_set) < 2:
        raise ValueError("adaptation needs at least two images")
    if len(train_set) < config["batch_size"]:
        raise ValueError("dataset smaller than batch_size; explicitly lower batch_size")
    generator = torch.Generator().manual_seed(config["seed"])
    loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True,
                        num_workers=config["workers"], drop_last=True,
                        pin_memory=device.type == "cuda", generator=generator)
    eval_loader = DataLoader(eval_set, batch_size=config["batch_size"]*2, shuffle=False,
                             num_workers=config["workers"], pin_memory=device.type == "cuda")
    student = GoalModel(classes, pretrained="none").to(device)
    source_metadata = load_checkpoint(student, args.source, args.source_classifier)
    teacher = GoalModel(classes, pretrained="none").to(device) if dual else None
    if teacher is not None:
        teacher.load_state_dict(student.state_dict())
        teacher.requires_grad_(False)
        teacher.eval()
    optimizer = torch.optim.SGD(student.parameter_groups(config["lr"], config.get("backbone_multiplier", .1)),
                                momentum=.9, weight_decay=config["weight_decay"], nesterov=True)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    bank = TargetMemory(len(train_set), 256, classes, device)
    step, start_epoch = 0, 0
    total_steps = config["epochs"] * len(loader)
    provenance = {"config": config, "preset": Path(args.config).stem,
                  "rho_source": rho_source,
                  "source_domain": args.source_domain, "target_domain": args.target_domain,
                  "data_root": str(Path(args.data_root).resolve()),
                  "input_cache": getattr(train_set, "cache_info", None),
                  "eval_every": args.eval_every, "source_metadata": source_metadata,
                  "target_list": str(Path(args.target_list).resolve()),
                  "target_list_sha256": sha256(args.target_list),
                  "source": str(Path(args.source).resolve()), "source_sha256": sha256(args.source),
                  "source_classifier": args.source_classifier,
                  "target_images": len(train_set), "steps_per_epoch": len(loader),
                  "device": str(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "torch": torch.__version__, "source_free": True,
                  "selection_rule": "final epoch, never target-accuracy checkpoint selection",
                  "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if args.source_classifier:
        provenance["source_classifier_sha256"] = sha256(args.source_classifier)
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if state["step"] >= total_steps or config.get("max_steps"):
            raise ValueError("completed runs and smoke tests cannot be resumed")
        if state.get("partial_epoch", False):
            raise ValueError("smoke checkpoint ended inside an epoch and cannot be resumed")
        previous_config = dict(state["provenance"]["config"])
        requested_config = dict(config)
        previous_config.setdefault("rho_source", "prediction_ema")
        requested_config.setdefault("rho_source", "prediction_ema")
        for key, default in (("reciprocal_k", 0), ("reciprocal_floor", .1), ("neighbor_ce_weight", 0.)):
            previous_config.setdefault(key, default)
            requested_config.setdefault(key, default)
        if previous_config != requested_config:
            raise ValueError("resume configuration differs from checkpoint")
        for key in ("target_list_sha256", "source_sha256", "source_classifier_sha256", "data_root", "eval_every"):
            if state["provenance"].get(key) != provenance.get(key):
                raise ValueError("resume provenance differs: " + key)
        student.load_state_dict(state["student"])
        if teacher is not None:
            teacher.load_state_dict(state["teacher"])
        optimizer.load_state_dict(state["optimizer"])
        bank.load_state_dict(state["memory"], rho_source=rho_source)
        step, start_epoch = state["step"], state["epoch"]
        generator.set_state(state["loader_rng"].cpu())
        torch.set_rng_state(state["torch_rng"].cpu())
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
        provenance = state["provenance"]
        provenance["rho_source"] = rho_source
        provenance.setdefault("resume_events", []).append({
            "checkpoint": str(Path(args.resume).resolve()), "epoch": start_epoch,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "input_cache": getattr(train_set, "cache_info", None)})
        metrics_path = output / "metrics.jsonl"
        if metrics_path.exists():
            previous_lines = metrics_path.read_text().splitlines()
            committed = [line for line in previous_lines if json.loads(line)["step"] <= step]
            if len(committed) != len(previous_lines):
                backup = output / ("metrics_before_resume_" + str(time.time_ns()) + ".jsonl")
                backup.write_text("\n".join(previous_lines) + "\n")
                metrics_path.write_text("\n".join(committed) + "\n")
                provenance["resume_events"][-1]["superseded_metrics"] = str(backup)
        (output / "run.json").write_text(json.dumps(provenance, indent=2))
    else:
        student.eval()
        memory_model = teacher if dual else student
        with torch.no_grad():
            for images, _, indices in eval_loader:
                features, logits = memory_model(images.to(device))
                bank.update(indices, features, logits.softmax(-1), momentum=0)
        (output / "run.json").write_text(json.dumps(provenance, indent=2))
    started = time.monotonic()

    def report(record):
        record.update(step=step, elapsed_seconds=round(time.monotonic()-started, 2))
        with open(output / "metrics.jsonl", "a") as log:
            log.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    if start_epoch == 0:
        report({"event": "source_only", "student": evaluate(student, eval_loader, device)})
    max_steps = config.get("max_steps")
    final_metrics = None
    for epoch in range(start_epoch, config["epochs"]):
        student.train()
        totals = {"loss": 0., "attraction": 0., "dispersion": 0., "topology": 0., "edges": 0., "neighbor_weight": 0.}
        if reciprocal_k:
            totals.update(reciprocal_fraction=0., no_reciprocal_fraction=0.)
        if neighbor_ce_weight:
            totals["neighbor_ce"] = 0.
        batches = 0
        # Once per epoch, using either q's EMA entropy (legacy default) or the
        # most recently observed instantaneous teacher distribution per sample.
        rho = bank.entropy_threshold(rho_source=rho_source)
        for batch in loader:
            if dual:
                images, teacher_images, _, indices = batch
            else:
                images, _, indices = batch
            images = images.to(device, non_blocking=True)
            features, logits = student(images)
            probs = logits.softmax(-1)
            with torch.no_grad():
                if dual:
                    teacher_features, teacher_logits = teacher(teacher_images.to(device, non_blocking=True))
                    teacher_probs = teacher_logits.softmax(-1)
                    bank.update(indices, teacher_features, teacher_probs, config["prediction_momentum"])
                    anchor = teacher_features
                else:
                    bank.update(indices, features, probs, momentum=0)
                    anchor = features
                near_indices, near_features, near_probs = bank.neighbors(indices, anchor, config["k"], include_self=dual)
                weights = neighbor_weights(anchor, near_features, near_probs, probs,
                                           config["top_dims"], config["entropy_mode"]) if dual else None
                if reciprocal_k:
                    factors, reciprocal = bank.reciprocal_weights(
                        indices, near_indices, reciprocal_k, reciprocal_floor, return_mask=True)
                    self_mask = near_indices == indices.to(near_indices.device)[:, None]
                    weights = reweight_reciprocal_neighbors(weights, factors, self_mask)
                    external = ~self_mask
                    totals["reciprocal_fraction"] += float((reciprocal & external).sum() / external.sum().clamp_min(1))
                    totals["no_reciprocal_fraction"] += float((~(reciprocal & external).any(-1)).float().mean())
            attract = attraction(probs, near_probs, weights)
            repel = dispersion(probs, config["dispersion_reduction"])
            topo, topology_info = (features.sum()*0, {"edges": 0})
            if dual and config["gamma"]:
                topo, topology_info = topology_loss(features, probs, teacher_probs, rho,
                                                    config["normalize_topology"], config["topology_mask"],
                                                    config["detach_reliability"],
                                                    config.get("topology_reduction", "valid_edges"))
            decay = (1 + 10*step/total_steps) ** (-config["beta"])
            loss = attract + config["lambda_scale"] * decay * repel + config["gamma"] * topo
            if neighbor_ce_weight:
                neighbor_ce = neighbor_cross_entropy(logits, near_probs, weights)
                loss = loss + neighbor_ce_weight * neighbor_ce
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss at step " + str(step))
            if config["lr_schedule"] == "poly":
                for group in optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * (1 + 10*step/total_steps)**(-.75)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.check_gradients:
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in student.parameters()):
                    raise FloatingPointError("non-finite gradient")
            optimizer.step()
            if dual:
                update_teacher(teacher, student, config["teacher_momentum"])
            step += 1
            batches += 1
            for key, value in (("loss", loss), ("attraction", attract), ("dispersion", repel), ("topology", topo)):
                totals[key] += float(value.detach())
            totals["edges"] += topology_info["edges"]
            totals["neighbor_weight"] += float(weights.mean()) if weights is not None else 1.
            if neighbor_ce_weight:
                totals["neighbor_ce"] += float(neighbor_ce.detach())
            if max_steps and step >= max_steps:
                break
        record = {"event": "epoch", "epoch": epoch+1,
                  "losses": {key: value/max(1, batches) for key, value in totals.items()},
                  "rho": float(rho), "rho_source": rho_source, "lambda": config["lambda_scale"]*decay,
                  "lr": [g["lr"] for g in optimizer.param_groups]}
        final = epoch+1 == config["epochs"] or bool(max_steps and step >= max_steps)
        if (epoch+1) % args.eval_every == 0 or final:
            record["student"] = evaluate(student, eval_loader, device)
            if dual:
                record["teacher"] = evaluate(teacher, eval_loader, device)
            final_metrics = record
        report(record)
        # Epoch-boundary resume restores RNG as well as model, optimizer and banks.
        if (epoch+1) % args.save_every == 0 or final:
            atomic_save({"student": student.state_dict(),
                         "teacher": teacher.state_dict() if dual else None,
                         "optimizer": optimizer.state_dict(), "memory": bank.state_dict(),
                         "epoch": epoch+1, "step": step, "provenance": provenance,
                         "partial_epoch": batches < len(loader),
                         "loader_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
                         "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                         "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []},
                        output / "last.pt")
        if final:
            break
    if final_metrics is not None:
        summary = {"status": "smoke_test" if max_steps and step < total_steps else "completed",
                   "selection_rule": "final", "config": config, "metrics": final_metrics,
                   "rho_source": rho_source,
                   "preset": provenance["preset"], "source_domain": args.source_domain,
                   "target_domain": args.target_domain,
                   "total_steps": total_steps, "completed_steps": step}
        (output / "summary.json").write_text(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--target-list", required=True)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-classifier")
    parser.add_argument("--source-domain")
    parser.add_argument("--target-domain")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--check-gradients", action="store_true")
    args = parser.parse_args()
    if args.eval_every < 1 or args.save_every < 1:
        parser.error("evaluation/save interval must be positive")
    run(args)


if __name__ == "__main__":
    main()
