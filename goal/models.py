"""AaD-compatible ResNet-50 and checkpoint conversion."""

from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import weight_norm


class ResNet_FE(nn.Module):
    """ResNet-50 -> linear 256-dimensional bottleneck -> batch normalization."""

    def __init__(self, bottleneck_dim=256, pretrained="none"):
        super().__init__()
        from torchvision import models

        if pretrained not in ("imagenet", "imagenet_v2", "none"):
            raise ValueError("pretrained must be 'imagenet', 'imagenet_v2', or 'none'")
        if hasattr(models, "ResNet50_Weights"):
            # Preserve the AaD-compatible V1 choice; never use DEFAULT, whose
            # meaning can change with torchvision. V2 is an explicit opt-in.
            if pretrained == "imagenet_v2":
                weights = models.ResNet50_Weights.IMAGENET1K_V2
            else:
                weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained == "imagenet" else None
            resnet = models.resnet50(weights=weights)
        else:
            if pretrained == "imagenet_v2":
                raise ValueError("imagenet_v2 requires torchvision with ResNet50_Weights.IMAGENET1K_V2")
            resnet = models.resnet50(pretrained=pretrained == "imagenet")
        # Preserve the canonical AaD extractor keys without registering its
        # duplicated feature_layers aliases a second time.
        for name in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool"):
            setattr(self, name, getattr(resnet, name))
        self.bottle = nn.Linear(2048, bottleneck_dim)
        self.bn = nn.BatchNorm1d(bottleneck_dim)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.bn(self.bottle(self.avgpool(x).flatten(1)))

    def backbone_parameters(self):
        for name in ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4"):
            yield from getattr(self, name).parameters()


class Classifier(nn.Module):
    def __init__(self, classes, bottleneck_dim=256):
        super().__init__()
        linear = nn.Linear(bottleneck_dim, classes)
        nn.init.xavier_normal_(linear.weight)
        nn.init.zeros_(linear.bias)
        self.fc = weight_norm(linear, name="weight")

    def forward(self, x):
        return self.fc(x)


class GoalModel(nn.Module):
    def __init__(self, classes, bottleneck_dim=256, pretrained="none"):
        super().__init__()
        self.classes = classes
        self.bottleneck_dim = bottleneck_dim
        self.extractor = ResNet_FE(bottleneck_dim, pretrained)
        self.classifier = Classifier(classes, bottleneck_dim)

    def forward(self, x):
        features = self.extractor(x)
        return features, self.classifier(features)

    def parameter_groups(self, lr, backbone_multiplier=0.1):
        return [
            {"params": list(self.extractor.backbone_parameters()), "lr": lr * backbone_multiplier,
             "initial_lr": lr * backbone_multiplier},
            {"params": list(self.extractor.bottle.parameters()) + list(self.extractor.bn.parameters()),
             "lr": lr, "initial_lr": lr},
            {"params": list(self.classifier.parameters()), "lr": lr, "initial_lr": lr},
        ]


def _torch_load(path, map_location="cpu"):
    # Explicit weights_only=False permits our locally produced training metadata
    # on modern PyTorch; older PyTorch has no weights_only argument.
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def _unwrap_state(value):
    if isinstance(value, dict):
        for key in ("state_dict", "model"):
            if key in value and isinstance(value[key], dict):
                return value[key]
    return value


def _canonical_feature_state(state):
    names = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool")
    result = {}
    for key, tensor in _unwrap_state(state).items():
        while key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("extractor."):
            key = key[len("extractor."):]
        if key.startswith("feature_layers."):
            fields = key.split(".", 2)
            if len(fields) != 3 or not fields[1].isdigit() or int(fields[1]) >= len(names):
                raise ValueError("Unsupported legacy feature key: {}".format(key))
            key = names[int(fields[1])] + "." + fields[2]
        if key in result and not torch.equal(result[key], tensor):
            raise ValueError("Conflicting duplicated AaD extractor weights: {}".format(key))
        result[key] = tensor
    return result


def load_checkpoint(model, path, classifier_path=None, strict=True):
    """Load a structured checkpoint, AaD F/C, or SHOT F/B/C weights.

    A directory resolves to source_best.pt, then source_F.pt/source_C.pt. Class
    dimensions and all parameter names are checked strictly by default. A
    source_B.pt beside source_F.pt identifies SHOT's separate BN bottleneck.
    """
    path = Path(path)
    if path.is_dir():
        if (path / "source_best.pt").is_file():
            path = path / "source_best.pt"
        else:
            classifier_path = path / "source_C.pt" if classifier_path is None else classifier_path
            path = path / "source_F.pt"
    checkpoint = _torch_load(path)
    state = _unwrap_state(checkpoint)
    if any(key.lstrip().replace("module.", "").startswith("extractor.") for key in state):
        canonical = {}
        extractor = {}
        for key, tensor in state.items():
            while key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("extractor."):
                extractor[key[len("extractor."):]] = tensor
            else:
                canonical[key] = tensor
        canonical.update({"extractor." + key: value for key, value in _canonical_feature_state(extractor).items()})
        model.load_state_dict(canonical, strict=strict)
        # Training checkpoints also contain tensor optimizer/scaler/RNG state;
        # provenance must remain JSON-serializable when source_last.pt is used.
        metadata = {key: checkpoint[key] for key in
                    ("config", "epoch", "step", "metrics", "best_accuracy", "best_epoch")
                    if key in checkpoint}
        metadata.update(origin=str(path), checkpoint_format="structured")
        return metadata
    if classifier_path is None:
        classifier_path = path.with_name(path.name.replace("_F", "_C"))
        if classifier_path == path:
            raise ValueError("Legacy extractor checkpoint needs classifier_path")
    extractor = _canonical_feature_state(state)
    bottleneck_path = path.with_name(path.name.replace("_F", "_B"))
    has_bottleneck = bottleneck_path != path and bottleneck_path.is_file()
    if has_bottleneck:
        bottleneck = _unwrap_state(_torch_load(bottleneck_path))
        for key, tensor in bottleneck.items():
            while key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("bottleneck."):
                key = "bottle." + key[len("bottleneck."):]
            if key in extractor and not torch.equal(extractor[key], tensor):
                raise ValueError("Conflicting backbone/bottleneck parameters: {}".format(key))
            extractor[key] = tensor
    model.extractor.load_state_dict(extractor, strict=strict)
    classifier = _unwrap_state(_torch_load(classifier_path))
    cleaned = {}
    for key, tensor in classifier.items():
        while key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("classifier."):
            key = key[len("classifier."):]
        cleaned[key] = tensor
    model.classifier.load_state_dict(cleaned, strict=strict)
    metadata = {"legacy_extractor": str(path), "legacy_classifier": str(classifier_path),
                "checkpoint_format": "shot_f_b_c" if has_bottleneck else "aad_f_c"}
    if has_bottleneck:
        metadata["legacy_bottleneck"] = str(bottleneck_path)
    return metadata


def legacy_feature_state(model):
    """Export original AaD aliases so its unmodified loader also accepts weights."""
    state = dict(model.extractor.state_dict())
    names = {name: str(i) for i, name in enumerate(
        ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool"))}
    for key, value in list(state.items()):
        stem, separator, suffix = key.partition(".")
        if stem in names and separator:
            state["feature_layers." + names[stem] + "." + suffix] = value
    return state
