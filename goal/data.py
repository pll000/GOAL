"""Image-list data loading and reproducible Office benchmark preparation."""

import json
import hashlib
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_CACHE_VERSION = 1
DEFAULT_IMAGE_CACHE = Path(__file__).resolve().parents[1] / "cache" / "resized_256"
_CACHE_MANIFESTS = {}
BENCHMARKS = {
    "office-home": {
        "classes": 65,
        "domains": {
            "Art": ("Art", "art"),
            "Clipart": ("Clipart", "clipart"),
            "Product": ("Product", "product"),
            "Real_World": ("Real_World", "RealWorld", "real_world", "realworld"),
        },
        "counts": {"Art": 2427, "Clipart": 4365, "Product": 4439, "Real_World": 4357},
    },
    "office-31": {
        "classes": 31,
        "domains": {"amazon": ("amazon",), "dslr": ("dslr",), "webcam": ("webcam",)},
        "counts": {"amazon": 2817, "dslr": 498, "webcam": 795},
    },
    "imageclef": {
        "classes": 12,
        "domains": {"c": ("c", "caltech"), "i": ("i", "imagenet"), "p": ("p", "pascal")},
        "counts": {"c": 600, "i": 600, "p": 600},
    },
}


def seed_everything(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id):
    """Give each DataLoader worker a different, reproducible augmentation seed."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def train_transform(resize=256, crop=224):
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.RandomCrop(crop),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def eval_transform(resize=256, crop=224):
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def image_cache_resize_spec():
    """Metadata identifying the exact deterministic PIL resize operation."""
    import PIL
    import torchvision

    return {"size": [256, 256], "interpolation": "bilinear", "mode": "RGB",
            "backend": "PIL", "torchvision_version": torchvision.__version__,
            "pillow_version": PIL.__version__}


def image_cache_manifest_path():
    value = os.environ.get("GOAL_IMAGE_CACHE", str(DEFAULT_IMAGE_CACHE))
    if value.lower() in ("off", "none", "false", "0", ""):
        return None
    path = Path(value).expanduser()
    return path if path.suffix.lower() == ".json" else path / "manifest.json"


def _read_image_cache_manifest(path):
    try:
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        previous = _CACHE_MANIFESTS.get(str(path))
        if previous and previous[0] == signature:
            return previous[1], previous[2]
        raw = path.read_bytes()
        manifest = json.loads(raw)
        digest = hashlib.sha256(raw).hexdigest()
        _CACHE_MANIFESTS[str(path)] = (signature, manifest, digest)
        return manifest, digest
    except (OSError, ValueError):
        return None, None


def _without_cacheable_resize(transform):
    """Return the unchanged random suffix only for an exact supported prefix."""
    from torchvision import transforms

    if type(transform) is not transforms.Compose or not transform.transforms:
        return None
    resize = transform.transforms[0]
    if (type(resize) is not transforms.Resize or
            not isinstance(resize.size, (list, tuple)) or tuple(resize.size) != (256, 256) or
            resize.interpolation != transforms.InterpolationMode.BILINEAR or resize.max_size is not None):
        return None
    return transforms.Compose(transform.transforms[1:])


def cache_entry_matches_source(entry, source_stat):
    return (entry.get("original_size") == source_stat.st_size and
            entry.get("original_mtime_ns") == source_stat.st_mtime_ns)


def resolve_image_path(path, data_root="", list_dir=None):
    """Resolve normal lists and AaD lists containing a stale machine prefix.

    Relative paths prefer the explicit data root. For legacy absolute prefixes,
    progressively shorter suffixes are checked under that root. A real absolute
    input path is respected without rebasing.
    """
    path = Path(path).expanduser()
    if path.is_absolute() and path.is_file():
        return path.resolve()
    root = Path(data_root).expanduser() if data_root else None
    candidates = []
    if root is not None:
        if not path.is_absolute():
            candidates.append(root / path)
        parts = [part for part in path.parts if part not in ("/", ".", "..")]
        candidates.extend(root.joinpath(*parts[i:]) for i in range(len(parts) - 1))
    if list_dir is not None and not path.is_absolute():
        candidates.append(Path(list_dir) / path)
    if not path.is_absolute():
        candidates.append(path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("Cannot resolve image {!s} with data_root={!s}".format(path, data_root))


def read_list(list_file, data_root="", validate=True):
    """Read '<image path> <integer label>'; -1 denotes an unlabeled target."""
    list_file = Path(list_file)
    samples = []
    for lineno, line in enumerate(list_file.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            path, label = line.strip().rsplit(None, 1)
            label = int(label)
            if label < -1:
                raise ValueError("label below -1")
        except ValueError as error:
            raise ValueError("{}:{}: expected image path and integer label >= -1".format(list_file, lineno)) from error
        if validate:
            path = str(resolve_image_path(path, data_root, list_file.parent))
        samples.append((str(path), label))
    if not samples:
        raise ValueError("Empty image list: {}".format(list_file))
    if len({path for path, _ in samples}) != len(samples):
        raise ValueError("Duplicate image paths in {}".format(list_file))
    return samples


class ImageList(Dataset):
    """Single view: (image, label, index); two views: (view1, view2, label, index)."""

    def __init__(self, list_file=None, data_root="", transform=None, two_views=False, samples=None,
                 teacher_transform=None):
        self.samples = list(samples) if samples is not None else read_list(list_file, data_root)
        if not self.samples:
            raise ValueError("ImageList must contain at least one image")
        self.imgs = self.samples
        self.transform = transform
        self.two_views = two_views
        self.teacher_transform = teacher_transform if teacher_transform is not None else transform
        self._cached_entries = [None] * len(self.samples)
        self._cached_transform = None
        self._cached_teacher_transform = None
        self.cache_info = {"enabled": False, "eligible_images": 0, "missing_images": len(self.samples),
                           "stale_images": 0, "manifest": None, "compatible_transform": False}
        self._initialize_image_cache()

    def _initialize_image_cache(self):
        path = image_cache_manifest_path()
        if path is None or not path.is_file():
            return
        self.cache_info["manifest"] = str(path.resolve())
        self._cached_transform = _without_cacheable_resize(self.transform)
        self._cached_teacher_transform = _without_cacheable_resize(self.teacher_transform) if self.two_views else None
        compatible = self._cached_transform is not None and (not self.two_views or self._cached_teacher_transform is not None)
        self.cache_info["compatible_transform"] = compatible
        if not compatible:
            return
        manifest, digest = _read_image_cache_manifest(path)
        if not manifest or manifest.get("version") != IMAGE_CACHE_VERSION or manifest.get("resize") != image_cache_resize_spec():
            self.cache_info["reason"] = "invalid manifest or resize/runtime version mismatch"
            return
        self.cache_info.update(manifest_sha256=digest, resize=manifest["resize"])
        missing, stale, eligible = 0, 0, 0
        entries = manifest.get("images", {})
        for index, (source, _) in enumerate(self.samples):
            entry = entries.get(str(Path(source).resolve()))
            if entry is None:
                missing += 1
                continue
            try:
                cached_path = path.parent / entry["cached_path"]
                if not cache_entry_matches_source(entry, Path(source).stat()):
                    stale += 1
                    continue
                cached_stat = cached_path.stat()
                if (cached_stat.st_size != entry["cached_file_size"] or
                        cached_stat.st_mtime_ns != entry["cached_mtime_ns"]):
                    stale += 1
                    continue
            except (OSError, KeyError, TypeError):
                stale += 1
                continue
            self._cached_entries[index] = (cached_path, entry)
            eligible += 1
        self.cache_info.update(enabled=eligible > 0, eligible_images=eligible, missing_images=missing, stale_images=stale)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image, cached = None, self._cached_entries[index]
        if cached is not None:
            cached_path, entry = cached
            try:
                # Source or cache changes after dataset construction also invalidate
                # the entry; never silently train on a stale thumbnail.
                source_stat, cached_stat = Path(path).stat(), cached_path.stat()
                if (cache_entry_matches_source(entry, source_stat) and
                        cached_stat.st_size == entry["cached_file_size"] and
                        cached_stat.st_mtime_ns == entry["cached_mtime_ns"]):
                    with Image.open(cached_path) as handle:
                        candidate = handle.convert("RGB")
                    if candidate.size == (256, 256):
                        image = candidate
            except (OSError, ValueError):
                pass
        use_cache = image is not None
        if image is None:
            with Image.open(path) as handle:
                image = handle.convert("RGB")
        transform = self._cached_transform if use_cache else self.transform
        teacher_transform = self._cached_teacher_transform if use_cache else self.teacher_transform
        student = transform(image.copy()) if transform is not None else image.copy()
        if self.two_views:
            teacher = teacher_transform(image.copy()) if teacher_transform is not None else image.copy()
            return student, teacher, label, index
        return student, label, index


def stratified_split(samples, val_fraction=0.1, seed=2021):
    """Hold out source examples per class, always retaining a training example."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    groups = defaultdict(list)
    for index, (_, label) in enumerate(samples):
        groups[label].append(index)
    rng = np.random.RandomState(seed)
    train_indices, val_indices = [], []
    for label in sorted(groups):
        indices = np.array(groups[label])
        rng.shuffle(indices)
        count = min(len(indices) - 1, max(1, int(round(len(indices) * val_fraction))))
        val_indices.extend(indices[:count].tolist())
        train_indices.extend(indices[count:].tolist())
    if not val_indices:
        raise ValueError("Not enough source examples to make a validation split")
    return [samples[i] for i in sorted(train_indices)], [samples[i] for i in sorted(val_indices)]


def write_list(samples, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join("{} {}\n".format(image, label) for image, label in samples))


def _domain_directory(root, aliases):
    for alias in aliases:
        path = root / alias
        if path.is_dir():
            return path / "images" if (path / "images").is_dir() else path
    raise FileNotFoundError("Could not find domain {} under {}".format(aliases, root))


def build_benchmark_lists(data_root, output_dir, benchmark="office-home", strict_counts=False, class_to_idx=None):
    """Index class directories; ImageCLEF numeric directory names are labels.

    Office benchmarks use alphabetical class names. ImageCLEF's distributed
    directory names 0..11 must be sorted numerically, otherwise classes 10 and
    11 would become labels 2 and 3. When reusing historical checkpoints, an
    explicit class_to_idx mapping takes precedence over this canonical default.
    """
    spec = BENCHMARKS[benchmark]
    root, output = Path(data_root).resolve(), Path(output_dir).resolve()
    domain_dirs = {name: _domain_directory(root, aliases) for name, aliases in spec["domains"].items()}
    classes_by_domain = {}
    for name, domain in domain_dirs.items():
        class_names = [p.name for p in domain.iterdir() if p.is_dir() and not p.name.startswith(".")]
        numeric = benchmark == "imageclef" and all(item.isdecimal() for item in class_names)
        classes_by_domain[name] = sorted(class_names, key=int if numeric else None)
        if numeric and classes_by_domain[name] != [str(i) for i in range(spec["classes"])]:
            raise ValueError("ImageCLEF numeric class directories must be exactly 0..11")
    class_names = next(iter(classes_by_domain.values()))
    if len(class_names) != spec["classes"]:
        raise ValueError("{} expects {} classes; found {}".format(benchmark, spec["classes"], len(class_names)))
    for domain, names in classes_by_domain.items():
        if names != class_names:
            raise ValueError("Class directory names differ for {}".format(domain))
    explicit_mapping = class_to_idx is not None
    if explicit_mapping:
        if not isinstance(class_to_idx, dict):
            raise ValueError("Explicit class map must be a JSON object mapping class names to integers")
        if set(class_to_idx) != set(class_names):
            raise ValueError("Explicit class map must contain exactly every dataset class directory")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in class_to_idx.values()):
            raise ValueError("Explicit class map labels must be integers")
        if sorted(class_to_idx.values()) != list(range(spec["classes"])):
            raise ValueError("Explicit class map must be a bijection onto 0..classes-1")
        class_to_idx = dict(class_to_idx)
    else:
        class_to_idx = {name: i for i, name in enumerate(class_names)}
    output.mkdir(parents=True, exist_ok=True)
    class_order = "numeric" if benchmark == "imageclef" and all(item.isdecimal() for item in class_names) else "alphabetical"
    manifest = {"benchmark": benchmark, "data_root": str(root),
                "classes": sorted(class_names, key=class_to_idx.__getitem__),
                "class_directories": class_names, "class_to_idx": class_to_idx, "domains": {},
                "class_order": class_order,
                "label_order": "explicit_source_checkpoint" if explicit_mapping else class_order}
    for domain, domain_dir in domain_dirs.items():
        samples, counts = [], Counter()
        for class_name in class_names:
            label = class_to_idx[class_name]
            images = sorted(p for p in (domain_dir / class_name).rglob("*")
                            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
            if not images:
                raise ValueError("No images in {}".format(domain_dir / class_name))
            for path in images:
                samples.append((path.relative_to(root).as_posix(), label))
            counts[class_name] = len(images)
        expected = spec["counts"][domain]
        if strict_counts and len(samples) != expected:
            raise ValueError("{} has {} images, expected {}".format(domain, len(samples), expected))
        list_path = output / (domain + ".txt")
        write_list(samples, list_path)
        manifest["domains"][domain] = {"list": str(list_path), "count": len(samples),
                                       "expected_count": expected, "counts_match": len(samples) == expected,
                                       "per_class": dict(counts)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
