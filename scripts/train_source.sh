#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)

usage() {
  cat <<'USAGE'
Usage: bash scripts/train_source.sh DATA_ROOT GPU_INDEX [OUTPUT_ROOT]

Train Office-Home source models A/C/P/R sequentially on one physical GPU.
Uses ImageNet V2, all source images, 20 epochs, and the final checkpoint.
OUTPUT_ROOT defaults to runs/source; relative output paths use the project root.
Lists must exist at data/officehome/{Art,Clipart,Product,Real_World}.txt.
Set PYTHON to override .venv/bin/python (or python3 if .venv is absent).

Example: bash scripts/train_source.sh /path/to/officeHome 6 runs/source
USAGE
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if (( $# < 2 || $# > 3 )); then
  usage >&2
  exit 2
fi

goal_data_root=$1
goal_gpu_index=$2
goal_output_root=${3:-runs/source}
if [[ ! -d "$goal_data_root" ]]; then
  printf 'Data directory does not exist: %s\n' "$goal_data_root" >&2
  exit 2
fi
goal_data_root=$(cd -- "$goal_data_root" && pwd -P)
if [[ ! "$goal_gpu_index" =~ ^[0-9]+$ ]]; then
  printf 'GPU_INDEX must be a physical nvidia-smi GPU number.\n' >&2
  exit 2
fi
if [[ "$goal_output_root" != /* ]]; then
  goal_output_root="$ROOT/$goal_output_root"
fi

cd -- "$ROOT"
if [[ -n "${PYTHON:-}" ]]; then
  goal_python=$PYTHON
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  goal_python="$ROOT/.venv/bin/python"
else
  goal_python=python3
fi
command -v "$goal_python" >/dev/null

for goal_domain in A:Art C:Clipart P:Product R:Real_World; do
  goal_code=${goal_domain%%:*}
  goal_list="$ROOT/data/officehome/${goal_domain#*:}.txt"
  if [[ ! -f "$goal_list" ]]; then
    printf 'Missing image list: %s\n' "$goal_list" >&2
    exit 2
  fi
  for goal_artifact in config.json train.jsonl result.json source_best.pt source_last.pt source_F.pt source_C.pt; do
    if [[ -e "$goal_output_root/$goal_code/$goal_artifact" ]]; then
      printf 'Existing source run: %s; choose a new OUTPUT_ROOT.\n' "$goal_output_root/$goal_code" >&2
      exit 2
    fi
  done
done

goal_gpu_uuid=$(nvidia-smi -i "$goal_gpu_index" --query-gpu=uuid --format=csv,noheader)
if [[ ! "$goal_gpu_uuid" =~ ^GPU-[[:xdigit:]-]+$ ]]; then
  printf 'Could not resolve exactly one GPU UUID for physical GPU %s.\n' "$goal_gpu_index" >&2
  exit 2
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$goal_gpu_uuid"
export TORCH_HOME="$ROOT/cache/torch"

for goal_domain in A:Art C:Clipart P:Product R:Real_World; do
  goal_code=${goal_domain%%:*}
  printf 'Training source %s on physical GPU %s (%s)\n' "$goal_code" "$goal_gpu_index" "$goal_gpu_uuid"
  "$goal_python" -u -m goal.source \
    --train-list "$ROOT/data/officehome/${goal_domain#*:}.txt" \
    --data-root "$goal_data_root" --output "$goal_output_root/$goal_code" \
    --classes 65 --bottleneck-dim 256 --pretrained imagenet_v2 \
    --selection final --val-fraction 0 --epochs 20 --seed 2021 \
    --batch-size 64 --lr 0.01 --lr-schedule constant \
    --momentum 0.9 --weight-decay 0.0005 --smoothing 0.1 \
    --workers 4 --threads 4 --device cuda:0
done
