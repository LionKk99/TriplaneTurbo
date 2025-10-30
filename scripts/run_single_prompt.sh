#!/bin/bash
# Usage:
#   bash scripts/run_single_prompt.sh \
#     --gpu 0 \
#     --prompt "a futuristic glass chair" \
#     --fmt glb \
#     --output /data2/hja/test_output/triplaneturbo/test_chair.glb
#
# Notes:
# - Requires TriplaneTurbo deps installed in the active Python environment.
# - Uses launch.py export flow; creates a temp prompt library with a single prompt.

set -euo pipefail

# resolve repo directory so script can be run from anywhere
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# defaults
GPU="0"
PROMPT=""
FMT="obj"
OUTPUT_PATH="./tmp/triplaneturbo_single.obj"
CONFIG="configs/TriplaneTurbo_v1_acc-2.yaml"
CKPT="pretrained/triplane_turbo_sd_v1.pth"

while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu)
      GPU="$2"; shift 2;;
    --prompt)
      PROMPT="$2"; shift 2;;
    --fmt)
      FMT="$2"; shift 2;;
    --output)
      OUTPUT_PATH="$2"; shift 2;;
    --config)
      CONFIG="$2"; shift 2;;
    --ckpt)
      CKPT="$2"; shift 2;;
    *)
      echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$PROMPT" ]]; then
  echo "--prompt is required"; exit 1
fi

# sanitize fmt
case "$FMT" in
  obj|ply|glb) ;;
  *) echo "--fmt must be one of obj|ply|glb"; exit 1;;
esac

# temp prompt library
TMP_DIR=$(mktemp -d)
LIB_DIR="$TMP_DIR"
LIB_NAME="single_prompt"
LIB_JSON="$LIB_DIR/${LIB_NAME}.json"
mkdir -p "$LIB_DIR"
cat > "$LIB_JSON" <<EOF
{
  "train": ["$PROMPT"],
  "val": ["$PROMPT"],
  "test": ["$PROMPT"]
}
EOF

# optional mirrors (uncomment if needed)
# export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
# export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HOME/.cache/huggingface}

export CUDA_VISIBLE_DEVICES="$GPU"

# run launch export
echo "[run_single_prompt] Using GPUs: $GPU"
# choose python interpreter (can be overridden by env var PYTHON_BIN)
PY_BIN=${PYTHON_BIN:-python}
CMD=($PY_BIN launch.py \
  --config "$CONFIG" \
  --export \
  system.exporter_type="multiprompt-mesh-exporter" \
  system.weights="$CKPT" \
  data.prompt_library="$LIB_NAME" \
  data.prompt_library_dir="$LIB_DIR" \
  system.exporter.fmt="$FMT")

echo "[run_single_prompt] Running (cwd=$REPO_DIR): ${CMD[*]}"
(cd "$REPO_DIR" && "${CMD[@]}")

# find first exported file and copy to OUTPUT_PATH
OUT_ROOT="$REPO_DIR/outputs"
FOUND=$(ls -t $(find "$OUT_ROOT" -type f -path "*/save/*-export/*.$FMT" 2>/dev/null) | head -n 1 || true)
if [[ -z "$FOUND" ]]; then
  echo "[run_single_prompt] Error: no exported *.$FMT found under $OUT_ROOT" >&2
  exit 1
fi
mkdir -p "$(dirname "$OUTPUT_PATH")"
cp -f "$FOUND" "$OUTPUT_PATH"
echo "[run_single_prompt] Copied $FOUND -> $OUTPUT_PATH"

# cleanup
rm -rf "$TMP_DIR"
