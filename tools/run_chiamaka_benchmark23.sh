#!/usr/bin/env bash
set -euo pipefail

# Run only fastEmbedR BENCHMARK #2 and #3 on chiamaka.
# This script intentionally does not run BENCHMARK #1.
#
# Usage on chiamaka:
#   bash /mnt/sata_ssd/fastEmbedR/benchmark_scripts/run_chiamaka_benchmark23.sh
#
# Optional environment variables:
#   DATA_ROOT=/mnt/sata_ssd/fastEmbedR/Data
#   SCRIPT_DIR=/mnt/sata_ssd/fastEmbedR/benchmark_scripts
#   OUT_PARENT=/mnt/sata_ssd
#   THREADS=4
#   TIMEOUT=600
#   PERPLEXITY=auto
#   K=auto
#   KNN_K=auto
#   DATASETS=MNIST,FashionMNIST
#   METHODS2=fastEmbedR_opentsne_cuda,Rtsne_neighbors
#   METHODS3=fastEmbedR_umap_cuda_binary,uwot_umap_fast_sgd

DATA_ROOT="${DATA_ROOT:-/mnt/sata_ssd/fastEmbedR/Data}"
SCRIPT_DIR="${SCRIPT_DIR:-/mnt/sata_ssd/fastEmbedR/benchmark_scripts}"
OUT_PARENT="${OUT_PARENT:-/mnt/sata_ssd}"
THREADS="${THREADS:-4}"
TIMEOUT="${TIMEOUT:-600}"
PERPLEXITY="${PERPLEXITY:-auto}"
K="${K:-auto}"
KNN_K="${KNN_K:-auto}"
DATASETS="${DATASETS:-USPS,FashionMNIST,FlowRepository_FR-FCM-ZYRM_files,flow18,MNIST,imagenet,MetRef,mass41,COIL20}"

RUN_ROOT="${OUT_PARENT}/fastEmbedR_BENCHMARK23_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT"

ENV_DIR="${ENV_DIR:-/home/chiamaka/.fastEmbedR/micromamba/envs/fastembedr-faissgpu-cuvs}"
export CONDA_PREFIX="$ENV_DIR"
export LD_LIBRARY_PATH="$ENV_DIR/lib:$ENV_DIR/targets/x86_64-linux/lib:/usr/local/cuda-13.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$ENV_DIR/lib/libstdc++.so.6:$ENV_DIR/lib/libgcc_s.so.1${LD_PRELOAD:+:$LD_PRELOAD}"

export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export VECLIB_MAXIMUM_THREADS="$THREADS"
export FASTEMBEDR_BENCHMARK_BACKENDS="${FASTEMBEDR_BENCHMARK_BACKENDS:-cpu,cuda}"
export FASTEMBEDR_BENCHMARK_PERPLEXITY="$PERPLEXITY"
export FASTEMBEDR_BENCHMARK_KNN_K="$KNN_K"

{
  echo "run_root=$RUN_ROOT"
  echo "data_root=$DATA_ROOT"
  echo "script_dir=$SCRIPT_DIR"
  echo "threads=$THREADS"
  echo "timeout=$TIMEOUT"
  echo "perplexity=$PERPLEXITY"
  echo "k=$K"
  echo "knn_k=$KNN_K"
  echo "datasets=$DATASETS"
  echo "env_dir=$ENV_DIR"
  echo "started=$(date -Is)"
} > "$RUN_ROOT/run_config.txt"

Rscript -e 'library(faissR); library(fastEmbedR); cat("faissR/fastEmbedR load OK\n")' \
  > "$RUN_ROOT/package_load_check.log" 2>&1

run_benchmark() {
  local name="$1"
  shift
  local log="$RUN_ROOT/${name}.log"
  echo "[$(date -Is)] starting $name" | tee -a "$RUN_ROOT/run_status.log"
  "$@" > "$log" 2>&1
  local status=$?
  echo "[$(date -Is)] finished $name status=$status" | tee -a "$RUN_ROOT/run_status.log"
  return "$status"
}

benchmark2_args=(
  Rscript "$SCRIPT_DIR/benchmark2_tsne_speed_accuracy.R"
  "--data_root=$DATA_ROOT"
  "--out_dir=$RUN_ROOT/BENCHMARK2"
  "--datasets=$DATASETS"
  "--perplexity=$PERPLEXITY"
  "--k=$K"
  "--knn_k=$KNN_K"
  "--threads=$THREADS"
  "--timeout=$TIMEOUT"
)
if [[ -n "${METHODS2:-}" ]]; then
  benchmark2_args+=("--methods=$METHODS2")
fi

benchmark3_args=(
  Rscript "$SCRIPT_DIR/benchmark3_umap_speed_accuracy.R"
  "--data_root=$DATA_ROOT"
  "--out_dir=$RUN_ROOT/BENCHMARK3"
  "--datasets=$DATASETS"
  "--perplexity=$PERPLEXITY"
  "--k=$K"
  "--knn_k=$KNN_K"
  "--threads=$THREADS"
  "--timeout=$TIMEOUT"
)
if [[ -n "${METHODS3:-}" ]]; then
  benchmark3_args+=("--methods=$METHODS3")
fi

run_benchmark benchmark2 "${benchmark2_args[@]}"
run_benchmark benchmark3 "${benchmark3_args[@]}"

echo "completed=$(date -Is)" >> "$RUN_ROOT/run_config.txt"
echo "$RUN_ROOT"
