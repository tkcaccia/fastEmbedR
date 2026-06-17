#!/usr/bin/env bash
set -u -o pipefail

DATA_ROOT="${DATA_ROOT:-/mnt/sata_ssd/fastEmbedR/Data}"
SCRIPT_DIR="${SCRIPT_DIR:-/mnt/sata_ssd/fastEmbedR/benchmark_scripts}"
OUT_PARENT="${OUT_PARENT:-/mnt/sata_ssd}"
THREADS="${THREADS:-4}"
K="${K:-100}"
NN_BACKEND="${NN_BACKEND:-faiss_gpu_ivf_flat}"
PCA_BACKEND="${PCA_BACKEND:-cuda}"
DATASETS="${DATASETS:-USPS,FashionMNIST,FlowRepository_FR-FCM-ZYRM_files,flow18,MNIST,imagenet,MetRef,mass41,COIL20}"

RUN_ROOT="${OUT_PARENT}/fastEmbedR_PRECOMPUTE_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT/logs"

ENV_DIR="${ENV_DIR:-/home/chiamaka/.fastEmbedR/micromamba/envs/fastembedr-faissgpu-cuvs}"
export CONDA_PREFIX="$ENV_DIR"
export LD_LIBRARY_PATH="$ENV_DIR/lib:$ENV_DIR/targets/x86_64-linux/lib:/usr/local/cuda-13.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$ENV_DIR/lib/libstdc++.so.6:$ENV_DIR/lib/libgcc_s.so.1${LD_PRELOAD:+:$LD_PRELOAD}"

export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export VECLIB_MAXIMUM_THREADS="$THREADS"

{
  echo "run_root=$RUN_ROOT"
  echo "data_root=$DATA_ROOT"
  echo "script_dir=$SCRIPT_DIR"
  echo "threads=$THREADS"
  echo "k=$K"
  echo "nn_backend=$NN_BACKEND"
  echo "pca_backend=$PCA_BACKEND"
  echo "datasets=$DATASETS"
  echo "env_dir=$ENV_DIR"
  echo "started=$(date -Is)"
} > "$RUN_ROOT/run_config.txt"

IFS=',' read -r -a DATASET_ARRAY <<< "$DATASETS"
summary="$RUN_ROOT/precompute_summary.csv"
echo "dataset,status,log" > "$summary"

for dataset in "${DATASET_ARRAY[@]}"; do
  dataset="$(echo "$dataset" | xargs)"
  [ -n "$dataset" ] || continue
  log="$RUN_ROOT/logs/${dataset}.log"
  echo "[$(date -Is)] starting $dataset" | tee -a "$RUN_ROOT/run_status.log"
  Rscript "$SCRIPT_DIR/prepare_centered_benchmark_inputs.R" \
    "--data_root=$DATA_ROOT" \
    "--datasets=$dataset" \
    "--k=$K" \
    "--threads=$THREADS" \
    "--force=TRUE" \
    "--nn_backend=$NN_BACKEND" \
    "--pca_backend=$PCA_BACKEND" \
    > "$log" 2>&1
  code=$?
  if grep -q "FAILED:" "$log" 2>/dev/null; then
    status="failed_inner"
  elif [ "$code" -eq 0 ] && grep -q "status.*,success\\|success" "$DATA_ROOT/precompute_centered_raw_summary.csv" 2>/dev/null; then
    status="success"
  elif [ "$code" -eq 0 ]; then
    status="completed_check_log"
  else
    status="failed_code_${code}"
  fi
  echo "$dataset,$status,$log" >> "$summary"
  echo "[$(date -Is)] finished $dataset status=$status" | tee -a "$RUN_ROOT/run_status.log"
done

echo "completed=$(date -Is)" >> "$RUN_ROOT/run_config.txt"
echo "$RUN_ROOT"
