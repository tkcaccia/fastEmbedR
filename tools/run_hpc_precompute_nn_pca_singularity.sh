#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${BASE_DIR:-/scratch/firenze/NN}"
DATA_ROOT="${DATA_ROOT:-${BASE_DIR}/Data}"
SIF="${SIF:-${BASE_DIR}/singularity/fastembedr_cuda.sif}"
SCRIPT="${SCRIPT:-${BASE_DIR}/hpc_precompute_nn_pca.R}"
K="${K:-100}"
THREADS="${THREADS:-4}"
SEED="${SEED:-4}"
FORCE="${FORCE:-TRUE}"
DATASETS="${DATASETS:-COIL20,USPS,FashionMNIST,FlowRepository_FR-FCM-ZYRM_files,flow18,MNIST,imagenet,MetRef,mass41,TabulaMuris}"
NN_BACKEND="${NN_BACKEND:-faiss_gpu_ivf_flat,cuda_cuvs_cagra,faiss_hnsw,faiss_nndescent,faiss_flat_l2}"
LOG_DIR="${LOG_DIR:-${BASE_DIR}/precompute_logs}"

mkdir -p "${LOG_DIR}"
if [[ ! -f "${SIF}" ]]; then
  echo "Missing Singularity image: ${SIF}" >&2
  exit 1
fi
if [[ ! -f "${SCRIPT}" ]]; then
  echo "Missing R script: ${SCRIPT}" >&2
  echo "Copy tools/hpc_precompute_nn_pca.R to ${SCRIPT} before running." >&2
  exit 1
fi

run_id="$(date +%Y%m%d_%H%M%S)"
log="${LOG_DIR}/precompute_nn_pca_k${K}_${run_id}.log"
config="${LOG_DIR}/precompute_nn_pca_k${K}_${run_id}.config"

cat > "${config}" <<EOF
BASE_DIR=${BASE_DIR}
DATA_ROOT=${DATA_ROOT}
SIF=${SIF}
SCRIPT=${SCRIPT}
K=${K}
THREADS=${THREADS}
SEED=${SEED}
FORCE=${FORCE}
DATASETS=${DATASETS}
NN_BACKEND=${NN_BACKEND}
started=$(date -Is)
EOF

export SINGULARITYENV_R_LIBS_USER="${SINGULARITYENV_R_LIBS_USER:-/opt/conda/lib/R/library}"
export SINGULARITYENV_LD_LIBRARY_PATH="${SINGULARITYENV_LD_LIBRARY_PATH:-/opt/conda/targets/x86_64-linux/lib:/opt/conda/lib}"
export SINGULARITYENV_OMP_NUM_THREADS="${THREADS}"
export SINGULARITYENV_OPENBLAS_NUM_THREADS="${THREADS}"
export SINGULARITYENV_MKL_NUM_THREADS="${THREADS}"

echo "Running precompute job. Log: ${log}"
singularity exec --nv --no-home \
  -B "${BASE_DIR}:${BASE_DIR}" \
  "${SIF}" \
  Rscript "${SCRIPT}" \
    "--base_dir=${BASE_DIR}" \
    "--data_root=${DATA_ROOT}" \
    "--k=${K}" \
    "--threads=${THREADS}" \
    "--seed=${SEED}" \
    "--force=${FORCE}" \
    "--datasets=${DATASETS}" \
    "--nn_backend=${NN_BACKEND}" \
  2>&1 | tee "${log}"

echo "completed=$(date -Is)" >> "${config}"
echo "Done. Summary:"
echo "  ${DATA_ROOT}/centered_raw_precompute_summary_k${K}.csv"
echo "  ${DATA_ROOT}/pca_init_manifest_centered_raw.csv"
echo "  ${DATA_ROOT}/knn_manifest_centered_raw_k${K}.csv"
