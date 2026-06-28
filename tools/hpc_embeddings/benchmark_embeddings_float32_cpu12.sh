#!/usr/bin/env bash

#SBATCH --account=immunology
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --job-name="fastEmbedR_emb_CPU12"
#SBATCH --chdir=/scratch/firenze/NN
#SBATCH --output=/scratch/firenze/NN/benchmark_logs/fastEmbedR_emb_cpu12_%j.out
#SBATCH --error=/scratch/firenze/NN/benchmark_logs/fastEmbedR_emb_cpu12_%j.err

set -euo pipefail

# CPU-only embedding benchmark for publication.
#
# fastEmbedR methods use the *_float32.RData files.
# Reference methods use the standard dataset .RData files.
#
# Submit on the HPC with:
#   sbatch /scratch/firenze/NN/benchmark_embeddings_float32_cpu12.sh

export BASE_DIR="${BASE_DIR:-/scratch/firenze/NN}"
export DATA_ROOT="${DATA_ROOT:-${BASE_DIR}/Data}"
export SCRIPT_DIR="${SCRIPT_DIR:-${BASE_DIR}}"
export LOG_DIR="${LOG_DIR:-${BASE_DIR}/benchmark_logs}"
export OUT_DIR="${OUT_DIR:-${BASE_DIR}/benchmark_embeddings_float32_CPU12_$(date +%Y%m%d_%H%M%S)}"
export SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-${BASE_DIR}/singularity/fastembedr_cuda.sif}"
export THREADS="${THREADS:-12}"
export TIMEOUT="${TIMEOUT:-10800}"
export K="${K:-30}"
export PERPLEXITY="${PERPLEXITY:-15}"
export SEED="${SEED:-4}"

export DATASETS="${DATASETS:-COIL20,USPS,FashionMNIST,FlowRepository_FR-FCM-ZYRM_files,flow18,MNIST,imagenet,MetRef,mass41,TabulaMuris}"
export METHODS="${METHODS:-fastEmbedR_opentsne_cpu,fastEmbedR_umap_cpu_fuzzy,fastEmbedR_umap_cpu_binary,Rtsne_full,KlugerLab_FItSNE,umap_package,uwot_default,uwot_fast_sgd}"

export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export VECLIB_MAXIMUM_THREADS="${THREADS}"
export RCPP_PARALLEL_NUM_THREADS="${THREADS}"
export APPTAINERENV_OMP_NUM_THREADS="${THREADS}"
export APPTAINERENV_OPENBLAS_NUM_THREADS="${THREADS}"
export APPTAINERENV_MKL_NUM_THREADS="${THREADS}"
export APPTAINERENV_RCPP_PARALLEL_NUM_THREADS="${THREADS}"
export SINGULARITYENV_OMP_NUM_THREADS="${THREADS}"
export SINGULARITYENV_OPENBLAS_NUM_THREADS="${THREADS}"
export SINGULARITYENV_MKL_NUM_THREADS="${THREADS}"
export SINGULARITYENV_RCPP_PARALLEL_NUM_THREADS="${THREADS}"

mkdir -p "${LOG_DIR}" "${OUT_DIR}"
cd "${BASE_DIR}"

if [[ -f "${SCRIPT_DIR}/benchmark_embeddings_float32_publication.R" ]]; then
  BENCH_R="${SCRIPT_DIR}/benchmark_embeddings_float32_publication.R"
elif [[ -f "${BASE_DIR}/benchmark_embeddings_float32_publication.R" ]]; then
  BENCH_R="${BASE_DIR}/benchmark_embeddings_float32_publication.R"
else
  echo "Cannot find benchmark_embeddings_float32_publication.R in ${SCRIPT_DIR} or ${BASE_DIR}" >&2
  exit 1
fi

RUNNER=()
if [[ -n "${SINGULARITY_IMAGE}" && -f "${SINGULARITY_IMAGE}" ]]; then
  CONTAINER_BIN="${CONTAINER_BIN:-$(command -v apptainer || command -v singularity || true)}"
  if [[ -z "${CONTAINER_BIN}" ]]; then
    echo "Cannot find apptainer or singularity although SINGULARITY_IMAGE is set." >&2
    exit 1
  fi
  RUNNER=("${CONTAINER_BIN}" exec --bind "${BASE_DIR}:${BASE_DIR}" --pwd "${BASE_DIR}" "${SINGULARITY_IMAGE}")
fi

{
  echo "Starting CPU embedding benchmark"
  echo "BASE_DIR=${BASE_DIR}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "THREADS=${THREADS}"
  echo "TIMEOUT=${TIMEOUT}"
  echo "DATASETS=${DATASETS}"
  echo "METHODS=${METHODS}"
  "${RUNNER[@]}" Rscript "${BENCH_R}" \
    --script="${BENCH_R}" \
    --backend_group=cpu \
    --base_dir="${BASE_DIR}" \
    --data_root="${DATA_ROOT}" \
    --out_dir="${OUT_DIR}" \
    --datasets="${DATASETS}" \
    --methods="${METHODS}" \
    --threads="${THREADS}" \
    --timeout="${TIMEOUT}" \
    --k="${K}" \
    --perplexity="${PERPLEXITY}" \
    --seed="${SEED}"
  echo "DONE: ${OUT_DIR}"
} 2>&1 | tee -a "${OUT_DIR}/benchmark_cpu12.log"
