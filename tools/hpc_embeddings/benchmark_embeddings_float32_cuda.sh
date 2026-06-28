#!/usr/bin/env bash

#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=48:00:00
#SBATCH --job-name="fastEmbedR_emb_CUDA"
#SBATCH --chdir=/scratch/firenze/NN
#SBATCH --output=/scratch/firenze/NN/benchmark_logs/fastEmbedR_emb_cuda_%j.out
#SBATCH --error=/scratch/firenze/NN/benchmark_logs/fastEmbedR_emb_cuda_%j.err

set -euo pipefail

# CUDA-only embedding benchmark for publication.
#
# Only native fastEmbedR CUDA methods are run here. Reference R packages are
# deliberately kept in the CPU script so they are not reported as GPU work.
#
# Submit on the HPC with:
#   sbatch /scratch/firenze/NN/benchmark_embeddings_float32_cuda.sh

export BASE_DIR="${BASE_DIR:-/scratch/firenze/NN}"
export DATA_ROOT="${DATA_ROOT:-${BASE_DIR}/Data}"
export SCRIPT_DIR="${SCRIPT_DIR:-${BASE_DIR}}"
export LOG_DIR="${LOG_DIR:-${BASE_DIR}/benchmark_logs}"
export OUT_DIR="${OUT_DIR:-${BASE_DIR}/benchmark_embeddings_float32_CUDA_$(date +%Y%m%d_%H%M%S)}"
export SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-${BASE_DIR}/singularity/fastembedr_cuda.sif}"
export THREADS="${THREADS:-12}"
export TIMEOUT="${TIMEOUT:-10800}"
export K="${K:-30}"
export PERPLEXITY="${PERPLEXITY:-15}"
export SEED="${SEED:-4}"

export DATASETS="${DATASETS:-COIL20,USPS,FashionMNIST,FlowRepository_FR-FCM-ZYRM_files,flow18,MNIST,imagenet,MetRef,mass41,TabulaMuris}"
export METHODS="${METHODS:-fastEmbedR_opentsne_cuda,fastEmbedR_umap_cuda_fuzzy,fastEmbedR_umap_cuda_binary}"

export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export VECLIB_MAXIMUM_THREADS="${THREADS}"
export RCPP_PARALLEL_NUM_THREADS="${THREADS}"
export APPTAINERENV_OMP_NUM_THREADS="${THREADS}"
export APPTAINERENV_OPENBLAS_NUM_THREADS="${THREADS}"
export APPTAINERENV_MKL_NUM_THREADS="${THREADS}"
export APPTAINERENV_RCPP_PARALLEL_NUM_THREADS="${THREADS}"
export APPTAINERENV_LD_LIBRARY_PATH="/opt/rapids/lib:/opt/faiss/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export SINGULARITYENV_OMP_NUM_THREADS="${THREADS}"
export SINGULARITYENV_OPENBLAS_NUM_THREADS="${THREADS}"
export SINGULARITYENV_MKL_NUM_THREADS="${THREADS}"
export SINGULARITYENV_RCPP_PARALLEL_NUM_THREADS="${THREADS}"
export SINGULARITYENV_LD_LIBRARY_PATH="/opt/rapids/lib:/opt/faiss/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

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
  RUNNER=("${CONTAINER_BIN}" exec --nv --bind "${BASE_DIR}:${BASE_DIR}" --pwd "${BASE_DIR}" "${SINGULARITY_IMAGE}")
fi

{
  echo "Starting CUDA embedding benchmark"
  echo "BASE_DIR=${BASE_DIR}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "THREADS=${THREADS}"
  echo "TIMEOUT=${TIMEOUT}"
  echo "DATASETS=${DATASETS}"
  echo "METHODS=${METHODS}"
  echo "CUDA diagnostics:"
  "${RUNNER[@]}" bash -c '
    nvidia-smi || true
    echo "PATH=${PATH}"
    RSCRIPT="$(command -v Rscript || true)"
    if [[ -z "${RSCRIPT}" ]]; then
      for candidate in /usr/local/bin/Rscript /usr/bin/Rscript /opt/conda/bin/Rscript /opt/R/*/bin/Rscript; do
        if [[ -x "${candidate}" ]]; then RSCRIPT="${candidate}"; break; fi
      done
    fi
    echo "Rscript=${RSCRIPT:-NOT_FOUND}"
    if [[ -n "${RSCRIPT}" ]]; then
      "${RSCRIPT}" -e "cat(\"fastEmbedR diagnostics\\n\"); library(fastEmbedR); print(utils::packageVersion(\"fastEmbedR\")); print(\"cuda_available\" %in% getNamespaceExports(\"fastEmbedR\")); print(\"backend_info\" %in% getNamespaceExports(\"fastEmbedR\")); cat(\"faissR diagnostics\\n\"); library(faissR); print(try(faissR::backend_info(), silent=TRUE)); print(try(faissR::cuda_available(), silent=TRUE)); print(try(faissR::cuvs_available(), silent=TRUE))"
    fi
  ' || true
  "${RUNNER[@]}" Rscript "${BENCH_R}" \
    --script="${BENCH_R}" \
    --backend_group=cuda \
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
} 2>&1 | tee -a "${OUT_DIR}/benchmark_cuda.log"
