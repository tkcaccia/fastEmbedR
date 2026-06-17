#!/bin/sh

#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=2 --gres=gpu:l40s:1
#SBATCH --time=48:00:00
#SBATCH --job-name="MyJob"


set -euo pipefail

# Run fastEmbedR embedding benchmarks on the HPC Singularity image.
#
# Defaults match the Firenze HPC layout:
#   /scratch/firenze/NN/Data
#   /scratch/firenze/NN/singularity/fastembedr_cuda.sif
#
# Override from the shell when needed, for example:
#   THREADS_CPU=2 DATASETS=MNIST METHODS=opentsne_cpu,opentsne_cuda bash run_hpc_benchmark_embeddings_singularity.sh

BASE_DIR="${BASE_DIR:-/scratch/firenze/NN}"
DATA_ROOT="${DATA_ROOT:-${BASE_DIR}/Data}"
SIF="${SIF:-${BASE_DIR}/singularity/fastembedr_cuda.sif}"
SCRIPT="${SCRIPT:-${BASE_DIR}/hpc_benchmark_embeddings.R}"
OUT_DIR="${OUT_DIR:-${BASE_DIR}/benchmark_embeddings_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${BASE_DIR}/benchmark_logs}"

THREADS_CPU="${THREADS_CPU:-2}"
SEED="${SEED:-4}"
SAVED_KNN_K="${SAVED_KNN_K:-100}"
EMBED_K="${EMBED_K:-15}"
PERPLEXITY="${PERPLEXITY:-15}"

DATASETS="${DATASETS:-COIL20,USPS,FashionMNIST,FlowRepository_FR-FCM-ZYRM_files,flow18,MNIST,imagenet,MetRef,mass41,TabulaMuris}"
METHODS="${METHODS:-opentsne_cpu,opentsne_cuda,umap_cpu_binary,umap_cpu_fuzzy,umap_cuda_binary,umap_cuda_fuzzy}"
FORCE="${FORCE:-FALSE}"

mkdir -p "${LOG_DIR}" "${OUT_DIR}" "${OUT_DIR}/home" "${OUT_DIR}/.cache/fontconfig" "${OUT_DIR}/tmp"

if [[ ! -f "${SIF}" ]]; then
  echo "Singularity image not found: ${SIF}" >&2
  exit 1
fi

if [[ ! -f "${SCRIPT}" ]]; then
  echo "Benchmark R script not found: ${SCRIPT}" >&2
  echo "Copy tools/hpc_benchmark_embeddings.R to ${SCRIPT} first." >&2
  exit 1
fi

container_ld_library_path="${LD_LIBRARY_PATH:-/usr/local/cuda/lib64:/opt/conda/lib:/opt/conda/targets/x86_64-linux/lib}"
container_r_libs_user="${R_LIBS_USER:-/opt/conda/lib/R/library}"
container_r_libs_site="${R_LIBS_SITE:-/opt/conda/lib/R/library}"

export APPTAINERENV_OMP_NUM_THREADS="${THREADS_CPU}"
export APPTAINERENV_OPENBLAS_NUM_THREADS="${THREADS_CPU}"
export APPTAINERENV_MKL_NUM_THREADS="${THREADS_CPU}"
export APPTAINERENV_RCPP_PARALLEL_NUM_THREADS="${THREADS_CPU}"
export APPTAINERENV_R_LIBS="${container_r_libs_user}"
export APPTAINERENV_R_LIBS_USER="${container_r_libs_user}"
export APPTAINERENV_R_LIBS_SITE="${container_r_libs_site}"
export APPTAINERENV_LD_LIBRARY_PATH="${container_ld_library_path}"
export APPTAINERENV_HOME="${OUT_DIR}/home"
export APPTAINERENV_XDG_CACHE_HOME="${OUT_DIR}/.cache"
export APPTAINERENV_FONTCONFIG_CACHE="${OUT_DIR}/.cache/fontconfig"
export APPTAINERENV_TMPDIR="${OUT_DIR}/tmp"
export APPTAINERENV_TEMPDIR="${OUT_DIR}/tmp"

# Legacy Singularity variables are opt-in because Apptainer prints warnings
# when both names are present.
if [[ "${USE_LEGACY_SINGULARITYENV:-FALSE}" == "TRUE" ]]; then
  export SINGULARITYENV_OMP_NUM_THREADS="${APPTAINERENV_OMP_NUM_THREADS}"
  export SINGULARITYENV_OPENBLAS_NUM_THREADS="${APPTAINERENV_OPENBLAS_NUM_THREADS}"
  export SINGULARITYENV_MKL_NUM_THREADS="${APPTAINERENV_MKL_NUM_THREADS}"
  export SINGULARITYENV_RCPP_PARALLEL_NUM_THREADS="${APPTAINERENV_RCPP_PARALLEL_NUM_THREADS}"
  export SINGULARITYENV_R_LIBS="${APPTAINERENV_R_LIBS}"
  export SINGULARITYENV_R_LIBS_USER="${APPTAINERENV_R_LIBS_USER}"
  export SINGULARITYENV_R_LIBS_SITE="${APPTAINERENV_R_LIBS_SITE}"
  export SINGULARITYENV_LD_LIBRARY_PATH="${APPTAINERENV_LD_LIBRARY_PATH}"
  export SINGULARITYENV_HOME="${APPTAINERENV_HOME}"
  export SINGULARITYENV_XDG_CACHE_HOME="${APPTAINERENV_XDG_CACHE_HOME}"
  export SINGULARITYENV_FONTCONFIG_CACHE="${APPTAINERENV_FONTCONFIG_CACHE}"
  export SINGULARITYENV_TMPDIR="${APPTAINERENV_TMPDIR}"
  export SINGULARITYENV_TEMPDIR="${APPTAINERENV_TEMPDIR}"
fi

RUN_LOG="${LOG_DIR}/benchmark_embeddings_$(date +%Y%m%d_%H%M%S).log"

echo "Starting fastEmbedR benchmark"
echo "  base dir:   ${BASE_DIR}"
echo "  data root:  ${DATA_ROOT}"
echo "  output dir: ${OUT_DIR}"
echo "  image:      ${SIF}"
echo "  methods:    ${METHODS}"
echo "  datasets:   ${DATASETS}"
echo "  CPU cores:  ${THREADS_CPU}"
echo "  log:        ${RUN_LOG}"

singularity exec --nv --no-home \
  -B "${BASE_DIR}:${BASE_DIR}" \
  "${SIF}" \
  Rscript --vanilla "${SCRIPT}" \
    --base_dir="${BASE_DIR}" \
    --data_root="${DATA_ROOT}" \
    --out_dir="${OUT_DIR}" \
    --threads_cpu="${THREADS_CPU}" \
    --seed="${SEED}" \
    --saved_knn_k="${SAVED_KNN_K}" \
    --embed_k="${EMBED_K}" \
    --perplexity="${PERPLEXITY}" \
    --datasets="${DATASETS}" \
    --methods="${METHODS}" \
    --force="${FORCE}" \
  2>&1 | tee "${RUN_LOG}"

echo "Benchmark finished"
echo "Results:"
echo "  ${OUT_DIR}/embedding_benchmark_results.csv"
echo "  ${OUT_DIR}/plots"
echo "  ${OUT_DIR}/layouts"
