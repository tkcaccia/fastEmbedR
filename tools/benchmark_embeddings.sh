#!/bin/bash

#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=2 --gres=gpu:l40s:1
#SBATCH --time=48:00:00
#SBATCH --job-name="fastEmbedR"

set -euo pipefail

# Run fastEmbedR publication benchmarks on the Firenze HPC Singularity image.
#
# This is the main HPC entry point. It runs:
#   BENCHMARK #2: t-SNE/openTSNE implementations
#   BENCHMARK #3: UMAP implementations
#
# It intentionally uses the full benchmark scripts, not the older compact
# hpc_benchmark_embeddings.R smoke runner. External R implementations are
# included by default: Rtsne, tsne, KlugerLab/FIt-SNE, uwot, and umap.
#
# Typical HPC use:
#   sbatch /scratch/firenze/NN/benchmark_embeddings.sh
#
# Override examples:
#   DATASETS=MNIST,FashionMNIST THREADS_CPU=2 sbatch benchmark_embeddings.sh
#   METHODS=opentsne_cpu,opentsne_cuda,Rtsne,tsne,KlugerLab_FItSNE,umap_cpu_binary,umap_cpu_fuzzy,umap_cuda_binary,umap_cuda_fuzzy,uwot,umap sbatch benchmark_embeddings.sh

BASE_DIR="${BASE_DIR:-/scratch/firenze/NN}"
DATA_ROOT="${DATA_ROOT:-${BASE_DIR}/Data}"
SIF="${SIF:-${BASE_DIR}/singularity/fastembedr_cuda.sif}"
SCRIPT_DIR="${SCRIPT_DIR:-${BASE_DIR}}"
BENCHMARK2_SCRIPT="${BENCHMARK2_SCRIPT:-${SCRIPT_DIR}/benchmark2_tsne_speed_accuracy.R}"
BENCHMARK3_SCRIPT="${BENCHMARK3_SCRIPT:-${SCRIPT_DIR}/benchmark3_umap_speed_accuracy.R}"
OUT_DIR="${OUT_DIR:-${BASE_DIR}/benchmark_embeddings_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${BASE_DIR}/benchmark_logs}"

THREADS_CPU="${THREADS_CPU:-2}"
SEED="${SEED:-4}"
K="${K:-auto}"
KNN_K="${KNN_K:-100}"
PERPLEXITY="${PERPLEXITY:-auto}"
TIMEOUT="${TIMEOUT:-600}"
SHELL_SUPERVISE_WORKERS="${SHELL_SUPERVISE_WORKERS:-TRUE}"

DATASETS="${DATASETS:-USPS,FashionMNIST,FlowRepository_FR-FCM-ZYRM_files,flow18,MNIST,imagenet,MetRef,mass41,TabulaMuris,COIL20}"

# User-facing method names. These are expanded below into the exact internal
# rows used by BENCHMARK #2 and BENCHMARK #3.
METHODS="${METHODS:-opentsne_cpu,opentsne_cuda,umap_cpu_binary,umap_cpu_fuzzy,umap_cuda_binary,umap_cuda_fuzzy,Rtsne,tsne,KlugerLab_FItSNE,uwot,umap}"
BACKENDS2="${BACKENDS2:-cpu,cuda,cpu_fft}"
BACKENDS3="${BACKENDS3:-cpu,cuda}"

append_csv_unique() {
  local current="$1"
  local add="$2"
  local out="$current"
  IFS=',' read -r -a add_items <<< "$add"
  for item in "${add_items[@]}"; do
    item="$(echo "$item" | xargs)"
    [[ -z "$item" ]] && continue
    if [[ ",${out}," != *",${item},"* ]]; then
      if [[ -z "$out" ]]; then
        out="$item"
      else
        out="${out},${item}"
      fi
    fi
  done
  echo "$out"
}

METHODS2=""
METHODS3=""
IFS=',' read -r -a requested_methods <<< "$METHODS"
for method in "${requested_methods[@]}"; do
  method="$(echo "$method" | xargs)"
  case "$method" in
    opentsne_cpu)
      METHODS2="$(append_csv_unique "$METHODS2" "fastEmbedR_opentsne_cpu_grid128")"
      ;;
    opentsne_cuda)
      METHODS2="$(append_csv_unique "$METHODS2" "fastEmbedR_opentsne_cuda")"
      ;;
    Rtsne)
      METHODS2="$(append_csv_unique "$METHODS2" "Rtsne_neighbors,Rtsne_Rtsne")"
      ;;
    tsne)
      METHODS2="$(append_csv_unique "$METHODS2" "tsne_package")"
      ;;
    KlugerLab_FItSNE)
      METHODS2="$(append_csv_unique "$METHODS2" "KlugerLab_FItSNE")"
      ;;
    umap_cpu_binary)
      METHODS3="$(append_csv_unique "$METHODS3" "fastEmbedR_umap_cpu_binary")"
      ;;
    umap_cpu_fuzzy)
      METHODS3="$(append_csv_unique "$METHODS3" "fastEmbedR_umap_cpu_fuzzy")"
      ;;
    umap_cuda_binary)
      METHODS3="$(append_csv_unique "$METHODS3" "fastEmbedR_umap_cuda_binary")"
      ;;
    umap_cuda_fuzzy)
      METHODS3="$(append_csv_unique "$METHODS3" "fastEmbedR_umap_cuda_fuzzy")"
      ;;
    uwot)
      METHODS3="$(append_csv_unique "$METHODS3" "uwot_umap_fast_sgd_precomputed,uwot_umap_default_precomputed,uwot_umap_fast_sgd_internal_nn,uwot_umap_default_internal_nn")"
      ;;
    umap)
      METHODS3="$(append_csv_unique "$METHODS3" "umap_package")"
      ;;
    "")
      ;;
    *)
      echo "Unknown method in METHODS: ${method}" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$METHODS2" && -z "$METHODS3" ]]; then
  echo "No benchmark methods selected." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${OUT_DIR}" "${OUT_DIR}/home" "${OUT_DIR}/.cache/fontconfig" "${OUT_DIR}/tmp" "${OUT_DIR}/BENCHMARK2" "${OUT_DIR}/BENCHMARK3"

if [[ ! -f "${SIF}" ]]; then
  echo "Singularity image not found: ${SIF}" >&2
  exit 1
fi
if [[ ! -f "${BENCHMARK2_SCRIPT}" ]]; then
  echo "BENCHMARK #2 R script not found: ${BENCHMARK2_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${BENCHMARK3_SCRIPT}" ]]; then
  echo "BENCHMARK #3 R script not found: ${BENCHMARK3_SCRIPT}" >&2
  exit 1
fi

container_ld_library_path="${LD_LIBRARY_PATH:-/usr/local/cuda/lib64:/opt/conda/lib:/opt/conda/targets/x86_64-linux/lib}"
container_r_libs="/opt/conda/lib/R/library"

export APPTAINERENV_OMP_NUM_THREADS="${THREADS_CPU}"
export APPTAINERENV_OPENBLAS_NUM_THREADS="${THREADS_CPU}"
export APPTAINERENV_MKL_NUM_THREADS="${THREADS_CPU}"
export APPTAINERENV_RCPP_PARALLEL_NUM_THREADS="${THREADS_CPU}"
export APPTAINERENV_R_LIBS="${container_r_libs}"
export APPTAINERENV_R_LIBS_USER="${container_r_libs}"
export APPTAINERENV_R_LIBS_SITE="${container_r_libs}"
export APPTAINERENV_LD_LIBRARY_PATH="${container_ld_library_path}"
export APPTAINERENV_HOME="${OUT_DIR}/home"
export APPTAINERENV_XDG_CACHE_HOME="${OUT_DIR}/.cache"
export APPTAINERENV_FONTCONFIG_CACHE="${OUT_DIR}/.cache/fontconfig"
export APPTAINERENV_TMPDIR="${OUT_DIR}/tmp"
export APPTAINERENV_TEMPDIR="${OUT_DIR}/tmp"
export APPTAINERENV_FAST_TSNE_BIN="${FAST_TSNE_BIN:-/opt/fit-sne/bin/fast_tsne}"

RUN_LOG="${LOG_DIR}/benchmark_embeddings_$(date +%Y%m%d_%H%M%S).log"
STATUS_LOG="${OUT_DIR}/run_status.log"

{
  echo "Starting fastEmbedR publication benchmarks"
  echo "  base dir:     ${BASE_DIR}"
  echo "  data root:    ${DATA_ROOT}"
  echo "  output dir:   ${OUT_DIR}"
  echo "  image:        ${SIF}"
  echo "  benchmark #2: ${BENCHMARK2_SCRIPT}"
  echo "  benchmark #3: ${BENCHMARK3_SCRIPT}"
  echo "  methods #2:   ${METHODS2}"
  echo "  methods #3:   ${METHODS3}"
  echo "  datasets:     ${DATASETS}"
  echo "  CPU cores:    ${THREADS_CPU}"
  echo "  log:          ${RUN_LOG}"
} | tee "${STATUS_LOG}"

run_container_rscript() {
  singularity exec --nv --no-home \
    -B "${BASE_DIR}:${BASE_DIR}" \
    "${SIF}" \
    Rscript --vanilla "$@"
}

sanitize_name() {
  echo "$1" | sed 's/[^A-Za-z0-9_.-]/_/g'
}

write_failed_worker_row() {
  local row_file="$1"
  local dataset="$2"
  local method="$3"
  local code="$4"
  local benchmark="$5"
  local status="failed"
  local message="worker terminated with exit code ${code}"
  if [[ "${code}" == "124" || "${code}" == "137" ]]; then
    status="timeout_or_oom"
    message="worker exceeded timeout or was OOM-killed with exit code ${code}"
  fi
  mkdir -p "$(dirname "${row_file}")"
  {
    echo "dataset,method,package,backend_requested,backend_used,status,error_message"
    printf '"%s","%s","%s","%s","%s","%s","%s"\n' \
      "${dataset}" "${method}" "${benchmark}" "unknown" "unknown" "${status}" "${message}"
  } > "${row_file}"
}

run_worker_rows() {
  local benchmark="$1"
  local script="$2"
  local out_subdir="$3"
  local methods_csv="$4"
  local backends_csv="$5"
  local log_file="$6"

  mkdir -p "${out_subdir}/worker_rows" "${out_subdir}/worker_logs"
  IFS=',' read -r -a dataset_items <<< "${DATASETS}"
  IFS=',' read -r -a method_items <<< "${methods_csv}"

  for dataset in "${dataset_items[@]}"; do
    dataset="$(echo "${dataset}" | xargs)"
    [[ -z "${dataset}" ]] && continue
    for method in "${method_items[@]}"; do
      method="$(echo "${method}" | xargs)"
      [[ -z "${method}" ]] && continue
      local row_file="${out_subdir}/worker_rows/$(sanitize_name "${dataset}")__$(sanitize_name "${method}").csv"
      local worker_log="${out_subdir}/worker_logs/$(sanitize_name "${dataset}")__$(sanitize_name "${method}").log"
      if [[ "${FASTEMBEDR_BENCHMARK_RESUME:-TRUE}" == "TRUE" && -f "${row_file}" ]]; then
        echo "[$(date -Is)] ${benchmark}: skipping existing ${dataset} / ${method}" | tee -a "${log_file}"
        continue
      fi
      echo "[$(date -Is)] ${benchmark}: running ${dataset} / ${method}" | tee -a "${log_file}"
      set +e
      timeout --kill-after=30s "${TIMEOUT}" \
        singularity exec --nv --no-home \
          -B "${BASE_DIR}:${BASE_DIR}" \
          "${SIF}" \
          Rscript --vanilla "${script}" \
            --worker=TRUE \
            --data_root="${DATA_ROOT}" \
            --out_dir="${out_subdir}" \
            --datasets="${DATASETS}" \
            --methods="${method}" \
            --backends="${backends_csv}" \
            --benchmark1_dir="" \
            --dataset="${dataset}" \
            --method="${method}" \
            --perplexity="${PERPLEXITY}" \
            --k="${K}" \
            --knn_k="${KNN_K}" \
            --threads="${THREADS_CPU}" \
            --timeout="${TIMEOUT}" \
            --seed="${SEED}" \
            --row_out="${row_file}" \
        > "${worker_log}" 2>&1
      local code=$?
      set -e
      if [[ "${code}" -ne 0 || ! -f "${row_file}" ]]; then
        echo "[$(date -Is)] ${benchmark}: ${dataset} / ${method} failed code=${code}; recording row and continuing" | tee -a "${log_file}"
        write_failed_worker_row "${row_file}" "${dataset}" "${method}" "${code}" "${benchmark}"
      fi
    done
  done
}

finalize_benchmark() {
  local benchmark="$1"
  local script="$2"
  local out_subdir="$3"
  local methods_csv="$4"
  local backends_csv="$5"
  local finalize_log="$6"
  run_container_rscript "${script}" \
    --finalize=TRUE \
    --data_root="${DATA_ROOT}" \
    --out_dir="${out_subdir}" \
    --datasets="${DATASETS}" \
    --methods="${methods_csv}" \
    --backends="${backends_csv}" \
    --perplexity="${PERPLEXITY}" \
    --k="${K}" \
    --knn_k="${KNN_K}" \
    --threads="${THREADS_CPU}" \
    --timeout="${TIMEOUT}" \
    > "${finalize_log}" 2>&1
  echo "[$(date -Is)] finalized ${benchmark}" | tee -a "${STATUS_LOG}"
}

set +e
echo "[$(date -Is)] starting BENCHMARK #2" | tee -a "${STATUS_LOG}"
if [[ "${SHELL_SUPERVISE_WORKERS}" == "TRUE" ]]; then
  set -e
  run_worker_rows "BENCHMARK2" "${BENCHMARK2_SCRIPT}" "${OUT_DIR}/BENCHMARK2" "${METHODS2}" "${BACKENDS2}" "${OUT_DIR}/BENCHMARK2/benchmark2.log"
  finalize_benchmark "BENCHMARK2" "${BENCHMARK2_SCRIPT}" "${OUT_DIR}/BENCHMARK2" "${METHODS2}" "${BACKENDS2}" "${OUT_DIR}/BENCHMARK2/benchmark2_finalize.log"
  set +e
  status2=0
else
  run_container_rscript "${BENCHMARK2_SCRIPT}" \
    --data_root="${DATA_ROOT}" \
    --out_dir="${OUT_DIR}/BENCHMARK2" \
    --datasets="${DATASETS}" \
    --methods="${METHODS2}" \
    --backends="${BACKENDS2}" \
    --perplexity="${PERPLEXITY}" \
    --k="${K}" \
    --knn_k="${KNN_K}" \
    --threads="${THREADS_CPU}" \
    --timeout="${TIMEOUT}" \
    > "${OUT_DIR}/BENCHMARK2/benchmark2.log" 2>&1
  status2=$?
fi
echo "[$(date -Is)] finished BENCHMARK #2 status=${status2}" | tee -a "${STATUS_LOG}"

echo "[$(date -Is)] starting BENCHMARK #3" | tee -a "${STATUS_LOG}"
if [[ "${SHELL_SUPERVISE_WORKERS}" == "TRUE" ]]; then
  set -e
  run_worker_rows "BENCHMARK3" "${BENCHMARK3_SCRIPT}" "${OUT_DIR}/BENCHMARK3" "${METHODS3}" "${BACKENDS3}" "${OUT_DIR}/BENCHMARK3/benchmark3.log"
  finalize_benchmark "BENCHMARK3" "${BENCHMARK3_SCRIPT}" "${OUT_DIR}/BENCHMARK3" "${METHODS3}" "${BACKENDS3}" "${OUT_DIR}/BENCHMARK3/benchmark3_finalize.log"
  set +e
  status3=0
else
  run_container_rscript "${BENCHMARK3_SCRIPT}" \
    --data_root="${DATA_ROOT}" \
    --out_dir="${OUT_DIR}/BENCHMARK3" \
    --datasets="${DATASETS}" \
    --methods="${METHODS3}" \
    --backends="${BACKENDS3}" \
    --perplexity="${PERPLEXITY}" \
    --k="${K}" \
    --knn_k="${KNN_K}" \
    --threads="${THREADS_CPU}" \
    --timeout="${TIMEOUT}" \
    > "${OUT_DIR}/BENCHMARK3/benchmark3.log" 2>&1
  status3=$?
fi
echo "[$(date -Is)] finished BENCHMARK #3 status=${status3}" | tee -a "${STATUS_LOG}"
set -e

{
  echo "Benchmark finished"
  echo "  BENCHMARK #2 status: ${status2}"
  echo "  BENCHMARK #3 status: ${status3}"
  echo "Results:"
  echo "  ${OUT_DIR}/BENCHMARK2/benchmark2_tsne_results.csv"
  echo "  ${OUT_DIR}/BENCHMARK2/plots"
  echo "  ${OUT_DIR}/BENCHMARK3/benchmark3_umap_results.csv"
  echo "  ${OUT_DIR}/BENCHMARK3/plots"
} | tee -a "${STATUS_LOG}"

if [[ "${status2}" -ne 0 || "${status3}" -ne 0 ]]; then
  exit 1
fi
