#!/usr/bin/env bash

set -euo pipefail

# Install corrected benchmark scripts and the standard RData files required by
# reference R packages into the local HPC mirror. Run this locally, then sync
# /Users/stefano/HPC-firenze/NN to the HPC.

SRC_ROOT="${SRC_ROOT:-/Users/stefano/Documents/umap}"
STAGED_RDATA="${STAGED_RDATA:-${SRC_ROOT}/tmp_chiamaka_rdata}"
HPC_ROOT="${HPC_ROOT:-/Users/stefano/HPC-firenze/NN}"

copy_file() {
  local src="$1"
  local dst="$2"
  if [[ ! -f "${src}" ]]; then
    echo "Missing source: ${src}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${dst}")"
  echo "Copying ${src}"
  echo "     -> ${dst}"
  cp -p "${src}" "${dst}"
}

copy_file "${SRC_ROOT}/tools/hpc_embeddings/benchmark_embeddings_float32_publication.R" \
          "${HPC_ROOT}/benchmark_embeddings_float32_publication.R"

copy_file "${SRC_ROOT}/tools/hpc_embeddings/benchmark_embeddings_float32_cpu12.sh" \
          "${HPC_ROOT}/benchmark_embeddings_float32_cpu12.sh"

copy_file "${SRC_ROOT}/tools/hpc_embeddings/benchmark_embeddings_float32_cuda.sh" \
          "${HPC_ROOT}/benchmark_embeddings_float32_cuda.sh"

copy_file "${STAGED_RDATA}/COIL20/COIL20.RData" \
          "${HPC_ROOT}/Data/COIL20/COIL20.RData"

copy_file "${STAGED_RDATA}/FashionMNIST/FashionMNIST.RData" \
          "${HPC_ROOT}/Data/FashionMNIST/FashionMNIST.RData"

copy_file "${STAGED_RDATA}/FlowRepository_FR-FCM-ZYRM_files/van_unen_FR-FCM-ZYRM.RData" \
          "${HPC_ROOT}/Data/FlowRepository_FR-FCM-ZYRM_files/van_unen_FR-FCM-ZYRM.RData"

chmod +x "${HPC_ROOT}/benchmark_embeddings_float32_cpu12.sh" \
         "${HPC_ROOT}/benchmark_embeddings_float32_cuda.sh"

echo "Done. Now sync ${HPC_ROOT} to /scratch/firenze/NN on the HPC."
