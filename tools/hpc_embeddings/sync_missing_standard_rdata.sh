#!/usr/bin/env bash

set -euo pipefail

# Copy standard RData files that reference R packages need.
# fastEmbedR uses *_float32.RData, but Rtsne/uwot/umap/FIt-SNE benchmarks
# intentionally load the standard R dataset files.

SRC_ROOT="${SRC_ROOT:-/Users/stefano/Documents/fastEmbedR/Data}"
DST_ROOT="${DST_ROOT:-/Users/stefano/HPC-firenze/NN/Data}"

copy_one() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "${dst}")"
  if [[ ! -f "${src}" ]]; then
    echo "MISSING source: ${src}" >&2
    return 1
  fi
  echo "Copying ${src}"
  echo "     -> ${dst}"
  cp -p "${src}" "${dst}"
}

copy_one "${SRC_ROOT}/COIL20/COIL20.RData" \
         "${DST_ROOT}/COIL20/COIL20.RData"

copy_one "${SRC_ROOT}/FashionMNIST/FashionMNIST.RData" \
         "${DST_ROOT}/FashionMNIST/FashionMNIST.RData"

copy_one "${SRC_ROOT}/FlowRepository_FR-FCM-ZYRM_files/van_unen_FR-FCM-ZYRM.RData" \
         "${DST_ROOT}/FlowRepository_FR-FCM-ZYRM_files/van_unen_FR-FCM-ZYRM.RData"

echo "Done. Re-sync ${DST_ROOT} to /scratch/firenze/NN/Data before rerunning the HPC benchmark."
