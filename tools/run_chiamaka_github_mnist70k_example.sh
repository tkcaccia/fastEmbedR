#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-chiamaka@137.158.224.178}"
KEY="${KEY:-.codex_chiamaka_ed25519}"
REMOTE_BASE="${REMOTE_BASE:-/mnt/sata_ssd/fastEmbedR_github_example}"
REMOTE_SRC="${REMOTE_SRC:-$REMOTE_BASE/source}"
REMOTE_OUT="${REMOTE_OUT:-$REMOTE_BASE/results/mnist70k_github_chiamaka_$(date +%Y%m%d_%H%M%S)}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/mnt/sata_ssd/fastEmbedR/Data/MNIST}"
THREADS="${THREADS:-4}"

ssh_cmd=(ssh -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=no "$REMOTE")
rsync_cmd=(rsync -az --delete -e "ssh -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=no")

"${ssh_cmd[@]}" "mkdir -p '$REMOTE_SRC' '$REMOTE_OUT'"

"${rsync_cmd[@]}" \
  --exclude '.git' \
  --exclude '.r-lib' \
  --exclude 'results' \
  --exclude 'singularity' \
  --exclude '.codex_chiamaka_ed25519*' \
  ./ "$REMOTE:$REMOTE_SRC/"

"${ssh_cmd[@]}" "cd '$REMOTE_SRC' && R CMD INSTALL --preclean ."

"${ssh_cmd[@]}" "cd '$REMOTE_SRC' && \
  OMP_NUM_THREADS=$THREADS \
  OPENBLAS_NUM_THREADS=$THREADS \
  MKL_NUM_THREADS=$THREADS \
  RCPP_PARALLEL_NUM_THREADS=$THREADS \
  Rscript tools/benchmark_github_mnist70k.R \
    --n=70000 \
    --k=15 \
    --perplexity=15 \
    --threads=$THREADS \
    --run-metal=false \
    --run-cuda=true \
    --run-references=true \
    --data-root='$REMOTE_DATA_ROOT' \
    --out-dir='$REMOTE_OUT'"

mkdir -p results/chiamaka_github_example
"${rsync_cmd[@]}" "$REMOTE:$REMOTE_OUT/" "results/chiamaka_github_example/"

echo "Remote output: $REMOTE_OUT"
echo "Local copy: results/chiamaka_github_example"
