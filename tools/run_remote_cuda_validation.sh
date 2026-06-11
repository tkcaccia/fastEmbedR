#!/bin/sh

set -eu

repo_root=$(cd "$(dirname "$0")/.." && pwd)
stamp=$(date +%Y%m%d_%H%M%S)

remote_host=${FASTEMBEDR_REMOTE_HOST:-137.158.224.178}
remote_user=${FASTEMBEDR_REMOTE_USER:-chiamaka}
remote_base=${FASTEMBEDR_REMOTE_BASE:-/mnt/sata_ssd}
remote_dir=${FASTEMBEDR_REMOTE_DIR:-"$remote_base/fastEmbedR_cuda_$stamp"}
remote="$remote_user@$remote_host"

ssh_cmd=${SSH:-ssh}
scp_cmd=${SCP:-scp}

archive="${TMPDIR:-/tmp}/fastEmbedR_cuda_${stamp}.tar.gz"
remote_script="${TMPDIR:-/tmp}/fastEmbedR_cuda_remote_${stamp}.sh"
cleanup() {
  rm -f "$archive" "$remote_script"
}
trap cleanup EXIT

cd "$repo_root"
tar \
  --exclude='.git' \
  --exclude='fastEmbedR.Rcheck' \
  --exclude='*.tar.gz' \
  --exclude='src/*.o' \
  --exclude='src/*.so' \
  --exclude='src/*.dylib' \
  --exclude='results' \
  -czf "$archive" .

cat > "$remote_script" <<'REMOTE'
set -eu

log="$REMOTE_DIR/cuda_validation.log"
mkdir -p "$REMOTE_DIR/source"
exec > "$log" 2>&1

echo "fastEmbedR CUDA validation"
echo "date: $(date)"
echo "host: $(hostname)"
echo "workdir: $REMOTE_DIR"

cd "$REMOTE_DIR/source"
tar -xzf "$REMOTE_DIR/source.tar.gz"

echo
echo "Dataset discovery"
chmod +x tools/discover_fastpls_datasets.sh
tools/discover_fastpls_datasets.sh "$REMOTE_DIR"

echo
echo "System"
uname -a
if ! command -v R >/dev/null 2>&1; then
  echo "R is not available on the remote machine."
  exit 2
fi
R --version

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc is not available. Set CUDA_HOME/NVCC or install the CUDA toolkit."
  exit 2
fi
nvcc --version

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is not available; CUDA driver visibility cannot be verified."
  exit 2
fi
nvidia-smi

echo
echo "CUDA smoke test"
chmod +x tools/run_cuda_smoke_test.sh
FASTEMBEDR_USE_CUDA=1 tools/run_cuda_smoke_test.sh

echo
echo "Installed package CUDA tests"
if Rscript -e 'quit(status = if (requireNamespace("testthat", quietly = TRUE)) 0 else 1)' >/dev/null 2>&1; then
  Rscript - <<'RS'
library(testthat)
library(fastEmbedR)
stopifnot(isTRUE(cuda_available()))
stopifnot(isTRUE(fastEmbedR:::embedding_cuda_available_cpp()))
test_dir("tests/testthat", reporter = "summary")
RS
else
  echo "testthat is not installed; direct CUDA smoke tests already passed."
fi

echo
echo "CUDA validation completed successfully."
REMOTE

echo "Creating remote CUDA validation directory: $remote:$remote_dir"
$ssh_cmd "$remote" "mkdir -p '$remote_dir/source'"

echo "Uploading source archive..."
$scp_cmd "$archive" "$remote:$remote_dir/source.tar.gz"
echo "Uploading remote validation script..."
$scp_cmd "$remote_script" "$remote:$remote_dir/run_cuda_validation_remote.sh"

echo "Running remote CUDA validation..."
if ! $ssh_cmd "$remote" "REMOTE_DIR='$remote_dir' sh '$remote_dir/run_cuda_validation_remote.sh'"
then
  echo "Remote CUDA validation failed. Remote log follows if available:" >&2
  $ssh_cmd "$remote" "cat '$remote_dir/cuda_validation.log' 2>/dev/null || true" >&2
  exit 1
fi

echo
echo "Remote CUDA validation log:"
$ssh_cmd "$remote" "cat '$remote_dir/cuda_validation.log'"
echo
echo "Remote directory: $remote:$remote_dir"
