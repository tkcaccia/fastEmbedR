#!/bin/sh

set -eu

repo_root=$(cd "$(dirname "$0")/.." && pwd)
stamp=$(date +%Y%m%d_%H%M%S)

remote_host=${FASTEMBEDR_REMOTE_HOST:-137.158.224.178}
remote_user=${FASTEMBEDR_REMOTE_USER:-chiamaka}
remote_base=${FASTEMBEDR_REMOTE_BASE:-/mnt/sata_ssd}
remote_dir=${FASTEMBEDR_REMOTE_DIR:-"$remote_base/fastEmbedR_faiss_cuvs_$stamp"}
remote="$remote_user@$remote_host"

ssh_cmd=${SSH:-ssh}
scp_cmd=${SCP:-scp}

archive="${TMPDIR:-/tmp}/fastEmbedR_faiss_cuvs_${stamp}.tar.gz"
remote_script="${TMPDIR:-/tmp}/fastEmbedR_faiss_cuvs_remote_${stamp}.sh"
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

log="$REMOTE_DIR/mnist70k_faiss_cuvs_opentsne.log"
mkdir -p "$REMOTE_DIR/source" "$REMOTE_DIR/results"
exec > "$log" 2>&1

echo "fastEmbedR MNIST70k FAISS/cuVS KNN + openTSNE benchmark"
echo "date: $(date)"
echo "host: $(hostname)"
echo "workdir: $REMOTE_DIR"

cd "$REMOTE_DIR/source"
tar -xzf "$REMOTE_DIR/source.tar.gz"

echo
echo "System"
uname -a
R --version
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi unavailable"
fi

echo
echo "Checking R dependencies"
Rscript - <<'RS'
options(repos = c(CRAN = "https://cloud.r-project.org"))
pkgs <- c("Rcpp", "RcppHNSW", "jsonlite")
missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) install.packages(missing, dependencies = TRUE)
RS

echo
echo "Ensuring cuVS installation"
if [ -f "$HOME/.fastEmbedR/cuvs_env.sh" ]; then
  . "$HOME/.fastEmbedR/cuvs_env.sh"
else
  sh tools/install_cuvs_linux.sh
  . "$HOME/.fastEmbedR/cuvs_env.sh"
fi

echo
echo "Trying to add FAISS CPU to the cuVS micromamba environment"
if [ -n "${CUVS_HOME:-}" ] && [ -x "$HOME/.fastEmbedR/micromamba/bin/micromamba" ]; then
  "$HOME/.fastEmbedR/micromamba/bin/micromamba" install -y -p "$CUVS_HOME" -c conda-forge faiss-cpu || true
fi

if [ -n "${CUVS_HOME:-}" ] && [ -f "$CUVS_HOME/include/faiss/IndexFlat.h" ]; then
  export FASTEMBEDR_USE_FAISS=1
  export FAISS_HOME="$CUVS_HOME"
else
  export FASTEMBEDR_USE_FAISS=0
  unset FAISS_HOME || true
fi

echo
echo "Build environment"
echo "CUVS_HOME=${CUVS_HOME:-}"
echo "CUDA_HOME=${CUDA_HOME:-}"
echo "FASTEMBEDR_USE_FAISS=$FASTEMBEDR_USE_FAISS"
echo "FAISS_HOME=${FAISS_HOME:-}"

echo
echo "Installing fastEmbedR"
rm -f src/*.o src/*.so src/*.dylib
FASTEMBEDR_USE_CUDA=1 FASTEMBEDR_USE_CUVS=1 R CMD INSTALL .

echo
echo "Backend info"
Rscript - <<'RS'
library(fastEmbedR)
print(backend_info())
cat("faiss_available=", faiss_available(), "\n")
cat("cuvs_available=", cuvs_available(), "\n")
cat("cuda_available=", cuda_available(), "\n")
RS

methods="fastEmbedR_cuda_cuvs_nndescent,fastEmbedR_cuda_cuvs_cagra,fastEmbedR_cuda_cuvs_bruteforce,fastEmbedR_cpu_nndescent,RcppHNSW_hnsw"
if Rscript -e 'library(fastEmbedR); quit(status = if (isTRUE(faiss_available())) 0 else 1)' >/dev/null 2>&1; then
  methods="FAISS_Flat_L2,FAISS_Flat_IP,FAISS_IVF,FAISS_IVFPQ,FAISS_HNSW,FAISS_NSG,FAISS_NNDescent,$methods"
fi

echo
echo "Methods: $methods"
Rscript tools/benchmark_mnist70k_knn_fourway_opentsne.R \
  --n=70000 \
  --k=50 \
  --threads=4 \
  --methods="$methods" \
  --out-dir="$REMOTE_DIR/results/mnist70k_faiss_cuvs_opentsne"

echo
echo "Benchmark results"
cat "$REMOTE_DIR/results/mnist70k_faiss_cuvs_opentsne/opentsne_knn_fourway_70k.csv"
echo
echo "DONE_DIR=$REMOTE_DIR/results/mnist70k_faiss_cuvs_opentsne"
REMOTE

echo "Creating remote benchmark directory: $remote:$remote_dir"
$ssh_cmd "$remote" "mkdir -p '$remote_dir/source'"

echo "Uploading source archive..."
$scp_cmd "$archive" "$remote:$remote_dir/source.tar.gz"
echo "Uploading remote script..."
$scp_cmd "$remote_script" "$remote:$remote_dir/run_faiss_cuvs_opentsne_remote.sh"

echo "Running remote benchmark..."
if ! $ssh_cmd "$remote" "REMOTE_DIR='$remote_dir' sh '$remote_dir/run_faiss_cuvs_opentsne_remote.sh'"
then
  echo "Remote benchmark failed. Remote log follows if available:" >&2
  $ssh_cmd "$remote" "cat '$remote_dir/mnist70k_faiss_cuvs_opentsne.log' 2>/dev/null || true" >&2
  exit 1
fi

echo
echo "Remote benchmark log:"
$ssh_cmd "$remote" "tail -n 120 '$remote_dir/mnist70k_faiss_cuvs_opentsne.log'"
echo
echo "Fetching remote results..."
mkdir -p "$repo_root/results/chiamaka_faiss_cuvs_$stamp"
$scp_cmd "$remote:$remote_dir/results/mnist70k_faiss_cuvs_opentsne/opentsne_knn_fourway_70k.csv" \
  "$repo_root/results/chiamaka_faiss_cuvs_$stamp/"
$scp_cmd "$remote:$remote_dir/results/mnist70k_faiss_cuvs_opentsne/opentsne_knn_fourway_70k.png" \
  "$repo_root/results/chiamaka_faiss_cuvs_$stamp/" || true
echo "Local results directory: $repo_root/results/chiamaka_faiss_cuvs_$stamp"
echo "Remote directory: $remote:$remote_dir"
