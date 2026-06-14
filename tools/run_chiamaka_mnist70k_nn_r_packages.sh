#!/bin/sh

set -eu

repo_root=$(cd "$(dirname "$0")/.." && pwd)
stamp=$(date +%Y%m%d_%H%M%S)

remote_host=${FASTEMBEDR_REMOTE_HOST:-137.158.224.178}
remote_user=${FASTEMBEDR_REMOTE_USER:-chiamaka}
remote_base=${FASTEMBEDR_REMOTE_BASE:-/mnt/sata_ssd}
remote_dir=${FASTEMBEDR_REMOTE_DIR:-"$remote_base/fastEmbedR_nn_pkg_$stamp"}
remote="$remote_user@$remote_host"

ssh_cmd=${SSH:-ssh}
scp_cmd=${SCP:-scp}
timeout_cmd=${FASTEMBEDR_TIMEOUT:-timeout}
method_timeout=${FASTEMBEDR_METHOD_TIMEOUT_SEC:-1800}

archive="${TMPDIR:-/tmp}/fastEmbedR_nn_pkg_${stamp}.tar.gz"
remote_script="${TMPDIR:-/tmp}/fastEmbedR_nn_pkg_remote_${stamp}.sh"
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

log="$REMOTE_DIR/mnist70k_nn_r_packages.log"
mkdir -p "$REMOTE_DIR/source" "$REMOTE_DIR/results" "$REMOTE_DIR/method_results"
exec > "$log" 2>&1

echo "MNIST70k R NN package benchmark"
echo "date: $(date)"
echo "host: $(hostname)"
echo "workdir: $REMOTE_DIR"
echo "method timeout seconds: $METHOD_TIMEOUT"

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
echo "Installing/checking R packages"
Rscript - <<'RS'
options(repos = c(CRAN = "https://cloud.r-project.org"))
cran_pkgs <- c(
  "Rnanoflann", "RANN", "RcppHNSW", "RcppAnnoy",
  "Rtsne", "uwot", "umap", "jsonlite", "remotes"
)
missing <- cran_pkgs[!vapply(cran_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
  install.packages(missing, dependencies = TRUE)
}
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}
bioc_pkgs <- c("BiocNeighbors", "BiocParallel")
missing_bioc <- bioc_pkgs[!vapply(bioc_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_bioc)) {
  BiocManager::install(missing_bioc, ask = FALSE, update = FALSE)
}
if (!requireNamespace("cuda.ml", quietly = TRUE)) {
  try(remotes::install_github("mlverse/cuda.ml", upgrade = "never"), silent = TRUE)
}
pkgs <- c(cran_pkgs, bioc_pkgs, "cuda.ml")
for (pkg in pkgs) {
  cat(pkg, if (requireNamespace(pkg, quietly = TRUE)) as.character(packageVersion(pkg)) else "not_installed", "\n")
}
RS

methods="Rnanoflann_exact RANN_kd_exact RANN_bd_exact RcppHNSW_hnsw RcppAnnoy_euclidean BiocNeighbors_exhaustive BiocNeighbors_vptree BiocNeighbors_kmknn BiocNeighbors_annoy BiocNeighbors_hnsw uwot_fnn_internal uwot_annoy_internal uwot_hnsw_internal uwot_nndescent_internal Rtsne_neighbors_api umap_knn_api cuda_ml_knn"

echo
echo "Running methods"
for method in $methods; do
  echo
  echo "=== $method ==="
  out="$REMOTE_DIR/method_results/${method}.csv"
  set +e
  "$TIMEOUT_CMD" "$METHOD_TIMEOUT" Rscript tools/benchmark_mnist70k_nn_r_packages.R \
    --n=70000 \
    --k=50 \
    --threads=4 \
    --method="$method" \
    --cache-dir="$REMOTE_DIR/cache" \
    --out-dir="$REMOTE_DIR/method_results/${method}_out" \
    --out-file="$out"
  status=$?
  set -e
  if [ "$status" -eq 124 ] || [ "$status" -eq 137 ]; then
    echo "$method timed out"
    Rscript - <<RS
row <- data.frame(
  machine = Sys.info()[["nodename"]],
  dataset = "mnist70k_flattened_idx",
  method = "$method",
  package = NA_character_,
  algorithm = NA_character_,
  backend = NA_character_,
  status = "timeout",
  error_message = "Killed by shell timeout after $METHOD_TIMEOUT seconds",
  n = 70000L,
  p = 784L,
  k = 50L,
  threads = 4L,
  build_sec = NA_real_,
  query_sec = NA_real_,
  total_sec = NA_real_,
  n_neighbors_returned = NA_integer_,
  package_version = NA_character_,
  parameters_json = "{}",
  timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
  stringsAsFactors = FALSE
)
write.csv(row, "$out", row.names = FALSE)
RS
  elif [ "$status" -ne 0 ]; then
    echo "$method failed with exit status $status"
    if [ ! -f "$out" ]; then
      Rscript - <<RS
row <- data.frame(
  machine = Sys.info()[["nodename"]],
  dataset = "mnist70k_flattened_idx",
  method = "$method",
  package = NA_character_,
  algorithm = NA_character_,
  backend = NA_character_,
  status = "failed",
  error_message = "Rscript exited with status $status before writing output",
  n = 70000L,
  p = 784L,
  k = 50L,
  threads = 4L,
  build_sec = NA_real_,
  query_sec = NA_real_,
  total_sec = NA_real_,
  n_neighbors_returned = NA_integer_,
  package_version = NA_character_,
  parameters_json = "{}",
  timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
  stringsAsFactors = FALSE
)
write.csv(row, "$out", row.names = FALSE)
RS
    fi
  fi
done

echo
echo "Combining results"
Rscript - <<'RS'
files <- list.files(file.path(Sys.getenv("REMOTE_DIR"), "method_results"), pattern = "\\.csv$", full.names = TRUE)
rows <- lapply(files, function(f) {
  tryCatch(read.csv(f, stringsAsFactors = FALSE), error = function(e) NULL)
})
rows <- Filter(Negate(is.null), rows)
res <- do.call(rbind, rows)
res <- res[order(res$status != "success", res$total_sec, na.last = TRUE), , drop = FALSE]
out_dir <- file.path(Sys.getenv("REMOTE_DIR"), "results")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(res, file.path(out_dir, "mnist70k_nn_package_benchmark.csv"), row.names = FALSE)
ok <- res[res$status == "success", , drop = FALSE]
if (nrow(ok) > 0L) {
  png(file.path(out_dir, "mnist70k_nn_package_timing.png"), width = 1800, height = 1000, res = 145)
  par(mar = c(11, 5, 4, 2) + 0.1)
  ord <- order(ok$total_sec, decreasing = TRUE)
  cols <- ifelse(ok$backend[ord] == "cuda", "#2a9d8f", "#4361ee")
  barplot(ok$total_sec[ord],
          names.arg = paste(ok$package[ord], ok$algorithm[ord], sep = "\n"),
          las = 2, col = cols, border = NA,
          ylab = "NN time (seconds)",
          main = "MNIST 70k flattened NN package benchmark, k=50")
  box()
  dev.off()
}
print(res[, c("method", "package", "algorithm", "backend", "status", "total_sec", "error_message")], row.names = FALSE)
RS

echo
echo "Benchmark completed"
echo "results: $REMOTE_DIR/results"
REMOTE

echo "Creating remote NN benchmark directory: $remote:$remote_dir"
$ssh_cmd "$remote" "mkdir -p '$remote_dir/source'"

echo "Uploading source archive..."
$scp_cmd "$archive" "$remote:$remote_dir/source.tar.gz"
echo "Uploading remote benchmark script..."
$scp_cmd "$remote_script" "$remote:$remote_dir/run_mnist70k_nn_r_packages_remote.sh"

echo "Running remote NN package benchmark..."
if ! $ssh_cmd "$remote" "REMOTE_DIR='$remote_dir' TIMEOUT_CMD='$timeout_cmd' METHOD_TIMEOUT='$method_timeout' sh '$remote_dir/run_mnist70k_nn_r_packages_remote.sh'"
then
  echo "Remote NN benchmark failed. Remote log follows if available:" >&2
  $ssh_cmd "$remote" "cat '$remote_dir/mnist70k_nn_r_packages.log' 2>/dev/null || true" >&2
  exit 1
fi

local_out="$repo_root/results/chiamaka_mnist70k_nn_r_packages_$stamp"
mkdir -p "$local_out"
$scp_cmd "$remote:$remote_dir/results/"* "$local_out/" 2>/dev/null || true

echo
echo "Remote log:"
$ssh_cmd "$remote" "cat '$remote_dir/mnist70k_nn_r_packages.log'"
echo
echo "Local results: $local_out"
echo "Remote directory: $remote:$remote_dir"
