#!/usr/bin/env bash
set -euo pipefail

backend="${1:?backend is required}"
out_dir="${2:?output directory is required}"

case "$backend" in
  cpu|metal|cuda) ;;
  *) echo "backend must be cpu, metal, or cuda" >&2; exit 2 ;;
esac

root="$(git rev-parse --show-toplevel)"
out_dir="$(mkdir -p "$out_dir" && cd "$out_dir" && pwd)"
lib_dir="$out_dir/R-library"
mkdir -p "$lib_dir"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

cd "$root"

git rev-parse HEAD > "$out_dir/git-commit.txt"
git status --porcelain=v1 > "$out_dir/git-status.txt"
if [[ -s "$out_dir/git-status.txt" &&
      "${FASTEMBEDR_ALLOW_DIRTY_VALIDATION:-0}" != "1" ]]; then
  echo "Hardware validation requires a clean checkout." >&2
  cat "$out_dir/git-status.txt" >&2
  exit 3
fi

{
  echo "backend=$backend"
  echo "timestamp_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "runner_name=${RUNNER_NAME:-manual}"
  echo "runner_os=${RUNNER_OS:-$(uname -s)}"
  echo "runner_arch=${RUNNER_ARCH:-$(uname -m)}"
  uname -a
  R --version | head -n 1
  if [[ "$backend" == "metal" ]]; then
    xcrun --find metal
    xcrun metal --version
    system_profiler SPHardwareDataType SPDisplaysDataType
  elif [[ "$backend" == "cuda" ]]; then
    nvidia-smi
    "${NVCC:-${CUDA_HOME:-/usr/local/cuda}/bin/nvcc}" --version
  else
    command -v lscpu >/dev/null 2>&1 && lscpu || true
  fi
} > "$out_dir/hardware.txt" 2>&1

build_dir="$(mktemp -d "${TMPDIR:-/tmp}/fastembedr-hardware.XXXXXX")"
trap 'rm -rf "$build_dir" "$lib_dir"' EXIT

if [[ -n "${FASTEMBEDR_VALIDATION_LOCALE:-}" ]]; then
  build_locale="$FASTEMBEDR_VALIDATION_LOCALE"
elif [[ "$(uname -s)" == "Darwin" ]]; then
  build_locale="en_US.UTF-8"
else
  build_locale="C.UTF-8"
fi

if [[ "$backend" == "cpu" ]]; then
  export FASTEMBEDR_USE_CUDA=0
elif [[ "$backend" == "metal" ]]; then
  export FASTEMBEDR_USE_CUDA=0
  export SDKROOT="${SDKROOT:-$(xcrun --sdk macosx --show-sdk-path)}"
else
  export FASTEMBEDR_USE_CUDA=1
  export FASTEMBEDR_USE_CUVS=1
  export FASTEMBEDR_USE_RAFT="${FASTEMBEDR_USE_RAFT:-1}"
  export FASTEMBEDR_REQUIRE_CUDA=1
  export PACKAGE_REQUIRE_CUDA=1
fi

(
  cd "$build_dir"
  LC_ALL="$build_locale" LANG="$build_locale" \
    R CMD build "$root" --compact-vignettes=gs+qpdf
) > "$out_dir/build.log" 2>&1
package_tar="$(find "$build_dir" -maxdepth 1 -name 'fastEmbedR_*.tar.gz' -print -quit)"
test -n "$package_tar"
package_sha256="$(sha256_file "$package_tar")"
source_sha256="$package_sha256"
cp "$package_tar" "$out_dir/"

R CMD INSTALL --preclean --library="$lib_dir" "$package_tar" \
  > "$out_dir/install.log" 2>&1

export R_LIBS="$lib_dir${R_LIBS:+:$R_LIBS}"
export FASTEMBEDR_VALIDATION_BACKEND="$backend"

dll_path="$(Rscript -e 'library(fastEmbedR); cat(getLoadedDLLs()[["fastEmbedR"]][["path"]])' \
  2> "$out_dir/dll-load.log")"
dll_sha256="$(sha256_file "$dll_path")"

cat > "$out_dir/identity.csv" <<EOF
"backend","git_commit","source_sha256","package_tar_sha256","installed_dll_sha256","installed_dll"
"$backend","$(cat "$out_dir/git-commit.txt")","$source_sha256","$package_sha256","$dll_sha256","$dll_path"
EOF

cat > "$out_dir/evidence-scope.csv" <<EOF
"evidence_class","backend","performance_claim","description"
"hardware_smoke_and_correctness","$backend",FALSE,"Strict execution, backend identity, numerical checks, and package tests on the named device; not a full scientific performance benchmark"
EOF

Rscript .github/scripts/validate-hardware.R "$backend" "$out_dir" \
  > "$out_dir/runtime-validation.log" 2>&1

numerical_backends="cpu"
if [[ "$backend" != "cpu" ]]; then
  numerical_backends="cpu,$backend"
fi
Rscript tools/validate_tsne_numerics.R \
  --backends="$numerical_backends" \
  --out-dir="$out_dir/tsne-numerical-validation" \
  > "$out_dir/tsne-numerical-validation.log" 2>&1

Rscript .github/scripts/run-testthat-timed.R "$out_dir" \
  > "$out_dir/testthat.log" 2>&1

(
  cd "$build_dir"
  LC_ALL="$build_locale" LANG="$build_locale" \
    R CMD check --as-cran "$package_tar"
) > "$out_dir/check.log" 2>&1
check_dir="$build_dir/fastEmbedR.Rcheck"
test -f "$check_dir/00check.log"
cp "$check_dir/00check.log" "$out_dir/00check.log"
grep -E '^(Status:|[0-9]+ ERROR|[0-9]+ WARNING|[0-9]+ NOTE|\* DONE)' \
  "$check_dir/00check.log" > "$out_dir/check-summary.txt" || true
grep -Fq '* DONE' "$check_dir/00check.log"
if grep -Eq '^Status:.*(ERROR|WARNING)' \
  "$check_dir/00check.log"; then
  echo "R CMD check reported an ERROR or WARNING." >&2
  exit 5
fi

git status --porcelain=v1 > "$out_dir/git-status-after.txt"
if ! cmp -s "$out_dir/git-status.txt" "$out_dir/git-status-after.txt"; then
  echo "Validation modified the checkout." >&2
  diff -u "$out_dir/git-status.txt" \
    "$out_dir/git-status-after.txt" >&2 || true
  exit 4
fi

find "$out_dir" -type f ! -name SHA256SUMS ! -path '*/R-library/*' -print | LC_ALL=C sort | \
  while IFS= read -r file; do
    printf '%s  %s\n' "$(sha256_file "$file")" "${file#$out_dir/}"
  done > "$out_dir/SHA256SUMS"

echo "Validated fastEmbedR backend=$backend source_sha256=$source_sha256"
