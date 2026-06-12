#!/bin/sh

set -eu

# Install the RAPIDS cuVS C/C++ SDK into a local micromamba environment.
#
# This script intentionally does not vendor RAPIDS into fastEmbedR. It creates
# an external, reproducible cuVS installation and writes a small activation file
# that can be sourced before `R CMD INSTALL`.

prefix=${FASTEMBEDR_CUVS_PREFIX:-"$HOME/.fastEmbedR/micromamba"}
env_name=${FASTEMBEDR_CUVS_ENV:-fastembedr-cuvs}
cuda_version=${FASTEMBEDR_CUVS_CUDA_VERSION:-}
rapids_channel=${FASTEMBEDR_CUVS_RAPIDS_CHANNEL:-rapidsai}
conda_channel=${FASTEMBEDR_CUVS_CONDA_CHANNEL:-conda-forge}
nvidia_channel=${FASTEMBEDR_CUVS_NVIDIA_CHANNEL:-nvidia}
channel_priority=${FASTEMBEDR_CUVS_CHANNEL_PRIORITY:-flexible}
shell_hook="$prefix/etc/profile.d/micromamba.sh"
env_file="$HOME/.fastEmbedR/cuvs_env.sh"

if [ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]; then
  echo "cuVS binary packages are Linux-only. Use this script on a CUDA Linux host." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found. Install/run this on a machine with an NVIDIA GPU driver." >&2
  exit 1
fi

if [ -z "$cuda_version" ]; then
  if command -v nvcc >/dev/null 2>&1; then
    nvcc_major=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\).*/\1/p' | head -n 1)
  else
    nvcc_major=""
  fi
  case "$nvcc_major" in
    13) cuda_version=13.2 ;;
    12) cuda_version=12.9 ;;
    *)
      echo "Could not infer a supported RAPIDS CUDA version from nvcc." >&2
      echo "Set FASTEMBEDR_CUVS_CUDA_VERSION, for example 13.2 or 12.9." >&2
      exit 1
      ;;
  esac
fi

mkdir -p "$prefix" "$(dirname "$env_file")"

if [ ! -x "$prefix/bin/micromamba" ]; then
  echo "Installing micromamba into $prefix"
  tmpdir=$(mktemp -d)
  trap 'rm -rf "$tmpdir"' EXIT
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C "$tmpdir" bin/micromamba >/dev/null
  mkdir -p "$prefix/bin"
  mv "$tmpdir/bin/micromamba" "$prefix/bin/micromamba"
fi

if [ -f "$shell_hook" ]; then
  # shellcheck disable=SC1090
  . "$shell_hook"
else
  export MAMBA_ROOT_PREFIX="$prefix"
  eval "$("$prefix/bin/micromamba" shell hook -s posix -r "$prefix")"
fi

if ! "$prefix/bin/micromamba" env list | awk '{print $1}' | grep -qx "$env_name"; then
  echo "Creating micromamba environment $env_name with libcuvs cuda-version=$cuda_version"
  "$prefix/bin/micromamba" create -y -n "$env_name" \
    --channel-priority "$channel_priority" \
    -c "$rapids_channel" -c "$conda_channel" -c "$nvidia_channel" \
    "libcuvs" "cuda-nvcc" "cuda-cudart-dev" "libcufft-dev" "cuda-version=$cuda_version"
else
  echo "Updating micromamba environment $env_name with libcuvs cuda-version=$cuda_version"
  "$prefix/bin/micromamba" install -y -n "$env_name" \
    --channel-priority "$channel_priority" \
    -c "$rapids_channel" -c "$conda_channel" -c "$nvidia_channel" \
    "libcuvs" "cuda-nvcc" "cuda-cudart-dev" "libcufft-dev" "cuda-version=$cuda_version"
fi

env_prefix=$("$prefix/bin/micromamba" env list | awk -v name="$env_name" '$1 == name {print $NF}')
if [ -z "$env_prefix" ] || [ ! -f "$env_prefix/include/cuvs/neighbors/cagra.h" ]; then
  echo "cuVS headers were not found after installation." >&2
  exit 1
fi

cat > "$env_file" <<EOF
# Source this before building fastEmbedR with RAPIDS cuVS.
export CUVS_HOME="$env_prefix"
export CUDA_HOME="$env_prefix"
export NVCC="$env_prefix/bin/nvcc"
export FASTEMBEDR_USE_CUDA=1
export FASTEMBEDR_USE_CUVS=1
export PATH="$env_prefix/bin:\${PATH}"
export LD_LIBRARY_PATH="$env_prefix/lib:$env_prefix/lib64:\${LD_LIBRARY_PATH:-}"
export R_LD_LIBRARY_PATH="$env_prefix/lib:$env_prefix/lib64:\${R_LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$env_prefix/lib/libstdc++.so.6\${LD_PRELOAD:+:\$LD_PRELOAD}"
EOF

echo
echo "cuVS installed in: $env_prefix"
echo "Activation file: $env_file"
echo
echo "Build fastEmbedR with:"
echo "  . \"$env_file\""
echo "  R CMD INSTALL ."
