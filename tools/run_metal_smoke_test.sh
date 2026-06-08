#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KERNEL="$ROOT/tools/metal_smoke_kernel.metal"
AIR="$ROOT/tools/metal_smoke_kernel.air"
LIB="$ROOT/tools/metal_smoke_kernel.metallib"
BIN="$ROOT/tools/metal_smoke_test"

METAL_BIN="$(xcrun -find metal 2>/dev/null || true)"
METALLIB_BIN="$(xcrun -find metallib 2>/dev/null || true)"

if [[ -z "$METAL_BIN" || -z "$METALLIB_BIN" ]]; then
  echo "Metal command-line compiler not found. Install full Xcode or Metal toolchain support." >&2
  exit 2
fi

"$METAL_BIN" -c "$KERNEL" -o "$AIR"
"$METALLIB_BIN" "$AIR" -o "$LIB"
clang++ -std=c++17 -fobjc-arc "$ROOT/tools/metal_smoke_test.mm" -framework Foundation -framework Metal -o "$BIN"
"$BIN" "$LIB"
