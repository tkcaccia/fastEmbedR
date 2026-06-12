#!/usr/bin/env bash
set -euo pipefail

repo="${1:-/tmp/fastembedr_fft_eval/AppleSiliconFFT}"
src="$repo/src"

if [[ ! -d "$src" ]]; then
  echo "AppleSiliconFFT source directory not found: $src" >&2
  exit 1
fi

cd "$src"

xcrun metal -o fft_multi.air fft_multisize.metal
xcrun metallib -o default.metallib fft_multi.air

sdk="$(xcrun --sdk macosx --show-sdk-path)"
toolchain="/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/clang/include"

# Some local Macs have stale SDK module maps in /usr/local/include. The
# explicit include isolation below keeps Swift/Clang on the selected Xcode SDK.
swiftc -O -parse-as-library -framework Metal -framework Accelerate \
  -Xcc -nostdinc \
  -Xcc -isystem -Xcc "$toolchain" \
  -Xcc -isystem -Xcc "$sdk/usr/include" \
  -Xcc -iframeworkwithsysroot -Xcc /System/Library/Frameworks \
  -o fft_multi_host fft_multisize_host.swift

./fft_multi_host
