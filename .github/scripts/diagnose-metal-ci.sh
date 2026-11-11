#!/usr/bin/env bash
# Print bounded Metal diagnostics before R CMD check on hosted macOS runners.
set -u

printf 'Metal diagnostic host: %s\n' "$(sw_vers -productVersion)"
xcodebuild -version
system_profiler SPDisplaysDataType | head -60

if ! xcrun --find metal; then
  printf 'Metal command-line compiler is unavailable on this runner.\n'
  exit 0
fi

source=src/native_knn_metal_impl.mm
shader="$RUNNER_TEMP/fastembedr-knn.metal"
if [ "$(grep -cF 'const char* kMetalSource = R"METAL(' "$source")" \
    -ne 1 ] || [ "$(grep -cF ')METAL";' "$source")" -ne 1 ]; then
  printf 'Metal source extraction failed.\n'
  exit 0
fi
sed -n '/^const char\* kMetalSource = R"METAL($/,/^)METAL";$/p' \
  "$source" | sed '1d;$d' > "$shader"
if [ ! -s "$shader" ] ||
    ! grep -qF '#include <metal_stdlib>' "$shader"; then
  printf 'Metal source extraction failed.\n'
  exit 0
fi
printf 'Native KNN Metal source bytes: %s\n' "$(wc -c < "$shader")"

if xcrun -sdk macosx metal -std=metal3.1 -c \
    "$RUNNER_TEMP/fastembedr-knn.metal" \
    -o "$RUNNER_TEMP/fastembedr-knn.air" \
    >"$RUNNER_TEMP/fastembedr-metal-compile.log" 2>&1; then
  printf 'Offline native KNN Metal compilation passed.\n'
else
  printf 'Offline native KNN Metal compilation failed:\n'
  head -80 "$RUNNER_TEMP/fastembedr-metal-compile.log"
fi

swift .github/scripts/probe-metal-runtime.swift "$shader" \
  > "$RUNNER_TEMP/fastembedr-metal-runtime.out" \
  2> "$RUNNER_TEMP/fastembedr-metal-runtime.err" &
probe_pid=$!
for ((second = 0; second < 60; ++second)); do
  if ! kill -0 "$probe_pid" 2>/dev/null; then break; fi
  sleep 1
done
if kill -0 "$probe_pid" 2>/dev/null; then
  kill "$probe_pid" 2>/dev/null
  wait "$probe_pid" 2>/dev/null
  probe_status=124
else
  wait "$probe_pid"
  probe_status=$?
fi
printf 'Runtime Metal probe exit: %s\n' "$probe_status"
head -c 5000 "$RUNNER_TEMP/fastembedr-metal-runtime.out"
head -c 2000 "$RUNNER_TEMP/fastembedr-metal-runtime.err"
