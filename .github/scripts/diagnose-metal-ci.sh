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

if ! python3 - "$RUNNER_TEMP" <<'PY'
from pathlib import Path
import sys

source = Path('src/native_knn_metal_impl.mm').read_text()
start = 'const char* kMetalSource = R"METAL('
end = ')METAL";'
if source.count(start) != 1:
    raise SystemExit('Could not isolate the native KNN Metal source')
shader = source.split(start, 1)[1].split(end, 1)[0]
if not shader or '#include <metal_stdlib>' not in shader:
    raise SystemExit('Native KNN Metal source is incomplete')
Path(sys.argv[1], 'fastembedr-knn.metal').write_text(shader)
print('Native KNN Metal source bytes:', len(shader.encode()))
PY
then
  printf 'Metal source extraction failed.\n'
  exit 0
fi

if xcrun -sdk macosx metal -std=metal3.1 -c \
    "$RUNNER_TEMP/fastembedr-knn.metal" \
    -o "$RUNNER_TEMP/fastembedr-knn.air" \
    >"$RUNNER_TEMP/fastembedr-metal-compile.log" 2>&1; then
  printf 'Offline native KNN Metal compilation passed.\n'
else
  printf 'Offline native KNN Metal compilation failed:\n'
  head -80 "$RUNNER_TEMP/fastembedr-metal-compile.log"
fi

python3 - "$RUNNER_TEMP/fastembedr-knn.metal" <<'PY'
import subprocess
import sys

command = ["swift", ".github/scripts/probe-metal-runtime.swift", sys.argv[1]]
try:
    result = subprocess.run(command, capture_output=True, text=True,
                            timeout=60, check=False)
    print("Runtime Metal probe exit:", result.returncode)
    print(result.stdout[:5000])
    print(result.stderr[:2000])
except (OSError, subprocess.TimeoutExpired) as error:
    print("Runtime Metal probe unavailable or timed out:", error)
PY
