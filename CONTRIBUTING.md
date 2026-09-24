# Contributing to fastEmbedR

Thank you for helping improve fastEmbedR. The package combines R orchestration,
portable C++17, Apple Metal, and NVIDIA CUDA code, so changes should identify
their affected backend and include evidence appropriate to that backend.

## Before opening a change

1. Open an issue describing the behavior, affected backend, and a reproducible
   example. Security reports must follow `SECURITY.md` instead.
2. Create a focused branch from the current development branch.
3. Keep public APIs backend-explicit: an unavailable requested backend must
   fail clearly and must never fall back silently.
4. Retain upstream copyright and license notices for copied or adapted code.
   Update `inst/NOTICE`, `inst/COPYRIGHTS`, and
   `inst/THIRD_PARTY_DEPENDENCIES.json` when provenance changes.

## Development checks

For a CPU build:

```sh
R CMD build .
R CMD check --as-cran --no-manual fastEmbedR_*.tar.gz
```

Run the tests from an installed package, or during development with:

```sh
Rscript -e 'testthat::test_local()'
```

Generate host-side coverage evidence with:

```sh
Rscript tools/report_test_coverage.R --output-dir coverage-evidence/local
```

Coverage produced from a dirty checkout is diagnostic only. `covr` does not
measure execution inside Metal or CUDA kernels.

Metal changes require an Apple Silicon Mac with the supported macOS/Xcode
versions. CUDA changes require a compatible Linux/NVIDIA toolchain. Before a
release, run `.github/workflows/hardware-validation.yaml` on the real CPU,
Metal, and CUDA runners and archive all commit-named evidence artifacts.

## Tests and numerical changes

- Add unit tests for narrow behavior and integration tests for public workflows.
- For accelerator work, assert both availability and the backend recorded by
  the result; a CPU result is not evidence for a GPU path.
- Changes to affinities, graph construction, forces, or optimizers require
  comparison with the portable reference implementation and declared numeric
  tolerances.
- Performance changes require repeated timing after warm-up and a visual and
  quantitative quality comparison. Do not accept a speedup that changes the
  intended objective without documenting it as a separate method.

## Style

- Use `n.cores` for public CPU thread controls.
- Use American spelling, including “nearest-neighbor.”
- Keep authored lines at 80 columns or fewer and R functions at 50 lines or
  fewer.
- Prefer small native helpers with explicit ownership of buffers and backend
  dispatch over adding more branches to already large functions.
- Extend an existing execution path where possible. A new path must have a
  caller, visible metadata, and a focused test; remove the path it replaces.
- Do not add silent fallbacks, unused compatibility aliases, or benchmark-only
  branches to production code.
- Format new C++ with the surrounding file's style and compile as C++17.

## Pull requests

Pull requests should state the tested package commit, operating system,
compiler, hardware, requested and observed backend, test commands, and any
numerical or performance evidence. All CPU checks must pass. Backend-specific
changes additionally require the corresponding real-hardware evidence.
