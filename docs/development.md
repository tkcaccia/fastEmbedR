# Development And Software Quality

[Home](../README.md) |
[Installation](installation.md) |
[Implementation](implementation.md) |
[API](usage-api.md) |
[Backend validation](backend-validation.md) |
**Development**

fastEmbedR uses four complementary evidence layers. Passing one layer does not
substitute for another.

## Evidence Layers

1. **Portable package checks.** `R CMD check --as-cran` runs on hosted Linux,
   macOS, and Windows CPU environments, with an additional Linux R-devel lane.
   These checks compile the CPU package, execute tests and examples, and
   rebuild the vignettes.
2. **Host-side coverage.** `covr` records R and instrumentable compiled host
   lines. It does not observe Metal or CUDA device-kernel execution.
3. **Real-hardware smoke/correctness validation.** Explicitly labelled Apple
   Silicon Metal and Linux/NVIDIA CUDA runners verify capability,
   requested-versus-observed backend identity, matrix and precomputed-KNN
   workflows, numerical tests, and the complete installed-package test suite.
   This is not a scientific performance benchmark.
4. **Publication release evidence.** Clean source, installed binaries,
   containers, coverage reports, hardware logs, and benchmark results are tied
   to one package commit and one benchmark commit by SHA-256 and archived under
   a persistent DOI.

The [real-hardware validation contract](backend-validation.md) describes the
backend evidence. `CONTRIBUTING.md` describes the contributor workflow.

## Coverage

Run:

```sh
Rscript tools/report_test_coverage.R --output-dir coverage-evidence/local
```

The command writes per-file and per-component CSV files plus commit,
worktree-state, platform, and session metadata. A report from a dirty checkout
is diagnostic and is not release evidence. Device-kernel coverage remains
`not_measured` until collected by a device-aware tool on real hardware; it must
never be inferred from host-side bridge coverage.

## CI Matrix

| Workflow | Trigger | Platform | Evidence |
|---|---|---|---|
| R CMD check | Push, pull request, manual | Hosted Linux and macOS CPU | Package build, tests, examples, vignettes, compiled-code checks |
| Software quality | Push, pull request, manual | Hosted Linux CPU | Component coverage CSV, session and commit metadata |
| Real hardware validation | Manual and release tag | Hosted Linux CPU; self-hosted Apple Silicon Metal; self-hosted Linux/NVIDIA CUDA | Strict hardware smoke/correctness: backend identity, numerical tests, full test suite, hashes; no performance claim |
| pkgdown | Push and manual | Hosted macOS CPU | Published API and vignette site |

Configured workflows are not counted as executed evidence. A release is not
hardware-validated until successful, commit-named CPU, Metal, and CUDA
artifacts have been copied to the long-term archive.

## Sanitizers And Static Analysis

`R CMD check` performs R and compiled-code checks on every portable CI run.
AddressSanitizer, UndefinedBehaviorSanitizer, Valgrind, clang-tidy, and device
kernel coverage are not currently continuous release gates. Their status is
reported as `not_implemented` rather than implied by ordinary package checks.
This is tracked as a software-quality limitation and a development priority.

## Community And Governance

Bug and feature templates collect the package identity, backend, hardware, and
reproduction needed for useful reports. The pull-request template requires
backend-specific validation. `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, and
`SECURITY.md` define participation, review, and private vulnerability-reporting
paths.

Repository stars, downloads, contributors, issues, and pull requests are
time-varying community indicators, not correctness metrics. A dated community
snapshot may be included in a cover letter, while the article and release
archive focus on reproducible software evidence.
