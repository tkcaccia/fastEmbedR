# Real-Hardware Backend Validation

GPU tests are skipped during ordinary CPU-only package checks. A release must
therefore archive separate evidence from machines that actually expose Apple
Metal and NVIDIA CUDA. A macOS build alone is not accepted as Metal evidence:
the runtime capability checks and native kernels must execute successfully.

The package repository provides the manually dispatched **Real hardware
validation** GitHub Actions workflow. CPU uses a hosted Linux runner. Metal and
CUDA use explicitly labelled self-hosted runners:

- `[self-hosted, macOS, ARM64, fastembedr-metal]`;
- `[self-hosted, Linux, X64, fastembedr-cuda]`.

The project-owned labels ensure that accelerator jobs are not scheduled on a
machine without the requested hardware. Larger scientific benchmarks remain
in the separate
[`fastEmbedR-extra`](https://github.com/tkcaccia/fastEmbedR-extra)
repository.

Runner registration and custom labels follow GitHub's
[self-hosted runner guidance](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/use-in-a-workflow).

## Evidence Levels

Hardware statements use three non-interchangeable labels:

| Evidence level | What it establishes | What it does not establish |
|---|---|---|
| Full benchmark validation | Release-locked runtime, memory, and quality on the named machine and dataset suite, with repetitions and complete manifests | Performance on another processor, GPU generation, operating system, or release |
| Hardware smoke/correctness testing | The exact commit builds, executes the requested backend without fallback, passes numerical checks, and completes package tests on the named device | Representative throughput or scalability |
| Build-level architectural compatibility | The toolchain accepts the target and package plus linked libraries contain compatible device code | Successful execution, numerical correctness, or performance on that device |

At the current manuscript freeze, Metal performance evidence is limited to
seven datasets on one Apple M3 system. No performance claim is transferred to
another Apple Silicon generation. Other Apple Silicon configurations meeting
the documented macOS/Xcode contract are compatibility targets until strict
device evidence is archived. For CUDA, the historical broad benchmark used an
NVIDIA L40S (`sm_89`), and a current numerical diagnostic executed on an RTX
5060 Ti (`sm_120`). User-selected `FASTEMBEDR_CUDA_ARCH` values and linked
library architecture lists remain build-level claims until the corresponding
GPU executes the strict workflow; scientific performance requires the full
benchmark separately.

## Evidence Contract

Each hardware run records:

- the exact Git commit;
- UTC timestamp and backend requested;
- `sessionInfo()` and returned backend metadata from strict smoke calls;
- whether the requested backend was available;
- the evidence class, explicitly marked as hardware smoke/correctness rather
  than a full performance benchmark;
- the backend actually recorded by PCA, KNN, one-call and precomputed-KNN
  UMAP/t-SNE, held-out landmark projection, and Leiden results;
- elapsed smoke-test times;
- the complete installed-package `testthat` results; and
- SHA-256 identities for the Git archive, source package, installed shared
  library, benchmark results, session information, hardware report, and logs.

From the GitHub Actions page, select **Real hardware validation**, choose
`all`, and dispatch the workflow. Each successful job uploads a commit-named
artifact such as `fastEmbedR-hardware-cuda-<commit>`. The same validation can
be run directly on a prepared hardware runner:

```sh
bash .github/scripts/run-hardware-validation.sh metal validation-artifacts/metal
```

```sh
bash .github/scripts/run-hardware-validation.sh cuda validation-artifacts/cuda
```

The command fails if the requested accelerator is unavailable, if a result
records a different backend, or if a backend-specific expectation fails. This
prevents a CPU fallback from being archived as GPU evidence.

## Archived Results

The workflow artifact and the corresponding long-term copy in the benchmark
archive must refer to the same clean Git commit. Evidence from a dirty working
tree, a different package version, or a run that skipped the requested backend
is not release evidence. GitHub artifact retention is finite, so accepted
release artifacts are copied without modification to the benchmark archive;
the included `SHA256SUMS` file makes that copy verifiable.

See [the validation evidence schema](validation/README.md) for the required
files and release checklist.

Because GPU optimizers use asynchronous atomic updates, a fixed seed controls
the package random streams but does not guarantee bitwise-identical layouts
across runs or devices. Validation therefore checks backend identity,
objective behavior, neighborhood agreement, and numerical tolerances rather
than byte-for-byte coordinate identity.
