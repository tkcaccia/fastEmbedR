# Backend Performance Engineering

[Home](../README.md) |
[Installation](installation.md) |
[Implementation](implementation.md) |
**Performance Engineering** |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
[Reproducibility](reproducibility.md) |
[References](references.md)

This document records how the native CPU, Apple Metal, and NVIDIA CUDA
implementations are organized and how they are optimized. It separates three
different statements:

- **Implemented and retained** means the code is in the current package and
  passed runtime and embedding-agreement checks.
- **Tested and removed** means the experiment was measured but did not improve
  the accepted speed-quality result.
- **Proposed** means the change has not yet passed the acceptance protocol and
  is not claimed as a package feature.

This distinction matters because a faster dimensional-reduction kernel is not
an improvement if it changes the scientific interpretation of the embedding.

## Performance Acceptance Protocol

Backend engineering is evaluated without changing the mathematical workload.
For the optimization results in this document, the following were held fixed:

- input observations and supplied KNN graph;
- distance metric, k, and perplexity;
- PCA initialization and random seed;
- early exaggeration and normal optimization iteration counts;
- learning-rate, momentum, adaptive-gain, clipping, and centering rules;
- FFT-grid resolution;
- UMAP graph mode, epochs, negative-sampling policy, and objective.

A change is retained only when it satisfies all applicable gates:

1. repeated elapsed time is lower than the unmodified implementation;
2. CPU coordinates are identical when the operation order is intended to be
   deterministic;
3. parallel GPU output remains inside the variability of repeated atomic GPU
   runs;
4. coordinate correlation, Procrustes-aligned RMSD, and neighborhood overlap
   do not identify a material regression;
5. the label-colored layout is inspected visually on the full dataset;
6. a second, structurally different dataset is used as a regression check.

No change below obtains speed by reducing k, perplexity, epochs, grid size, or
the number of optimization iterations.

## Component Ownership And Data Flow

The public one-call functions combine nearest-neighbor search and embedding;
the KNN-input functions start at an already computed graph. The current
ownership boundaries are:

| Stage | CPU | Metal | CUDA |
| --- | --- | --- | --- |
| One-call KNN | Native float32 HNSW | Native exact or recall-tuned IVF-Flat | cuVS exact or IVF-Flat; optional FAISS GPU exact |
| Reusable host KNN | plain `indices`/`distances` list | Same KNN-input API | Same host KNN-input API; one-call native KNN can remain device-resident |
| PCA/t-SNE initialization | Native rSVD using BLAS-backed products | Float32 block-subspace rSVD using MPS matrix products | Automatic native rSVD or RAPIDS RAFT TSVD |
| t-SNE affinities | Native sparse C++ construction | Host construction followed by one graph upload | Native CUDA construction for resident one-call KNN |
| UMAP graph | Native compact sparse graph | Prepared sparse graph uploaded once | Native CUDA construction for resident one-call KNN |
| t-SNE optimization | Native C++ FFT-grid | Native Objective-C++/Metal FFT-grid | Native CUDA/cuFFT FFT-grid |
| UMAP optimization | Native C++ stochastic optimizer | Native Metal atomic in-place optimizer | Native CUDA atomic optimizer |
| Returned data | Final R layout | Final layout copied after optimization | Final layout copied after optimization |

## Compiler-Controlled Performance

All retained CPU results use the compiler and release flags selected by the
active R installation. fastEmbedR adds `-pthread` but does not replace
`CXX17FLAGS`. This distinction mattered in HNSW diagnosis: the same source
inside the validated Conda/R environment was substantially faster than a
generic system R/GCC build on the same remote machine. That difference was a
toolchain comparison, not an algorithmic speedup, and is not mixed into the
tables below.

An experimental host build with `-O3 -march=native` was slower for the
MNIST70k HNSW workload and was discarded. Global `-ffast-math` was not tested
as an accepted route because relaxed comparisons can change graph ties and
embedding trajectories. Publication comparisons therefore compile competing
R methods in the same environment and archive `R CMD config CXX17`,
`CXX17FLAGS`, compiler versions, and the generated package `src/Makevars`.

CUDA packages must also distinguish source flags from dependency
architectures. `FASTEMBEDR_CUDA_ARCH="75 89"` creates package kernels for a T4
and L40S, but linked FAISS/cuVS/RAFT libraries must independently contain
compatible kernels. A CUDA build made only for the build-host GPU is not a
valid multi-machine benchmark artifact.

The Metal t-SNE optimizer is GPU-resident after sparse graph construction:
layout, gains, updates, grid masses, FFT spectra, interpolated fields, and
reduction buffers persist on the GPU for the complete optimization. The graph
and affinities are currently prepared before the upload. CUDA one-call paths
can additionally consume the package-owned device KNN buffers without creating
R index and distance matrices.

## Native t-SNE Pipeline

All three backends implement the same high-level optimizer [1,3-5]:

1. Binary search converts KNN distances into conditional probabilities for the
   requested perplexity.
2. Sparse probabilities are symmetrized and normalized.
3. The supplied or PCA-derived two-dimensional layout is initialized at the
   t-SNE scale.
4. Sparse attractive forces are evaluated from the affinity graph.
5. Repulsive forces are approximated on a regular two-dimensional grid:
   points are splatted bilinearly, kernel fields are convolved by FFT, and the
   fields are interpolated back to the observations.
6. Early-exaggeration and normal phases update adaptive gains, momentum, and
   coordinates, clip steps, and recenter the layout.

The term **interpolation-based t-SNE** refers only to algorithmic lineage: the sparse
affinity, two-phase optimization, interpolation/FFT repulsion, and
fixed-reference workflow. It is not a claim of Python API, object, default, or
coordinate compatibility. The public CPU, Metal, and CUDA functions do not
call Python openTSNE or use `reticulate`.

### CPU t-SNE

The CPU path in `src/tsne_neighbors.cpp` stores float32 affinities and optimizer
buffers for float32 input. Its FFT-grid implementation uses a padded radix-2
two-dimensional transform. Rows and columns are transformed independently;
grid kernels represent the t-SNE normalization and repulsive fields.

The retained CPU optimizations are implementation-only changes:

1. **Reusable worker team.** A scoped worker executor is created once per
   t-SNE or transform call. Repeated FFT support operations no longer create
   and join new `std::thread` objects.
2. **Reusable FFT plan.** Bit-reversal maps and forward/inverse stage roots are
   computed once for a grid size and reused. The butterfly order is unchanged.
3. **Reusable column scratch.** Each worker receives persistent column storage
   instead of allocating a vector during every two-dimensional FFT.
4. **Parallel independent support arrays.** Grid padding, result extraction,
   kernel construction, and pointwise spectrum multiplication are partitioned
   across the same worker team.
5. **Stable accumulation order.** Point splatting was deliberately left in its
   original order because parallel writes would alter floating-point
   accumulation and visual reproducibility.

The same executor is also used by `transform_tsne()`, where fixed-reference
query transformations are independent by query row.

### Metal t-SNE

The Metal path in `src/embedding_metal_impl.mm` is native Objective-C++ and
Metal Shading Language. It uses float32 buffers and a package-native Stockham
FFT. Private GPU buffers hold masses, spectra, kernels, interpolated fields,
layout, gains, updates, and reductions. No layout is copied to the host inside
the normal optimization loop.

The retained Metal optimization increases the number of unchanged optimizer
iterations encoded in one command buffer from 4 to 16. This reduces command
submission and synchronization overhead while preserving the exact kernel
sequence and parameters. Error checking occurs when a completed batch is
flushed; diagnostic runs keep their finer synchronization and timing path.

Metal stage timestamps for 10 diagnostic iterations on an Apple M3 showed:

| Stage | GPU time, seconds | Share of measured GPU stage time |
| --- | ---: | ---: |
| Layout bounds/statistics | 0.00178 | 3.4% |
| Clear, splat, and load | 0.00564 | 10.9% |
| Forward FFTs | 0.01378 | 26.6% |
| Spectral convolution/inverse FFTs | 0.01314 | 25.4% |
| Normalization reduction | 0.00149 | 2.9% |
| Attractive/repulsive update | 0.00783 | 15.1% |
| Centering | 0.00815 | 15.7% |

The FFT family is therefore the main Metal optimizer cost, but replacing code
is accepted only when end-to-end repeated timing improves.

### CUDA t-SNE

The CUDA path in `src/embedding_cuda_kernels.cpp` keeps float32 layout,
affinities, optimizer state, grid fields, and cuFFT work buffers on the selected
device. cuFFT plans and work arrays are created once per run. Native one-call
KNN may remain device-resident through graph construction, so only the final
layout needs to cross the R boundary.

The retained CUDA optimization captures 25 unchanged iterations in each CUDA
Graph. Separate graph executables represent the early-exaggeration phase, the
normal phase, and any shorter tail. This reduces repeated host launch overhead
without fusing or reordering the mathematical operations inside an iteration.
The validated complex-to-complex cuFFT formulation remains in use.

## Native UMAP Pipeline

UMAP consumes a KNN graph and follows two explicit graph definitions [7,13]:

- `graph_mode = "fuzzy"` estimates local `rho` and `sigma`, converts directed
  distances to membership strengths, and forms the fuzzy symmetric union;
- `graph_mode = "binary"` assigns binary neighbor edges before applying the
  same low-dimensional optimizer.

The CPU path builds compact sparse graph arrays and precomputes sampling
schedules. The Metal and CUDA paths upload or construct the graph once, retain
the float32 layout and optimizer state on device, generate negative samples in
their native kernels, and return only the final coordinates. The validated
Metal implementation is the atomic in-place optimizer; the CUDA implementation
uses the package-native atomic optimizer. This performance pass did not change
the UMAP objective or scheduler. Full-MNIST smoke timings are reported below
to demonstrate that the t-SNE changes did not regress UMAP.

## PCA Initialization

`tsne()` uses a small PCA initialization unless an explicit `Y_init` is
supplied. The decomposition uses fastEmbedR's package-native randomized-SVD
implementation rather than IRLBA:

- CPU uses blocked float32 randomized subspace products, with Accelerate SGEMM
  on Apple builds and threaded package kernels elsewhere;
- Metal uses a resident float32 block-subspace iteration with MPS matrix
  multiplication and a small CPU eigensolve of the projected Gram matrix.
  Float32 input is copied directly into unified Metal storage; a native Metal
  reduction validates, centers, and optionally scales every feature in place;
- CUDA selects package-native rSVD for sufficiently wide, low-rank matrices
  and RAPIDS RAFT TSVD otherwise.

Scores are centered and rescaled so the largest component standard deviation
is `1e-4`. PCA timing is not included in the embedding-only results below;
the same precomputed initialization was supplied to every baseline and
optimized run.

The CPU and Metal implementations were also compared directly on all 70,000
flattened MNIST observations using float32 input, two components, centering,
no feature scaling, seed 4, and three repetitions:

| Backend | Median PCA time, s | Precision | Engine |
| --- | ---: | --- | --- |
| CPU, 1 core, current warm path | 0.213 | float32 | Native blocked rSVD |
| CPU, 4 cores, current warm path | 0.161 | float32 | Native blocked rSVD |
| Metal, current warm path | 0.096 | float32 | Resident preprocessing plus MPS rSVD |

The current warm Metal path is 2.22 times faster than the one-core CPU path and
1.68 times faster than the four-core CPU path in this float32 MNIST test. Its
first invocation also compiles the Metal pipeline; cold and warm timings must
therefore be reported separately. Moving preprocessing to Metal reduced host
work, while the optimized CPU route now also centers and sketches once in
contiguous float32 storage. Both implementations return actual
`float::float32` score and loading matrices for float32 input instead of
constructing double R matrices and attaching float metadata. The 70,000 by 2
score object occupies approximately 0.56 MB rather than the approximately
1.12 MB double payload.

Against the CPU rSVD reference, the current Metal loading subspace had minimum
canonical correlation 0.9980 (maximum principal angle 3.65 degrees), while
pairwise distances between a deterministic sample of 2,000 score rows had
Pearson correlation 0.9977. Three full t-SNE runs initialized by this path
had coordinate correlation 0.99897-0.99909 and normalized Procrustes RMSD
0.0426-0.0454 against the accepted Metal baseline. Their embedding times were
2.568-2.621 seconds and the label-colored full-MNIST layout passed visual
inspection. The small-data test suite also compares Metal scores against
`stats::prcomp()` and requires correlation of at least 0.99 for both
components.

The CUDA implementation automatically selects a native float32 rSVD or RAFT
TSVD path. The selector accounts for matrix size, width, target rank, sketch
width, and estimated rSVD pass cost. It therefore avoids the large covariance
cost for very wide, low-rank inputs while retaining TSVD for narrower or
less-truncated problems. CUDA requests are not silently replaced by CPU PCA.
The CPU/Metal comparison files are
stored under `results/cpu_metal_optimization/pca_validation/`.

## Landmark Projection And Transform

Landmark workflows embed a reference subset and then transform the remaining
observations against a fixed reference layout. The implementation keeps this
approximation explicit:

- query-to-landmark affinities are computed once;
- CPU query transformations are partitioned across the reusable worker team;
- fixed-reference layout and query batches use compact contiguous buffers;
- native Metal/CUDA transforms keep reference coordinates and optimizer state
  on device where the backend supports the operation;
- projection and transform times are reported separately in landmark
  benchmarks.

Full UMAP and t-SNE never switch to landmarking silently.

## Measured Improvements

### Benchmark protocol

The primary optimization benchmark used all 70,000 MNIST observations and all
784 flattened pixel variables. KNN (`k = 15`) and the two-dimensional PCA
initialization were cached and reused, so the table measures embedding only.
The random seed was 4, CPU used 4 threads, and the normal public t-SNE
iteration schedule was unchanged. The local system was an Apple M3 with 8
logical CPU cores and a 10-core Metal GPU. Medians are used because individual
wall-clock measurements include ordinary system noise.

| Backend | Baseline median, s | Retained median, s | Improvement | Repetitions | Agreement with baseline |
| --- | ---: | ---: | ---: | ---: | --- |
| CPU t-SNE | 16.259 | 14.233 | 12.5% | 3 baseline, 5 retained | Coordinate correlation 1.000; Procrustes RMSD approximately 0; overlap@15 1.000 |
| Metal t-SNE | 2.602 | 2.545 | 2.2% | 3 baseline, 5 retained | Correlation 0.99909-0.99967; normalized Procrustes RMSD 0.0257-0.0426; overlap@15 0.8816-0.9196 |

Repeated unmodified Metal runs already showed correlation
`0.99925-0.99931` and normalized Procrustes RMSD `0.037-0.039`. The retained
Metal result is therefore inside the expected variability of parallel atomic
updates. The accepted full-data comparison plot is stored at
`results/cpu_metal_optimization/final_mnist70k/accepted_layouts.png`.

The same final build produced the following UMAP regression timings from the
cached MNIST KNN: 3.290 seconds for CPU fuzzy UMAP and 0.978 seconds for Metal
fuzzy UMAP. These are smoke measurements, not a claim that the UMAP algorithm
was optimized in this pass.

### CPU thread scaling

The retained build was measured separately with 1, 2, 4, and 8 CPU threads.
The machine has four performance and four efficiency cores. KNN and PCA were
cached, and every row used the same MNIST70k graph, initialization, seed, and
optimizer settings:

| Threads | t-SNE embedding, s | t-SNE speedup | Fuzzy UMAP embedding, s | UMAP speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 46.988 | 1.00x | 13.151 | 1.00x |
| 2 | 25.636 | 1.83x | 6.030 | 2.18x |
| 4 | 14.304 | 3.28x | 3.299 | 3.99x |
| 8 | 12.510 | 3.76x | 3.335 | 3.94x |

The multicore implementation therefore performs well through four threads.
UMAP is nearly linear to the four performance cores. t-SNE retains serial
or memory-bound grid work and reaches 3.28x at four threads. Adding the four
efficiency cores gives only a small t-SNE improvement and no UMAP benefit,
so four threads are the sensible default on this Apple M3. Raw rows are stored
in `results/cpu_metal_optimization/final_cpu_scaling/`.

On MetRef, baseline and retained CPU/Metal layouts were exactly equal after the
changes (coordinate correlation 1.0). Single-run elapsed times were 0.581 to
0.531 seconds on CPU and 1.200 to 0.954 seconds on Metal. Because this dataset
is small and timing variance is proportionally large, these values are used as
a regression check rather than a speed claim. The visual comparison is stored
at `results/cpu_metal_optimization/metref/comparison.png`.

### CUDA graph result

The CUDA optimization was measured separately on an NVIDIA T4 with full
MNIST70k flattened input. With the same KNN and optimization mathematics,
capturing 25 iterations per CUDA Graph changed median t-SNE embedding time
from 3.293 to 2.534 seconds (23.1%) and median full-call time from 5.234 to
4.534 seconds (13.4%). Agreement with the accepted baseline was:

- coordinate correlation: 0.999691;
- normalized Procrustes RMSD: 0.024861;
- neighborhood overlap@15: 0.903091;
- sampled distance Pearson/Spearman correlations: 0.999287/0.999268.

Baseline-to-baseline CUDA overlap@15 was 0.905777, confirming that the retained
result is inside normal atomic-run variability. The final one-off full-call
run required 4.897 seconds: 2.088 seconds for KNN and 2.582 seconds for
embedding. The corresponding CUDA UMAP runs were 2.317 seconds (fuzzy) and
2.307 seconds (binary).

## Experiments Tested And Removed

| Experiment | Backend | Result | Decision |
| --- | --- | --- | --- |
| Real-to-complex/complex-to-real FFT conversion | CUDA t-SNE | Only marginal timing change and weaker local agreement than the accepted C2C path | Removed; retain C2C cuFFT |
| Cached Stockham twiddle table | Metal t-SNE | Warm median 2.596 s versus 2.576 s for the simpler batching candidate | Removed |
| Five-channel batched forward FFT | Metal t-SNE | Median 2.629 s; extra occupancy/scratch traffic outweighed fewer dispatches | Removed |
| Parallel point splatting with private grids | CPU t-SNE | Full-MNIST median 14.897 s versus 14.233 s; reduction traffic exceeded the parallel splat gain | Removed |
| Fused post-update centering | Metal t-SNE | Full-MNIST median 2.613 s versus 2.545 s | Removed |
| GPU orthonormalization of the small PCA basis | Metal PCA | Median 0.449 s versus 0.429 s; extra dispatch exceeded the tiny CPU QR cost | Removed |

Failed experiments are not exposed as parameters. Keeping them out of the
public implementation reduces maintenance, branching, and accidental use of a
slower or less faithful path.

## Remaining Profile-Guided Work

The following work is proposed, not yet claimed:

1. **Metal FFT and centering.** Profile the accepted 16-iteration command batch
   in Xcode GPU Capture. A replacement should target the FFT or centering
   stages together and must beat the existing Stockham path end to end; merely
   reducing dispatch count has already failed once.
2. **Metal sparse attraction.** Reorder CSR rows and edge values for more
   coalesced reads while preserving each row's accumulation order. This may
   reduce the 15% attractive/update share without changing the objective.
3. **CPU platform FFT evaluation.** Compare the package plan against an
   Accelerate/FFTW abstraction only if CRAN portability, licensing, and exact
   output checks can be maintained. A platform-specific dependency is not
   justified by microbenchmarks alone.
4. **CPU sparse-force locality.** Pack affinity rows by degree and reuse
   per-thread gradient scratch to reduce cache misses, while retaining the
   current deterministic row partition.
5. **CUDA graph/affinity primitives.** Replace custom scans and reductions with
   CUB or RAFT only after an isolated profile demonstrates that graph setup,
   rather than KNN or optimization, is limiting the full call.
6. **CUDA KNN batching.** Tune exact-search batches and IVF workspace reuse on
   the target GPU, with recall measured against an exact subset. KNN remains a
   larger fraction of the full CUDA call than the optimized embedding.

## Reproducing The Measurements

The reusable scripts below are maintained in
[`fastEmbedR-benchmark`](https://github.com/tkcaccia/fastEmbedR-benchmark).
Clone that repository and run the MNIST benchmark command from its root:

```bash
git clone https://github.com/tkcaccia/fastEmbedR-benchmark.git
cd fastEmbedR-benchmark
Rscript tools/benchmark_mnist70k_embedding_only_scaling.R \
  --data=/Users/stefano/Documents/fastEmbedR/Data/MNIST/MNIST.RData \
  --cache-dir=results/mnist70k_embedding_only_scaling_cache \
  --threads=4 \
  --backends=cpu,metal \
  --k=15 \
  --perplexity=15 \
  --out-dir=results/cpu_metal_optimization/reproduction
```

The Metal stage profiler is:

```bash
Rscript tools/profile_metal_opentsne_fft.R \
  --data=/Users/stefano/Documents/fastEmbedR/Data/MNIST/MNIST.RData \
  --cache-dir=results/mnist70k_embedding_only_scaling_cache \
  --iterations=10 \
  --out-dir=results/cpu_metal_optimization/profile_reproduction
```

Raw accepted measurements are retained in:

- `results/cpu_metal_optimization/baseline/timing.csv`;
- `results/cpu_metal_optimization/final_mnist70k/timing.csv`;
- `results/cpu_metal_optimization/final_mnist70k/layout_agreement.csv`;
- `results/cpu_metal_optimization/baseline/metal_stage_10iter.csv`;
- `results/cuda_opentsne_r2c/timing_c2c_baseline.csv`;
- `results/cuda_opentsne_r2c/timing_c2c_chunk.csv`;
- `results/cuda_opentsne_r2c/layout_c2c_chunk_agreement.csv`.

See [Reproducibility](reproducibility.md) for the full system, session, seed,
and benchmark manifest requirements.

## Licensing And Attribution

The retained CPU worker/FFT engineering and command batching are package-local
implementations. The algorithmic design was informed by t-SNE, FIt-SNE,
openTSNE, t-SNE-CUDA, UMAP, FAISS, RAPIDS cuVS/RAFT, and Apple GPU FFT work
[1,3-9,12-13]. No GPL benchmark implementation is copied or linked into the
MIT package. Exact source pins and permissive notices for adapted or closely
studied software are retained in `inst/NOTICE`, `inst/LICENSES/`, and
`inst/ALGORITHMIC_REFERENCES.md`.

The numbered bibliography is in [References](references.md).
