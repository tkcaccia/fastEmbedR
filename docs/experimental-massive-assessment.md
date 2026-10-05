# EXPERIMENTAL massive-data architecture assessment

This assessment describes the current in-memory boundary and the first
implemented out-of-core route. It does not claim that the complete massive
backend exists.

1. Matrix inputs enter public `umap()`, `tsne()`, and `pca()` as resident R
   matrices or float32 objects. Embedding preprocessing in `R/embed.R` and
   `R/umap.R` assumes all rows are addressable.
2. CPU PCA in `src/embedding_utils.cpp` centers/scales into a full float32
   buffer, builds QR/SVD intermediates, and returns full score matrices.
3. `R/embed.R` converts between R double and float32 for several routes;
   `pca_rsvd_cpu_cpp()` copies input and intermediate matrices across R/C++.
4. Existing PCA uses randomized matrix multiplication in blocks, but its
   source matrix and scores remain resident. It is not out-of-core PCA.
5. Native CPU and CUDA KNN have batched query/search loops, but their
   reference matrices and result objects are resident.
6. `src/native_knn_cuda_impl.cpp` uploads the full reference for cuVS query
   search; the existing CUDA PCA and embedding paths also allocate resident
   feature, KNN, or layout buffers.
7. `R/graph.R` materializes KNN indices and distances before converting
   them into in-memory `from`, `to`, and `weight` graph vectors.
8. `src/tsne_neighbors.cpp` stores sparse affinity arrays and full layout,
   gradient, update, and gain vectors in memory. FFT workspaces are additional.
9. The public random-walk method is `method = "walktrap"`. Its native
   implementation in `src/walktrap.cpp` receives complete edge vectors.
10. Native Louvain in `src/graph_clustering.cpp` builds resident adjacency
    and contracted graphs from complete edge vectors.
11. Native Leiden shares that input boundary and adds resident refinement
    and hierarchy state.
12. A row-partitioned CSR graph with 64-bit offsets and bounded edge blocks
    could replace edge-vector input for traversal. The present clustering
    code cannot consume it without storage/traversal adaptation.
13. Existing CPU threading, CUDA streams, cuVS search, native graph builders,
    and landmark projection kernels are reusable after their inputs become
    block-accessible. No additional ANN library is required for phase one.
14. A 64-bit `MatrixSource` reader and float32 file sink now support CPU
    and CUDA streamed covariance PCA with file-backed scores. CUDA
    covariance and projection use reusable device buffers; the small
    covariance eigendecomposition runs on CPU. Reservoir selection
    writes a landmark file; a persistent CPU HNSW index or exact reference
    search produces disk-backed query-to-landmark KNN files. CPU landmark
    projection now reads that graph in bounded batches and writes UMAP or
    t-SNE coordinates to another float32 file. The existing Walktrap,
    Louvain, and Leiden methods cluster a resident landmark graph, and
    streamed weighted votes write full-data label/confidence files. The
    experimental CUDA exact and IVF-Flat KNN routes retain one cuVS reference
    index across bounded query batches. Explicit sharded-exact
    landmark search instead builds bounded cuVS reference shards and merges
    exact candidates in disk-backed output. Disjoint query ranges can run
    on different CUDA devices. The multi-GPU KNN route has not been
    validated on a two-GPU host. Repeated full query scans make
    it a correctness route, not a billion-row ANN solution. The single-device
    route checkpoints every 100 query blocks and at each shard boundary;
    resume verifies both inputs and a rebuilt-index pilot. The multi-GPU
    route now checkpoints each shard and the merge, and verifies its plan
    and saved shard identities before reuse. IVF-Flat calibrates on the
    first batch using distance-at-k recall and also reports strict ID
    overlap. Equivalent tied neighbors may have different IDs. Neither
    pilot measure certifies every query. Before allocating the index,
    a conservative estimate limits
    the query batch to 65% of currently free VRAM, including an index and
    workspace allowance. This is an estimate, not a cuVS workspace guarantee.
    The existing CUDA projection kernel
    consumes saved UMAP KNN batches for zero-epoch projection.
    CPU full-graph UMAP streams a symmetric fuzzy graph while retaining
    the float32 layout in RAM or writable mmap. CUDA full-graph UMAP
    streams bounded graph-edge batches using the existing atomic
    optimizer kernels for 2D or 3D while retaining the default layout on
    one GPU; it rejects default requests whose layout cannot fit free VRAM.
    Explicit managed CUDA coordinates can page from host RAM on devices
    supporting concurrent managed access. This does not make coordinates
    disk-backed, and its over-VRAM speed and quality remain unverified.
    Completed CUDA
    epochs can now checkpoint a full coordinate download and resume on
    the same backend. Each snapshot costs an additional full-layout disk
    write and device-to-host transfer. Asynchronous atomics prevent
    bitwise trajectory equivalence across runs. An explicit CPU sharded
    HNSW route builds a full-data approximate KNN graph without a resident
    full-matrix index, but repeated passes make its billion-row
    performance unproven. CPU Louvain now has file-backed local-moving
    levels and external contraction. A higher level uses resident Louvain
    only after a conservative memory check. Other CUDA streaming stages
    remain incomplete.
    A synthetic 20-million-row, eight-neighbor ring graph passed bounded
    construction and five CUDA UMAP epochs. All output coordinates were
    checked through bounded reads. The graph exceeded a configured 1 GB
    working-buffer budget, but not the host's RAM or the GPU's VRAM.
    The five-epoch run used an earlier dirty source archive. A fresh
    archive separately confirmed that one-epoch requests now fail at
    public and native CUDA boundaries rather than silently leaving the
    layout unchanged. The fresh build also passed a five-epoch refit on
    the saved 20-million-row graph. External sampling observed up to
    300 MiB for its compute process, but could miss shorter allocation
    peaks. These are functional checks on an easy synthetic graph, not
    quality or speed evidence for real massive datasets.
    A newer dirty-source build also completed a 100-million-row,
    eight-neighbor synthetic ring graph: 800 million directed and fuzzy
    edges, two CUDA UMAP epochs, and bounded finite-value checks of every
    output coordinate. The optimizer reported CUDA and took 43.546 seconds;
    the complete graph-generation and fitting trace spanned about
    9 minutes 42 seconds. R peak RSS was 679,108 KiB, and external
    sampling observed at most 910 MiB for the compute process. This
    confirms bounded host-memory construction at that scale, not
    billion-row ANN, real-data quality, or a GPU layout larger than VRAM.
    Source and measurement identities are in the `fastEmbedR-extra`
    massive-data evidence directory.

CPU full-graph t-SNE now accepts the saved symmetric compact-support
affinities and a file-backed 2D or 3D initialization. It streams CSR
attraction each iteration and reuses the resident CPU FFT repulsion and
optimizer update. In small fixed-input tests, five-step 2D and 3D layouts
match the in-memory optimizer within 1e-4. This is correctness evidence for
the new storage boundary, not a large-graph speed or memory result. The
shared CSR stream reader now reads contiguous bounded edge blocks across
row boundaries rather than seeking separately for every row. It preserves
strict row validation and edge order; the before/after diagnostic is in
`fastEmbedR-extra/benchmarks/macos/massive-data/evidence/`. The coordinate,
gradient, update, and gain arrays still consume RAM; the
resource preflight rejects requests that exceed its budget. The new
full-graph CUDA route streams the same affinity graph in bounded batches,
reuses the package's 2D cuFFT repulsion and optimizer update, and retains
the coordinate state on one GPU. It rejects three-dimensional inputs and
layouts that exceed free VRAM; it does not provide over-VRAM optimizer
state. On one RTX 5060 Ti, a 32-row five-step fixture agreed with the CPU
FFT result within 5.83e-10 maximum absolute difference and reported the
CUDA backend. This is narrow functional evidence, not a large-graph
speed, memory, or embedding-quality result. CPU full-graph t-SNE can save
coordinates, momentum updates, and adaptive gains at completed iterations.
A forced interruption after iteration two resumed to a byte-identical
five-iteration layout; changed optimizer controls were rejected. Each
snapshot occupies three times the layout size, and this small fixture
does not establish recovery throughput on a large graph.
CUDA saves the same three state arrays. A forced interruption after
iteration two resumed to a byte-identical five-iteration CUDA layout on
a 16-row fixture. One larger synthetic checkpoint test is described below.
On the saved MNIST70k fixed-input graph, 25 CPU FFT iterations over
3.07 million affinity edges took 36.728 seconds with streamed attraction
and 31.983 seconds with resident affinities. The resulting coordinates
agreed within 1.46e-11 in maximum absolute difference. Separate macOS
processes peaked at 137 MB and 324 MB RSS, respectively, but the resident
process also loaded a larger original KNN object. These are diagnostic
timings on a dirty source checkout, not matched end-to-end speed or
optimizer-only memory claims; 25 iterations do not establish embedding
quality. The driver, source hashes, and logs are in `fastEmbedR-extra`.
The same CUDA graph-streaming route subsequently completed two
25-iteration fits from a saved 10-million-row compact-affinity graph
with 311,818,176 directed edges. That graph was derived earlier from
a physical 10M-by-1,024 float32 input larger than the host's 31 GiB
RAM. Both fits used the same chunk-generated initialization and a
1,024-cell FFT grid. With learning rate 1, the public call took
32.729 seconds but barely moved the coordinates. With the package's
automatic rate, it took 32.942 seconds and produced finite coordinates
spanning approximately -19.4 to 19.4. Every saved row was scanned in
bounded reads. Peak process RSS was 628,068 KiB; sampled GPU memory
rose by at most 852 MiB over its pre-run baseline. This validates a
large disk-backed affinity boundary and nontrivial short-run updates,
not convergence or quality on real data. The layout and FFT state still
fit in VRAM. Exact identities and raw measurements are in the
`distinct10m-full-tsne-cuda-20261004` evidence directory of
`fastEmbedR-extra`.
On the original MNIST70k input, the streamed CUDA t-SNE route and the
resident CUDA `tsne_knn()` route each completed 250 early-exaggeration
plus 750 normal iterations from the same initialization and 30-neighbor
KNN. Both reported a 512-cell FFT grid. Their final coordinates had
Procrustes correlation 0.998706 on the same 2,000 evaluation rows.
Streamed versus resident trustworthiness was 0.941693 versus 0.941562;
Preserve@30 was 0.436983 versus 0.437950. All 70,000 coordinates were
finite, and both routes reported CUDA. This is a real-data long-run
quality check, but not a matched speed comparison: the streamed call
read saved affinities from disk for each iteration, whereas the
resident call built affinities from KNN input and kept them in GPU
memory. Their respective public-call times were 13.017 and 1.356
seconds in one sequential, unwarmed process. The driver, source archive,
saved graph, coordinates, raw quality table, and resource measurements
are in `mnist70k-full-tsne-quality-20261004` under `fastEmbedR-extra`.
On that same 70,000-row graph, two runs per existing graph-access mode
found a median CUDA call time of 10.199 seconds with `mmap` and 12.480
seconds with `stream`. Peak process RSS remained near 569 MB, and a
fixed 2,000-row quality check showed similar trustworthiness and
neighborhood preservation across modes. The saved graph is only about
25 MB and likely fits the host page cache; this is not evidence of an
equivalent gain when the graph exceeds RAM. Returned stage timings are
host-call wall times. In particular, CUDA may run queued FFT work
during graph traversal, and the attraction synchronization may include
that work. The raw paired results and scorer are in
`mnist70k-tsne-graph-access-20261004` under `fastEmbedR-extra`.

An independent storage stress test used a synthetic symmetric 50-neighbor
ring affinity graph with 100 million rows and 5 billion directed edges.
Its 40.8 GB CSR files exceeded Chiamaka's 31 GiB physical RAM and
16,311 MiB GPU memory, and the edge count crossed the 2^32 offset.
The experimental CUDA t-SNE route completed two streamed iterations and
scanned every output row for finite coordinates. A second run with its
automatic learning rate changed all 200 million coordinate values;
an independent bounded verifier checked the graph rows surrounding the
2^32-edge boundary. The complete second command took 300.10 seconds,
including graph validation, optimizer execution, and output scanning;
peak process RSS was 629,292 KiB and sampled GPU-process memory was
4,266 MiB. This proves a larger-than-RAM graph access path on one GPU,
not quality, convergence, or a scalable real-data KNN graph builder.
Source identities and raw evidence are in `ring100m-overram-20261004`
under `fastEmbedR-extra`.
A fault-injection follow-up on that same graph committed a 2.4 GB CUDA
optimizer checkpoint after iteration one, stopped, and resumed in a new
R process for iteration two. All 100 million output rows were finite;
the maximum absolute difference from the uninterrupted automatic-rate
fit was 4.66e-10 over all 200 million coordinate values. Each process
revalidated the full graph, so the complete interrupt and resume commands
took 210.17 and 216.82 seconds. This tests one deliberate interruption
after a committed snapshot, not arbitrary power-loss recovery.

The experimental `.f32` and `.fbin` readers require little-endian float32
rows. The output is a raw `.f32` file. The PCA path performs a mean pass,
covariance pass, and score-projection pass; for CPU `p <= 1024` and CUDA
`p <= 2048` it retains bounded row blocks and `O(p^2)` statistics.
On one RTX 5060 Ti, a strict-CUDA install of the frozen fastEmbedR 0.1
archive with SHA-256
`327eb45d6c667819b2c8bcf58f786947b76243175c4cc6d64558ac03dd74a302`
completed rank-2 PCA on a physical 7,000,000-by-1,200 float32 input
(33.6 GB), larger than host RAM and VRAM. The full call took 222.976
seconds, reached 1,609,084 KiB peak observed process RSS, and the sampled
compute-process GPU peak was 852 MiB. Every saved score row was checked
against the repeated-block projection; maximum absolute error was
1.85e-5. This is one synthetic functional run, not a speed comparison.
The source, image, and input hashes and raw memory samples are in
`fastEmbedR-extra/benchmarks/linux/massive-data/evidence/`.
Using the same physical input, a later source-locked run reopened PCA
scores and completed CUDA landmark UMAP and Leiden with the saved KNN
graph. All 7 million output rows passed bounded validity scans; PCA,
UMAP, and clustering took 301.668, 3.544, and 1.452 seconds, with a
869,264 KiB maximum process RSS. This synthetic repeated-block test
demonstrates the pipeline's storage boundary, not real-data quality.
The original rank-30 scores from this repeated input also completed a
full CUDA IVF-sharded graph after the tied-neighbor pilot was corrected.
At k=30, its 256-query pilot had distance-at-k recall 1.0 but strict
ID overlap 0.226; all 7 million saved graph rows passed validation.
The graph build took 226.274 seconds with 1,001,500 KiB peak host RSS.
Fuzzy UMAP graph construction produced 410.7 million directed edges,
and a five-epoch streamed CUDA optimizer check completed with finite
coordinates. These synthetic runs do not establish real-data quality;
their identities and memory samples are in the `repeated-fullgraph-7m-20261004`
evidence directory of `fastEmbedR-extra`.
On a separate non-repeated 10-million-by-1,024 synthetic input (40.96 GB,
larger than host RAM), the same strict-CUDA build completed streamed
rank-30 PCA, five-shard cuVS IVF-Flat kNN, fuzzy graph construction,
five CUDA UMAP epochs, full-graph Leiden, and compact t-SNE affinity
construction. A 256-row exact CPU audit found recall@30 of 1.0 for
every sampled graph row; unsampled rows remain unverified. Peak host RSS
was 1,031,724 KiB for the kNN stage, and sampled GPU process memory
was 2,062 MiB. This single synthetic run tests bounded-memory execution,
not converged embedding quality or full-data t-SNE optimization. The
archive and measurements are in `fastEmbedR-extra` under
`distinct-fullgraph-10m-20261004`.
A separate physical 100-million-by-96 float32 input (38.4 GB) passed
single-GPU streamed rank-30 PCA with a 12 GB file-backed score output.
All saved score rows were scanned for finite values, and three
128-row blocks agreed with independent projection within 7.83e-7.
The wrapper peaked at 789,416 KiB host RSS; sampled GPU process memory
reached 196 MiB. This tests large file offsets and bounded-memory PCA,
not a 100-million-row graph or embedding. See the `distinct-pca-100m-20261004`
evidence directory in `fastEmbedR-extra`.
An earlier checkpoint policy took 1,711.216 seconds for PCA on the
same input. It saved a large covariance state too often. Scaling the
checkpoint interval by state and input-block bytes cut filesystem
output from 13.11 GB to 1.70 GB while retaining recovery points.
The new PCA score file, saved KNN, and Leiden labels matched the old
run byte-for-byte. CUDA UMAP coordinates differed, as expected from
asynchronous updates. The matched no-checkpoint PCA took 224.965
seconds with the old source. Exact source and image identities and
logs are preserved in the same evidence directory.
Wider inputs use a seeded
randomized sketch with two streamed covariance-action passes and a small
projected eigendecomposition. The sketch buffers scale with
`O(p * (rank + 16) * workers)`, not `O(p^2)`. This is an approximation.
Completed PCA writes a model-and-score manifest. `massive_open_pca()`
checks the saved score-file identity and lets a later graph job reuse the
file-backed scores without rerunning the decomposition.
A two-process smoke test reopened CPU PCA scores and a fuzzy graph, then
ran memory-mapped full-graph UMAP and file-backed Louvain from that same
graph. A separate Chiamaka test reopened CUDA PCA scores and used the
resulting fuzzy graph for five CUDA UMAP epochs in 3D. These are
functional checks on small inputs, not large-data quality or speed evidence.
For wider CUDA inputs, the mean/scale pass runs on CPU, then the sketch
uses bounded GPU blocks before CUDA score projection. These fits report
`moments_backend` and `sketch_backend`. Dense-reference numerical checks
passed on one RTX 5060 Ti, as did a 100,000-by-1,200 file-backed smoke
test. Out-of-core CUDA PCA can project disjoint score rows with separate
workers on selected devices. Decomposition runs on the first device.
Optional checkpoints save that model and committed blocks for each shard;
resume validates the source, model, controls, and device list before
continuing projection or merging completed shards. Completed PCA and
landmark-projection shards retain small `.result.rds` manifests; resume
rejects a missing manifest or a changed shard file. An already completed
multi-device projection merge is compared byte-for-byte with its verified
shards before a resumed call accepts it. The same check applies to
completed merged CUDA KNN files. It scans the full output with bounded
buffers. For covariance PCA, completed mean and covariance blocks can be
reused after interruption; blocks since the last covariance checkpoint
repeat. Wide randomized PCA now saves
completed column moments before the first sketch action. An interrupted
sketch action still restarts that action.
One-device CUDA projection and a two-worker merge on the same GPU passed
focused tests. A committed partial shard resumed in a fresh worker
process and matched an uninterrupted projection. Projection across two
distinct A10G GPUs matched single-device PCA scores on a 4,096-row input;
two-device landmark projection matched CPU for 2D/3D UMAP and 2D t-SNE
on 39 query rows. A 10-million-row by 16-feature repeated-block PCA input
also gave byte-identical one- and two-GPU scores with 10,000-row batches;
the two-GPU call was slower (1.329 versus 1.071 seconds). These are
functional and memory checks, not multi-GPU acceleration evidence.
The public landmark-clustering workflow can now take a prior massive UMAP
or t-SNE fit as `nn`. It verifies the source, landmark selection, saved
query KNN, backend, and neighbor count before reusing the same KNN files.
It still builds the small resident graph among landmarks. This avoids a
second all-row KNN pass and second KNN file pair; it does not change the
clustering objective or improve approximate assignment quality.
Metal streaming is not implemented.
For landmark embeddings, the reported `peak_ram_bytes` and
`peak_vram_bytes` are the largest estimates across reference fitting,
KNN, and projection, not measured peaks or guaranteed ceilings. They
exclude R's baseline and CUDA library allocations. In one CUDA image, the
process peaked near
680 MB even for a small input; `memory_limit` is not a process RSS cap.
With opt-in checkpointing, covariance PCA saves completed mean and
covariance blocks;
wide randomized PCA saves completed column moments. Both save the fitted
decomposition before projection, and each flushed score block advances
a checkpoint. Resume reuses completed stages and overwrites any
uncommitted partial-file tail. Covariance snapshots occur at most every
16 blocks, or more often when the saved sums are large. An interrupted
wide moments or sketch pass still restarts that pass. Checkpoint format
version 4 does not resume older experimental PCA checkpoints.

The current native APIs use 64-bit row and byte counts, but R inspection
returns only bounded double matrices. `as.matrix()` refuses outputs over
128 MB. A sparse one-billion-row float32 fixture verifies last-row reads
through both streamed and mmap access and through a row view; it tests
file offsets, not a full scan or a billion-row computation. A complete
1B-row workflow additionally needs scalable full-data
ANN, resumable graph and layout stages, and validation on real accelerators.
On an 8 GB Apple M3, a CPU PCA check processed a deterministic virtual
100-million-by-32 float32 source (12.8 GB logical input) and wrote 800 MB
of file-backed scores. An independent closed-form mean, loading, and
sampled-score check passed; process RSS peaked below 90 MB. Because the
source generated rows on demand, this verifies bounded computation and
large offsets, not sustained disk I/O on a larger-than-RAM input. The
script, result CSV, and source identity are in `fastEmbedR-extra`.
An additional physically written, non-sparse 70-million-by-32 float32
input occupied 8.96 GB on the same 8 GB Mac. Four-worker streamed
covariance PCA wrote 560 MB of rank-two scores in 66.07 seconds, with
89,473,024 bytes peak process RSS. A bounded independent verifier read
all 70 million saved rows and checked centers, singular values, and
scores against the known repeated-block input. This establishes
physical beyond-RAM PCA I/O and output ordering, but the repeated
synthetic pattern does not validate performance or quality on varied
real data. Source and measurement hashes are in the `fastEmbedR-extra`
`physical-pca-over-ram-20261002` evidence directory.
On Chiamaka, a physical 100-million-by-96 float32 input and its
file-backed rank-30 PCA scores supported a CUDA landmark workflow with
100,000 landmarks. The strict-CUDA build selected landmarks, wrote a
100-million-row IVF-Flat query-to-landmark graph, fit fuzzy UMAP on the
landmarks, and projected every row into a file-backed 2D result. The graph
search took 192.89 seconds; projection took 125.85 seconds. Their peak
process RSS values were about 885 MB and 606 MB, respectively, while
sampled compute-process GPU memory stayed below 300 MB. All graph and
coordinate rows passed bounded validity scans. Exact reference search
gave recall@30 of 1.0 on 256 sampled rows, and sampled landmark
coordinates were preserved exactly. These are synthetic engineering
checks, not guarantees of unsampled recall or real-data embedding quality.
The immutable identities and raw stage logs are in the `fastEmbedR-extra`
`projection-float-fix-20261004` evidence directory.

Using the same saved 100-million-row PCA scores and query-to-100,000-
landmark graph, the same strict-CUDA build fitted a landmark t-SNE model
in 6.095 seconds and projected all 100 million rows in 881.273 seconds.
All coordinate rows were finite in a bounded scan, with 553,776 KiB
in-call peak RSS and 162 MiB sampled compute-process GPU memory during
projection. A CUDA-built 100,000-landmark graph and the existing CUDA
Leiden and Louvain implementations produced 186 and 185 communities;
streamed CPU weighted assignment covered all 100 million rows in 128.283
and 127.514 seconds. Whole-output Leiden/Louvain agreement was adjusted
Rand index 0.850 and normalized mutual information 0.962. This is method
agreement on synthetic data, not external clustering accuracy. The PCA
scores came from an earlier binary, so this is not a one-binary end-to-end
reproduction. The exact source, container, output hashes, stage boundaries,
and logs are in the `tsne-clustering-100m-20261004` evidence directory of
`fastEmbedR-extra`. Real-data landmark t-SNE quality remains a separate
gate; landmark projection is not a joint full-data t-SNE fit.

An external prototype in `fastEmbedR-extra` now physically groups a
70,000-row PCA source into 256 feature-space posting lists. Its selected-list
search reproduced the earlier exact-sample routing recall, but its
R-level query loop was slower than native full-source exact search at
this size. It does not provide a package ANN backend. In-list indexing,
batched queries, tail-recall controls, and recovery remain open work.
The package now has an EXPERIMENTAL two-pass physical posting builder for
file-backed float32 input and supplied centers. It stores features, original
row IDs, and 64-bit list offsets without loading all input rows. A
successful build saves a descriptor-and-centers manifest after the data files;
`massive_open_postings()` validates and reuses it in a later R session. A
10-million-row by eight-feature synthetic check retained every ID exactly
once, but this builder performs no search. Opt-in checkpoints save
completed write-pass blocks and permit validated resume; an interrupted
count pass restarts. If the final write checkpoint was saved, resume now
checks feature and ID byte counts, compares saved offset values with
checkpoint counts, and finishes interrupted final renames without
repeating the source scan. It does not hash feature or ID contents.
Its current row-ID limit is 2^31 - 1. Benchmarks and source identity
are recorded in `fastEmbedR-extra`. A subsequent fresh-source CPU-only
build grouped a distinct 10-million-row by 1,024-feature file of
40.96 GB, larger than Chiamaka's 31 GiB host RAM. The two streaming
passes and complete row-ID verification took 5:40.75 wall time and
peaked at 249,416 KiB process RSS. All 10 million IDs occurred exactly
once, six sampled feature rows matched the input, and the finished
manifest reopened. The two deliberately simple centers made very
uneven lists, so this is an over-RAM storage demonstration, not a
validated ANN routing or recall result. Source archive and container
checksums, logs, and the verification script are in the
`postings-overram-10m-20261004` evidence directory of `fastEmbedR-extra`.
`massive_search_postings()` now provides a bounded native CPU query batch
over selected physical lists. It scans a posting once for all routed queries
in the batch and reports per-query candidate counts. A small all-list test
matches streamed exact KNN, but fewer probes remain approximate with no
recall guarantee. On the 40.96 GB input above, eight all-list posting
queries returned exactly the same 30 neighbor IDs and matching distances
as an independent original-file exact scan. Their respective times were
94.674 and 102.942 seconds, with 112,824 KiB peak process RSS for the
combined run. Both scanned the full dataset; this establishes an over-RAM
search boundary, not an ANN speedup. An explicit `coarse_postings` route now
writes a full
file-backed KNN graph through the existing checkpoint and sampled-recall
audit path. Its all-list case matches streamed exact graph output in a
small test; fewer probes are not certified by that test. This is not yet
a measured large-input ANN speedup or a validated default search route.
`massive_posting_recall_pilot()` compares evenly spaced and distant-center
rows with a bounded streamed exact reference before writing a full graph.
It reports both mean
and minimum sampled-row recall and the backend actually used. A weak
single-list search on a boundary fixture is flagged below its 0.99 row
criterion; probing every list matches exact search. Callers must check
the returned flag before starting the full graph. This pilot still needs
one full-reference exact pass and cannot certify unsampled rows. Center
ranking now selects only the requested top lists before sorting them;
posting scans and their Euclidean objective are unchanged. No speedup is
claimed without a same-input performance comparison.
The pilot now also samples rows most distant from their assigned centers,
sharing the full-graph audit's selector. For smaller sources the selector
scores all source rows; when row-column-center work exceeds one billion
comparisons, it scores a bounded, evenly spaced candidate set instead.
The streamed exact reference still reads the full source once. On an
eight-row boundary
fixture, a two-row pilot selects the two central rows and rejects one-list
search. Candidate-based center scoring is only a diagnostic, not a
recall certificate for untested rows.
The full-graph route reduces an oversized query block to fit its declared
buffer budget and reports the actual block size; bounded batch search keeps
its requested row count or fails. A 64-query USPS probe on the available
11,000-row benchmark matrix found mean recall 0.998 with eight of 32 lists,
but minimum per-query recall was 0.90. A full file-backed graph at that
setting had mean 0.9948 over 64 audited rows, also with a 0.90 minimum.
These are single local observations on an input subset, not a general recall
or speed guarantee. A separate whole-process macOS measurement peaked at
178 MB RSS, versus a 72 MB algorithm-buffer estimate, confirming that the
declared budget does not cap total process memory. Reproducible inputs and
results are in `fastEmbedR-extra`.
A separate 256-query MNIST70k diagnostic used each posting list's
maximum center radius and an oracle exact kth-neighbor distance as a
lower bound on the lists needed for exact search. It still admitted
98.3% of rows with 32 lists and 87.7% with 128 lists on average.
All saved exact neighbor IDs were covered in the sample. This simple
bound is too weak to justify a production exact-posting branch; it
does not certify other queries or exclude stronger indexing methods.
Full posting-graph search now defaults to at most 65,536 query rows per
block, subject to the same RAM and native-buffer caps. It reports a
conservative upper bound on posting bytes reread across query blocks.
Bounded queries may also read the physical posting feature file in posting
order; native self-exclusion uses its saved original row IDs. On a
10,000-row, 16-list synthetic fixture with 256-row query batches, this
reduced logical posting reads from 14.4 MB to 5.15 MB while producing
identical original-row-order neighbors and distances. The full-graph
route now
accepts `query_order = "posting"` and performs that bounded native
reordering. On the same 10,000-row fixture with 256-row query batches,
the final graph files were byte-identical to original-order output.
Across two one-shot runs, the grouped route took 0.816-0.841 seconds
versus 0.852-1.374 seconds for the 256-row original-order route;
reordering took 0.056 seconds in each run. The default original-order
route chose a single 10,000-row batch and finished in 0.661-0.776
seconds, with lower estimated algorithm-buffer RAM. These are
warm-cache diagnostics, not evidence of a general
runtime or peak-memory improvement. The grouped route trades additional
temporary disk space and reordering work for fewer logical posting
reads when query batches must be small. A failed reorder can be retried
from completed grouped graph files with `checkpoint = TRUE` and
`resume = TRUE`. Fault-injection tests resume the grouped search when
execution stops before, between, or after its two file renames. A
committed reorder manifest also lets `resume = TRUE` finish a half-renamed
final graph without repeating search or reorder. Resume refuses files
whose recorded size or modification time has changed. These checks
require `checkpoint = TRUE`; they cannot detect all storage corruption.
After a completed run, `resume = TRUE` reopens the same checked graph
without another search or reorder. If completion was interrupted during
cleanup, it removes only grouped data files matching their saved identities;
a small grouped manifest may remain for manual inspection.
In a separate one-million-row, ten-neighbor synthetic reorder test,
the native stage handled 84 MB of grouped inputs and wrote 80 MB of
ordered outputs in 4.336 seconds. The fresh reorder process peaked at
151,633,920 bytes of RSS on macOS. A bounded verifier checked every
row, and an independent reorder produced the same output hashes.
This isolates graph reordering, not posting search or end-to-end graph
construction; one size does not establish constant RSS at larger scales.
The native search also reports logical bytes actually requested from
selected posting files; this is not physical SSD I/O because the OS may
serve repeated reads from cache. The count is carried across completed
checkpoint blocks. An old checkpoint without the counter resumes with
an unknown (`NA`) total rather than silently underreporting reads.
This reduces avoidable rescans but does not make the route a scalable
billion-row ANN index. On the USPS probe, one of 32 lists gave mean
recall 0.759 at 30 neighbors; eight lists gave 0.998. A one-pass,
single-list shortcut would therefore sacrifice too much recall on this
input. Separate HNSW indices inside physical postings were slower than
native exact search in small local pilots and had imperfect recall.
An additional pilot reused the package's native HNSW index separately
inside each physical posting. At 128 or 256 probed lists, it still missed
one neighbor among 7,680 sampled top-30 IDs, whereas exact search over
128 lists recovered all of them. Per-posting native search was fast, but
index rebuilding and R orchestration dominated the 70k-row pilot.
This is evidence to improve or calibrate in-list search, not evidence
of a production out-of-core ANN pipeline.
In a separate 5,000-row posting-like test with 20,000 queries, the
native HNSW default gave mean recall@30 of 0.960 and 67.1% of queries
below 0.99. Raising search effort to 480 gave mean recall 0.9998,
but median query time was 4.45 seconds versus 1.52 seconds for exact
search, before the 1.54-second index build. The explicit search-effort
control is diagnostic; this result rejects native HNSW as the current
default in-list search. Raw query recalls and timings are retained in
`fastEmbedR-extra`.
A 15,000-row, 20,000-query synthetic follow-up found the same tradeoff:
HNSW effort 120 searched in 2.64 seconds versus 6.46 seconds for exact,
but 57.48% of queries had recall@30 below 0.99. At effort 480, HNSW
searched in 8.34 seconds versus 6.59 seconds for exact, and 4.07% of
queries still fell below 0.99. The 9.8-second index build was excluded
from those search times. These one-shape local timings do not establish
an out-of-core speed crossover; they rule out this simple in-list HNSW
configuration as a high-recall default.
`massive_knn_graph()` now imports an externally produced full, directed
non-self KNN file pair without copying it. Native validation streams every
edge, checks dimensions, vertex IDs, distances, self edges, and repeated
neighbors. `massive_read_graph_edges()` exposes bounded edge blocks with
stream or mmap access and 64-bit file offsets. Full-file validation always
uses bounded sequential reads to avoid resident mmap growth. The descriptor
records file
identity and refuses reads after a path, size, or modification-time change.
`massive_graph_partitions()` adds a constant-size row-partition plan;
`massive_full_knn_graph()` can create such a graph from file-backed
float32 rows by scanning bounded query and reference blocks with native
multicore exact Euclidean search. It writes query blocks to two graph
files and optionally checkpoints committed rows. This is quadratic in
both work and repeated input reads. It provides a correctness baseline,
not a practical full-data ANN builder for the target massive scale. A
CUDA exact route builds a cuVS index on each bounded reference shard,
searches all query blocks, and merges shard candidates into the same
file-backed graph. An explicit CUDA IVF-Flat route uses the same bounded
shards and merge. It requests 0.99 recall per index, but only a sampled
full-graph audit can measure the merged result. A strict-CUDA test on an
RTX 5060 Ti exercised two IVF shards, checked native cuVS metadata, and
audited 12 query rows. Both CUDA routes check available VRAM and avoid
loading the full reference; repeated query scans still prevent a
practical billion-row ANN builder. CPU sharded HNSW and single-device
CUDA shards can checkpoint completed query batches.
The sharded resource report now includes planned query-batch calls and
logical query-source GiB across all reference passes. These are lower
bounds on feature bytes requested from the source, not measured SSD
traffic; page caching can reduce physical reads. They expose the scan
cost before a full-graph job starts.
In a separate eight-run RTX 5060 Ti diagnostic on diffuse float32 data,
the explicit IVF route was slower and had higher process peak RSS than
sharded exact search at 10,000, 70,000, and 200,000 rows. At 70,000 rows with
2,500-row shards, graph construction took 10.480 versus 9.341 seconds;
with 35,000-row shards, it took 1.290 versus 0.813 seconds. Both routes
recovered every exact top-30 neighbor in 100 sampled rows per run. At
200,000 rows with 100,000-row shards, IVF took 4.416 versus 2.933
seconds, with whole-process peak RSS of 863 versus 670 MiB.
These are single runs on small synthetic inputs, not evidence of a
general IVF disadvantage or a recall guarantee. The shard-size contrast
shows the cost of scanning all queries for every shard. IVF remains an
explicit choice; it is not automatically selected for the full graph.
The script, source identity, raw results, sampled row recalls, and
whole-process RSS logs are in the `cuda-sharded-ivf-20261002` evidence
directory of `fastEmbedR-extra`. VRAM was estimated, not measured.
On the same RTX 5060 Ti, a later one-million-row, four-shard clustered
float32 diagnostic compared CUDA IVF query batches of 256 and 8,192 rows.
Graph-call times were 26.393 and 22.707 seconds, respectively, with
byte-identical saved neighbors and distances. Whole-process peak RSS
rose from 1,229,640 to 1,544,576 KiB. The sampled 100-row recall was
1.0 for both. The experimental sharded CUDA default is now 8,192 rows,
subject to the existing RAM and VRAM budget clamp; an explicit
`chunk_rows` still overrides it. A fresh strict-CUDA build selected
8,192 rows and reproduced the saved graph. These single runs do not
establish broad speed or quality gains. With 100 million rows and
250,000-row reference shards, even this default requires about 4.9
million query batches and repeated scans of the 12 GB PCA-score file.
The route still needs a more scalable full-graph search strategy.
The benchmark script, archived source, and measurements are in
`fastEmbedR-extra/benchmarks/linux/massive-data/evidence/`
`cuda-shard-batch-20261004/`.
On a separate one-million-row diffuse input with the same query batch,
four 250,000-row IVF shards took 79.914 seconds and one million-row
IVF shard took 87.017 seconds for the graph call. The four shards
each searched all 500 IVF lists; the one large shard searched 512 of
1,024 lists and differed from a one-shard exact graph on 54,097 of
one million rows, with mean top-30 overlap 0.9981. Explicit exact
cuVS took 40.134 and 26.052 seconds with four and one reference
shards, respectively. All four saved graphs were audited on 100 exact
query rows, but this is one synthetic shape and one run per setting.
It does not justify an automatic algorithm or shard-size change.
Source, image, full-file comparison, recall, timing, and memory evidence
are in `cuda-reference-shard-20261004/` in `fastEmbedR-extra`.
Those earlier peak-RSS measurements included generation of a full R
double matrix in each measured process, so they cannot isolate KNN
memory use. A corrected diagnostic generated one 1,000,000-by-32
Gaussian float32 file in bounded blocks in a separate process. On the
current strict-CUDA source archive, exact cuVS graph calls took 26.000
seconds with one reference shard and 39.968 seconds with four, with
R-process peak RSS of 686.6 and 600.4 MiB. IVF-Flat took 86.753 and
79.655 seconds, with peak RSS of 887.0 and 815.3 MiB. An external
0.2-second sampler observed compute-process GPU allocations of 268
and 248 MiB for exact search, and 592 MiB for each IVF route; these
are sampled values, not guaranteed instantaneous peaks. Whole-file
comparison with the one-shard exact graph found neighbor-set changes
on 7, 53,827, and 55 of one million rows for four-shard exact,
one-shard IVF, and four-shard IVF, respectively. The maximum relative
increase in distance at rank 30 was zero, 1.45%, and 5.82e-7 in that
same order. This is one synthetic input, not a general routing policy.
The corrected driver and evidence are in
`cuda-streamed-input-shard-20261004/` under `fastEmbedR-extra`.
The default commits every 100 blocks and at every shard boundary;
uncommitted blocks are replayed after resume. The interval is part of
the checkpoint identity.
Resume rebuilds the current shard index and compares its first query
batch before touching partial output. A replayed incomplete batch is
merged idempotently; the pilot does not prove full HNSW index identity.
The graph can feed experimental CPU and CUDA full-data UMAP after fuzzy
symmetrization. That fuzzy graph can also feed CPU full-graph Louvain,
Leiden, or approximate partitioned Walktrap; CUDA clustering remains
unavailable.
`massive_read_graph_partition()` loads one bounded edge list at a time.
`massive_umap_memberships()` now scans a fixed-k distance graph twice and
writes its directed UMAP membership weights to a file-backed `.weights.f32`
file, sharing the existing indices. Its scale calculation is shared with
the in-memory CPU UMAP graph builder. Independent rows can use multiple CPU
workers; the bounded input scan and output write remain sequential.
On a synthetic 100-million-row, eight-neighbor graph, two one-worker
calls took 68.0 and 66.5 seconds, while two four-worker calls took 33.0
and 32.7 seconds. All four 3.2 GB weight outputs were byte-identical;
process peak RSS stayed near 83 MB. Cache state differed among calls,
so this is a stage-level diagnostic, not a complete UMAP speed claim.
Checkpointing commits completed blocks of both the global-mean scan and
weight write pass. The partial mean sum is serialized at native precision,
so resume continues the scan without rereading committed blocks.
`massive_umap_fuzzy_graph()` now externally sorts those directed weights,
applies the fuzzy union to reciprocal edges, and writes a symmetric,
variable-degree CSR graph. The sort bounds RAM by `memory_limit`, checks
available temporary disk space, and uses only the CPU. Opt-in checkpoints
retain completed pair-sort runs and resume the input scan at the last
committed block. Resume checks source and native-library identity, controls,
run paths, sizes, and modification times. It also accepts a checkpoint
written before the work directory was created. Uncommitted runs and merge
files are discarded on resume; the final merge and CSR write repeat.
Preserving pair runs increases temporary disk use. If final output files
exist but their manifest was not committed, manual inspection is still
required. File identity does not detect same-size, same-time corruption.
A small
fixture matches the in-memory CPU UMAP graph builder's edges and weights.
`massive_tsne_affinities()` reuses that bounded external sort with the
native float32 t-SNE conditional-probability calculation. It consumes a
full distance-valued KNN file pair and writes symmetric normalized CSR
affinities. The same pair-run checkpoint and resume rules apply. A
regression test compares every row with the resident native
affinity calculation. Independent row probabilities can use `n.cores`
workers, while graph symmetrization and sorting remain serial. On a
synthetic one-million-row, 30-neighbor compact-support graph, one-worker
runs took 9.708 and 9.803 seconds; four-worker runs took 9.883 and
9.904 seconds. The four outputs were byte-identical, but parallel rows
did not accelerate this complete stage. Peak R process RSS was about
348-350 MB with a `64MB` algorithm-buffer budget. The input was synthetic
and the host was not isolated from other workloads. Logs and output
hashes are in `fastEmbedR-extra`. The output can be reopened with
`massive_affinity_graph()`, which checks its content-hashed manifest and
validates every CSR row. It does not optimize a full-data t-SNE layout;
the CPU optimizer described above consumes this graph. CUDA full-graph
t-SNE remains unimplemented.
`massive_umap_optimize()` consumes this graph in an experimental,
single-core CPU optimizer. It streams edges for every epoch. The default
keeps float32 coordinates in RAM; explicit POSIX `layout_storage = "mmap"`
stores them in a writable file. The public `umap()` full-graph route forwards
this explicit choice. Mapped updates have irregular disk access
and are not yet performance validated at scale. Mapped pages can still
enter process RSS through the OS page cache; `memory_limit` bounds
working buffers, not total resident pages in this mode.
It requires a file-backed 2D or 3D initialization. Opt-in epoch snapshots
allow resume with the same graph, initialization, optimizer controls, and
output path. A partial epoch is repeated after interruption. A forced-stop
test produces byte-identical final output to an uninterrupted CPU run.
Mapped storage can also resume from a completed epoch snapshot, rebuilding
its `.part` coordinate file from that snapshot. It needs room for the
mapped file and up to two snapshots while switching checkpoints.
Its performance and quality on large datasets have not been established.
`massive_weighted_graph()`
also imports externally supplied values in [0, 1] and does not verify
reciprocity. A variable-degree weighted graph can instead be imported from
`.offsets.u64`, `.indices.u32`, and `.weights.f32` files with
`massive_csr_graph()`. Its zero-based 64-bit offsets support edge counts
larger than signed 32-bit, while one-based neighbor IDs remain limited to
2^31 - 1 vertices. Import streams all rows and edges in blocks of at most
65,536 edges, validates offset monotonicity, sorted unique neighbors, and
weights, and records the maximum row degree for bounded partition reads.
It requires at least one edge. Materializing a high-degree row still needs
an operation-specific edge budget. The CSR descriptor does not certify
symmetry or a UMAP fuzzy-set construction; that distinction is recorded
only for a graph built by `massive_umap_fuzzy_graph()`. A 65,538-vertex
star from `k = 1` input passes fuzzy graph construction, a two-epoch
CPU UMAP fit, a Louvain local-moving level, and a Leiden refinement
level with a 65,537-edge hub. This checks the row-block boundary,
not large-data runtime or quality. The experimental sharded HNSW builder
can optionally audit evenly spaced query rows against
streamed exact neighbors and report observed mean and minimum row recall.
Sampled queries share bounded reference passes, reported as
`audit_reference_passes`. The audit is not a scale-validated full-data ANN
solution;
partitioned full-graph Walktrap is experimental and approximate.
Louvain and Leiden keep large
contracted levels file-backed and retains intermediate files.
`massive_graph_modularity()` now scans a symmetric fuzzy CSR graph and a
disk-backed `uint32` membership file to measure full-graph modularity
without creating a resident edge list. It agrees with the resident native
modularity calculation on a fixed test graph. This CPU-only POSIX route
memory-maps the labels: scan buffers are bounded, but mapped pages may
contribute to RSS and the declared memory limit is not an RSS guarantee.
It does not compute communities or make landmark memberships equivalent
to full-graph community detection. `massive_louvain_level()` separately
performs a CPU-only first local-moving pass on that fuzzy graph and
writes a mapped `uint32` label file. It uses the resident Louvain move
score and shuffles rows within bounded graph blocks instead of keeping
a full vertex permutation in RAM. Each pass chooses a seeded starting
block, then reads the remaining CSR blocks in cyclic order. This avoids
the scattered block reads of a random-stride traversal; it does not
establish a measured speedup on an out-of-RAM graph.
`massive_louvain()` adds external contraction and keeps higher levels
file-backed while their resident-memory estimate is too large. Coarse
weights use float64 CSR, including self-loops and rows with more than
65,536 edges. When a level passes a conservative memory check, the
existing resident Louvain hierarchy takes over. With explicit
`checkpoint = TRUE`, the route can resume from a verified completed
first-level label file or contracted graph. It cannot resume within a
native scan, contraction, or later hierarchy level; partial work files
are rejected. Checkpoints compare file path, size, and modification time,
not a full content hash, and do not enforce a process-wide RSS ceiling.
CUDA full-graph clustering remains absent.
An opt-in `initial =` argument on `massive_louvain()` and
`massive_leiden()` accepts a saved landmark-clustering result when its
labels match the full graph's row order. The full graph is still required;
this reuses the disk-backed partition as a starting point, not as a
replacement for full-graph local moves or a bounded-neighborhood
refinement algorithm. Both routes reject label files whose recorded path,
size, or modification time changes; they do not hash the contents.
The same route is available through `massive_cluster(fuzzy,
method = "louvain", massive = "out_of_core_graph", output = path)` so the
file-backed fuzzy graph can be reused for embedding and clustering.
The experimental Leiden route reuses these disk-backed local moves,
refines each parent partition with Leiden's constrained singleton-merge
rule, and contracts by the refined labels. It carries the parent
partition into the next local-moving level. Original-row labels and
all contracted levels remain on disk. A 512-vertex ring test reaches
five levels and checks connectivity, adjusted Rand agreement against
resident Leiden, and original-graph modularity; no out-of-RAM Leiden
benchmark has been completed yet. It is CPU-only and single-worker.
On a saved real MNIST70k fuzzy graph, three seeds gave file-backed versus
resident adjusted Rand indices of 0.867-0.931. Independent rescoring
confirmed both modularities on every seed. This is one-graph quality
evidence, not a matched performance comparison or out-of-RAM test.
Separate seed-4 processes measured 109 MB and 402 MB peak RSS for
file-backed and resident Leiden on that graph, respectively. The
same-seed label files were byte-identical to the paired-run labels;
one isolated measurement per route does not establish speedup.
A separate one-seed MNIST70k diagnostic initialized full-graph fits from
2,000 saved landmark labels. The seeded local-plus-hierarchy stage took
7.090 versus 8.966 seconds for Louvain and 11.996 versus 14.716 seconds
for Leiden. Including landmark selection, KNN, and label assignment
removed the apparent end-to-end speed gain. Louvain modularity fell from
0.862983 to 0.852007, whereas Leiden rose from 0.864403 to 0.866312.
Seeding remains opt-in; these results do not establish general quality
or memory advantages. The complete run and input identities are in
`fastEmbedR-extra/benchmarks/macos/massive-data/evidence/`
`seeded-mnist70k-20261004-r2/`.
Optional checkpoints save completed graph levels and the final mapping;
resume verifies graph, mapping, controls, native library, and mapped-label
identities. An interrupted local move or contraction retains partial files
and still requires separate recovery. Full-graph Walktrap remains absent.
Louvain's mapped label, count, and volume files require 16 bytes per vertex;
mapped pages may contribute to RSS beyond the scan-buffer budget.
The experimental full-graph UMAP optimizer currently works only with
the symmetric fuzzy graph produced by `massive_umap_fuzzy_graph()`;
it does not accept a generic imported CSR graph. Import validation does
not establish KNN recall.
The current landmark KNN format stores one-based `uint32` IDs and float32
distances in parallel row-major files. Indexed methods require the
landmark reference to fit in RAM or VRAM; landmark IDs are limited to the
signed 32-bit R range. Explicit CPU `stream_exact` scans bounded query
and reference blocks, so its reference need not fit in RAM, at the cost
of quadratic work and repeated sequential reference reads.
For file-backed CPU exact, HNSW, and CUDA indices, the reference now
enters native float32 storage directly, without an intermediate full R
double matrix. CPU exact search retains that reference across query
batches instead of converting it again for every batch.
On a small Apple M3 synthetic check, five paired runs gave median times
of 0.051 seconds for per-batch conversion and 0.046 seconds for reuse;
all neighbor outputs matched. This is noisy stage-level evidence, not
an end-to-end speed or peak-RSS claim. The script and raw CSV are in
`fastEmbedR-extra/benchmarks/macos/massive-data/evidence/`.
The result records `reference_storage` so this boundary is inspectable.
File-backed query batches now also enter native search as float32 buffers;
resident R-matrix queries retain their existing double-matrix route.
The result records `query_storage`. On Chiamaka, a 5,000-query by
1,024-reference MNIST CUDA exact-KNN check produced byte-identical old
and new output files. Four alternating runs had median call times of
0.0765 and 0.0200 seconds, and median process peak RAM of 588.4 and
554.8 MB, respectively. This is a small stage-level diagnostic, not an
end-to-end embedding benchmark. Source identities and raw results are in
the `fastEmbedR-extra` native-query-batches evidence directory.
The persistent CUDA cuVS index now retains query, raw-neighbor, and
final-neighbor device buffers across bounded batches. They grow only when
a larger batch arrives and are released with the index. A strict CUDA
test on Chiamaka verified reuse across smaller and larger batches and
matched every neighbor against independent CPU exact search; IVF and
landmark exact-search regression tests passed. This removes repeated
per-batch CUDA allocations, but a matched end-to-end speed comparison
has not yet been measured. Evidence is in the `fastEmbedR-extra`
cuda-buffer-reuse directory.
KNN can checkpoint both output files and resume after rebuilding the
reference index. An original first-batch result fingerprint must match
before resume, including IVF-Flat calibration metadata. This is a
bounded pilot check, not a full proof of approximate-index identity.
`massive_audit_landmark_knn()` can compare evenly spaced saved query
rows against an independent streamed exact CPU search. It reports mean
and minimum sampled recall, sampled row IDs, and bounded reference passes,
without loading the whole reference or query matrix. The result does not
certify unobserved rows.
For explicit multiple CUDA devices, disjoint file-backed query row views
run in separate R workers. Each worker builds a resident landmark index
and writes a shard; a bounded four-byte merger preserves input row order
for both IDs and distances. The reference must fit on every selected GPU,
and the route needs disk space for shards plus merged files. Multi-GPU
KNN now checkpoints query shards and both merged outputs. Resume checks
the package version, input identities, controls, device assignment,
shard metadata, and file sizes before reusing a completed shard. Local
tests cover interrupted merges and mismatched identities. A real CUDA
exact-neighbor check and completed-shard resume passed on one RTX 5060 Ti.
A separate two-process test on that same GPU searched disjoint query
ranges and merged both outputs. All saved IDs and distances matched a
single-worker exact cuVS run byte-for-byte for 4,096 queries against
2,048 references. This checks worker isolation and row ordering, not
cross-device execution or acceleration; the script and checksums are
in `fastEmbedR-extra` massive-data evidence.
A two-device run and its resource accounting remain unverified because
that test host exposes only one GPU.
Landmark Louvain and Leiden now expose the same `devices` query-sharding
route. The first selected GPU builds and clusters the resident reference
graph; the CPU assigns query communities from the merged file-backed
neighbor rows. Single-device clustering is tested on CUDA; a two-device
clustering run remains unverified on the available hardware.
An internal deterministic synthetic source tests reads beyond signed 32-bit
row offsets without allocating those rows or a large sparse file.

The experimental landmark workflow still fits its reference embedding in
RAM. The query data and KNN results remain file-backed. UMAP's zero-epoch
projection is weighted KNN interpolation, not full UMAP optimization;
positive epochs use the existing fixed-reference CPU optimizer. t-SNE uses
the existing fixed-reference CPU transform. Query batches are independent,
and changing the batch size may change stochastic optimizer results.
A physical 120-million-by-80 float32 source (38.4 GB, larger than the
test host's 33.3 GB RAM and 16.3 GB GPU memory) completed CUDA landmark
UMAP with 32 landmarks and five neighbors. The complete run took 309.5
seconds, wrote a 4.8 GB query-to-landmark KNN and 0.96 GB layout, and
peaked at 651 MB R-process RSS. A bounded sequential check passed all
120 million output rows. This demonstrates the streaming memory boundary,
not useful quality at that landmark count or full-data UMAP optimization.
The same source also completed CUDA landmark t-SNE in 318.9 seconds
with 844 MB peak R-process RSS. Its five-neighbor query KNN was
byte-identical to UMAP's but rebuilt by that earlier high-level call.
On a separate real MNIST70k diagnostic with a common 2,000-row quality
sample, a full CUDA t-SNE fit had trustworthiness 0.93857, Preserve@30
0.43502, and embedding-space label KNN accuracy 0.9590. File-backed
landmark fits with 5,000, 10,000, and 20,000 reference rows had
trustworthiness 0.93855, 0.93720, and 0.93929, respectively, but label
KNN accuracy was 0.8460, 0.8970, and 0.9195. Their peak R-process RSS
was 904-945 MB versus 1,028 MB for the full fit. Every fit used seed 4,
perplexity 30, and 1,000 reference optimization iterations. The full fit
started from a resident float32 object whereas landmark fits read a file,
so their single-run timings are not matched end-to-end comparisons.
The input fits RAM; this is quality-versus-memory evidence, not another
beyond-RAM test. Logs and fixed row IDs are in `fastEmbedR-extra`.
The high-level landmark API now accepts the prior result as `nn` and
checks source, landmark, neighbor, and file identities before reusing
the graph; a strict-CUDA 3,000-row smoke test confirms no second KNN
file is written. Large-scale reuse timing remains unmeasured. A separate
one-million-row synthetic diagnostic measured median t-SNE call times
of 1.436 seconds with a fresh KNN and 0.236 seconds with the checked
saved KNN over three same-seed pairs. This is not a general speedup
estimate or a real-data quality result. A separate
driver reused UMAP's saved KNN to run landmark Leiden, Louvain, and
Walktrap with 120-million-row file-backed label and confidence outputs.
The incremental post-KNN stages took 18.1, 7.7, and 15.1 seconds;
Walktrap ran on CPU and all three assignments ran on CPU. Bounded scans
verified every saved t-SNE coordinate, neighbor, label, and confidence.
These timings exclude a repeated graph search for clustering and do not
validate scientific quality with only 32 landmarks.
Source identity, measurements, and the verifier are in `fastEmbedR-extra`.
The opt-in `local_refine = TRUE` CPU UMAP route also constructs a bounded
query-to-query KNN within each chunk and its overlapping rows. The
existing masked optimizer updates core query rows while landmark and
overlap-only coordinates stay fixed. The query source must match the
saved landmark KNN source identity. This is not a full-data UMAP graph;
window-boundary effects and dependence on input row order remain
possible, and this route can be slower.
The projection file contains query rows only. When the queries include the
source rows, passing the original selection map substitutes exact
reference coordinates at landmark positions. Package-built fuzzy UMAP
graphs and t-SNE affinities have content-hashed manifests; projection
layouts have a local completion manifest. `massive_open_projection()`
reopens the descriptor only when the coordinate and KNN files retain their
paths, sizes, and modification times. This is not a portable or
content-hashed provenance record. Projection can checkpoint completed
batches and resume from its partial output. Resume checks the layout and
controls plus KNN file paths, sizes, and modification times; it does not
hash every byte of the potentially enormous KNN files.

Landmark clustering uses the package's native `graph_cluster()` on the
resident reference graph. Louvain and Leiden can run on CUDA; Walktrap
remains CPU-only. Non-landmark rows receive inverse-distance
weighted community votes from their saved query-to-reference KNN rows.
The saved KNN graph may be built on CUDA and reused without repeating
the neighbor search. Query community voting remains on CPU and is
reported separately from graph construction and landmark clustering.
The memory preflight includes a conservative allowance for resident
graph edges and native clustering buffers, plus VRAM for CUDA. Exact
Walktrap has a native ceiling of 4,000 reference vertices and allocates
two dense double-precision transition matrices, requiring at least
`16 * landmarks^2` bytes. Automatic selection and explicit requests
are checked against both limits before landmark extraction begins.
The file-backed result contains one-based `uint32` membership and float32
vote confidence. It does not update non-landmark edges. A distinct
file-backed full-graph CPU Louvain or Leiden route is available.
Partitioned Walktrap runs exact native walks within bounded induced
blocks, then contracts communities. Cross-block walks are absent at
the first level, so this is an explicit approximation. A connected
360-point synthetic comparison gave ARI 0.887-0.943 versus whole-graph
Walktrap for block widths 60-180; this does not establish large-graph
quality or runtime. A second 600-point diagnostic reordered the same
three-group data. At block width 100, agreement with planted groups
dropped from ARI 0.990 in grouped row order to 0.833 after shuffling;
partitioned modularity fell from 0.623 to 0.581, while whole-graph
Walktrap modularity remained 0.624. The first level retained 47.5% of
edge weight inside grouped-row blocks but only 17.0% after shuffling.
At block width 200 the fractions were 95.1% and 33.7%. This diagnostic
makes the vertex-order sensitivity visible; it does not provide a
general quality threshold. Graph-aware partitioning and larger
real-data checks are still needed. The assignment stage can
checkpoint the
resident landmark communities and completed output rows. Checkpoints
verify graph values and KNN file identity, but do not hash every byte of
the potentially large KNN files. The initial landmark graph must still
fit in RAM and is not yet checkpointed during its construction. Since
exact Walktrap also stores two dense transition matrices, file-backing
only its input CSR graph would not remove the quadratic memory cost.

The CUDA UMAP projection and 2D or 3D fixed-reference t-SNE
transformation
routes write one bounded query batch at a time. CUDA zero-epoch UMAP
retains its reference layout and query buffers when reference-transfer
bytes exceed planned batch-transfer bytes; otherwise it uses the existing
per-batch route. The selected engine is reported in result metadata.
t-SNE transformation still allocates and transfers per batch. The routes
cap batch size using free VRAM and report estimated peak VRAM. CUDA UMAP
refinement remains unsupported. Observed
recall audits across the complete query stream are still required.
An RTX 5060 Ti strict-CUDA build passed the public file-backed 3D
landmark t-SNE route and the complete installed-package test suite.
With fixed exact query neighbors and initial coordinates, CPU/CUDA
transform outputs differed by at most 3.95e-6 after ten iterations
on a small 3D fixture. This validates the transform path, not full-data
3D t-SNE, large-data quality, or throughput. Archive and test identities
are in the `cuda-3d-transform-20261004/attempt3` evidence directory of
`fastEmbedR-extra`.
Local query-graph refinement currently requires an explicit CPU backend;
a CUDA request fails rather than falling back.

## Full-data ANN direction and evidence

The current contiguous-row HNSW and CUDA shards query every shard, so their
work and repeated reads remain unsuitable as a billion-row default. A
feature-routed, disk-resident index is the next requirement, not merely a
smaller contiguous shard size. Search continues to use the package's CPU
exact/HNSW and CUDA cuVS exact/IVF routes. Physical postings are an explicit
experimental storage route, not an automatic replacement; NN-descent is not
part of this design. [SPANN](https://www.microsoft.com/en-us/research/wp-content/uploads/2021/11/SPANN_finalversion1.pdf)
demonstrates a RAM-resident routing layer over disk posting lists, including
list balancing and boundary replication. [DiskANN](https://www.microsoft.com/en-us/research/publication/diskann-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node/)
demonstrates the alternative SSD graph architecture. The
[FAISS on-disk inverted-list design](https://github.com/facebookresearch/faiss/wiki/Inverted-list-objects-and-scanners)
is a storage reference, not a package dependency or copied implementation.

The executable routing diagnostic in `fastEmbedR-extra` tests a deliberately
simple starting point: random reference anchors or sampled k-means
centroids, one to four assigned lists per row, and exact reranking of lists
selected by `nprobe`. It streams the source once for evaluation, so its time
is **not** the runtime of a completed disk ANN index. On deterministic
20,000-by-96 diffuse Gaussian input with 64 anchors and 32 audited queries,
`nprobe = 32` scanned 79.4% of candidates for only 94.4% mean recall at
15. Training 64 centroids improved recall to 99.6%, but scanned 94.7% of
candidates; 128 trained centroids scanned 91.0% for 99.2% mean recall.
The 64-anchor random assignment's largest list held 14.6% of rows, versus
1.6% for an even split. By contrast, 8 of 64 lists on a clustered
20,000-by-16 pilot scanned 12.8% with complete sampled recall. These
small synthetic pilots reject single-assignment centroid routing as a
universal full-data default. Replication was not a free repair: on that
diffuse pilot, two copies per row and 16 probes recovered 97.5% mean
recall while projecting 129.6% of a full posting scan; four copies and
16 probes recovered all sampled neighbors but projected 258.5% of a
full posting scan. These projected reads count duplicated rows and are
not measured SSD I/O. The pilots do not rank balanced posting lists,
SPANN, or DiskANN, and do not establish behavior on real datasets.

On real MNIST70k float32 input, a separate 256-query exact-search audit
showed that learned coarse routing can reduce candidate reads, but its
mean recall concealed weak individual queries. With 64 lists and no
replication, 8 probes read an estimated 14.0% of postings for mean
recall 0.987 at 30 neighbors; the minimum row recall was 0.633.
On file-backed 30-component PCA scores, 256 lists and 16 probes read
6.6% for mean recall 0.990 but minimum recall 0.700. At 128 probes,
the same sampled PCA-space queries all reached exact recall, while
reading 50.8% of postings. These are projected posting reads from a
streamed routing diagnostic, not measured SSD-index timings. They
support an optional feature-routed index with an observed-recall gate,
not an unqualified IVF default. The locked input, scripts, and CSVs
are documented in `fastEmbedR-extra`.

A fresh-source full-graph CPU diagnostic on all 70,000 MNIST rows after
file-backed 30-component PCA confirms the sampling risk. With 32
physical postings and `k = 30`, eight probes built the graph in 7.408
seconds versus 19.109 seconds for streamed exact search. Complete
row-by-row comparison found mean recall 0.99659 but a minimum of
0.43333, with 457 rows below 0.90. Sixteen probes took 12.417
seconds and raised mean recall to 0.99983, yet eight rows remained
below 0.90. The 64-row audit missed these worst cases at both probe
settings. These are single-run graph-plus-audit times on a matrix that
fits RAM; they are not a massive-scale speed claim. The complete
per-row evidence and the dirty-source archive are in
`fastEmbedR-extra/benchmarks/linux/massive-data/evidence/`
`mnist70k-pca30-postings-20261002/`. A full-data posting method cannot
be selected automatically from mean sampled recall alone. The audit
pass flag now requires every sampled row to reach 0.99 and reports the
count below that target; unsampled rows remain uncertified. A tested
center-radius distance bound certified no rows at eight or 16 probes,
so it is not a useful exactness shortcut on this input.

A production-ready route still needs shape-dependent routing and observed
recall gates before it can be selected automatically. The explicit posting
route physically groups feature vectors and one-based row IDs, routes
queries by feature similarity, and writes results in original row order.
An opt-in posting recall audit combines evenly spaced rows, rows farthest
from their nearest routing center, and representatives from the smallest
nonempty posting lists. This last component is necessary because the
previous selector sampled no rows from a 5,904-row list in a
10-million-row, 1,024-feature input. The revised eight-row selector
sampled two such rows. At `nprobe = 1`, all eight revised rows had exact
Recall@30 of 1.0, but the pilot read approximately 41 GB of postings
because the other list held 9,994,096 rows. Thus this is an audit
coverage improvement, not an ANN speed or general recall result. Source,
row-level outputs, and resource measurements are in
`posting-minority-audit-10m-20261005` under `fastEmbedR-extra`.
On a deterministic 50,000-row sample from that same 1,024-feature
source, replacing two arbitrary routing centers with 32 trained centers
raised the minimum eight-row Recall@30 from 0.467 to 1.0, but the
largest list still held 80.7% of rows. A streamed 30-component PCA
reduced the largest list to 49.4% and the sampled posting read volume
from 177 MB to 6.0 MB. However, independent exact searches on 15
identical rows showed only 0.273 mean and 0.033 minimum overlap between
PCA-space and original-feature top-30 neighbors. PCA routing therefore
cannot silently replace original-feature KNN, and center training alone
does not solve the imbalance. These are single-run, sampled diagnostics,
not full-data speed or quality claims. Scripts, source identity, and
row-level results are in `posting-routing-pca-20261005` under
`fastEmbedR-extra`.
The persistent CPU HNSW index had a separate reachability problem on
the same 50,000-row original-feature sample. Its default mean
Recall@30 was 0.413, and searching with effort equal to the reference
size still reached only 0.693. Preserving bidirectional insertion
links in this experimental index raised recall to 1.0 for all 256
independent queries at default effort, with query time increasing from
0.027 to 0.057 seconds and build time from 51.9 to 61.7 seconds.
One- and four-thread builds also reached exact recall on a 10,000-row
subset at full effort. This is a single synthetic-data diagnostic,
not a general 0.99-recall guarantee; exact search remains cheaper for
one small query batch when index construction is included. Raw results
and the corrected source-archive identity are in the
`hnsw-connectivity-20261005` evidence directory of `fastEmbedR-extra`.
On original MNIST70k features, a separate 20,000-reference,
256-held-out-query diagnostic reached mean Recall@30 of 0.9958 at
default effort, but 19 queries fell below 0.99 and the minimum was
0.633. Raising effort from 60 to 480 recovered all 30 exact neighbors
on every query; query time rose from 0.024 to 0.100 seconds after a
6.381-second index build. The public one-shard HNSW route also built a
30-non-self-neighbor graph for all 70,000 MNIST rows in 101.0 seconds,
including a 256-row exact audit. Sampled mean and minimum recall were
1.0; peak observed process RSS was 796,796 KiB. This input fits RAM,
and the sampled audit cannot certify every graph row. These results
do not establish reliable default recall on held-out queries or the
performance of a truly larger-than-RAM HNSW graph. Raw evidence is in
`mnist-hnsw-connected-20261005` under `fastEmbedR-extra`.
On one earlier MNIST70k PCA
input using 256 centers and 16 probes, equal-size 256-row audits found
35 rows below 0.99 with evenly spaced sampling, 105 with the mixed
selection, and 175 with distant rows only. Those counts predate the
small-posting addition and are not a result for its revised sample.
This is a biased diagnostic that catches more weak rows; it does not
estimate population recall or certify unobserved queries. The extra
distance scan runs only when
`audit_rows > 0`. Script, row IDs, and measurements are in the
`mnist-posting-audit-20261002` evidence directory of `fastEmbedR-extra`.
Its preflight must budget input, list files, graph output, merge runs, and
checkpoints. For 1B-by-96 float32 data, one uncompressed feature-and-ID
partition copy alone is about 388 GB, before the roughly 240 GB fixed-30
neighbor graph. The acceptance gates are measured recall at several list
probe budgets and input shapes, end-to-end wall time and peak RAM/VRAM versus
the existing exact and sharded baselines, source identity, and interruption
recovery. A CUDA request must not silently use CPU search. None of these
gates is yet met by a full-data ANN builder.

Overlapping local partitions are a useful literature direction, but simple
replication did not pass an initial real-data gate. On one MNIST70k PCA30
input with 256 centers, a fixed 256-row audit compared exact-neighbor
candidate coverage at similar logical posting-read counts. One copy with
16 query probes covered 0.9473 of exact top-30 IDs on average; four copies
with four local partitions covered 0.9302. One copy with 32 probes covered
0.9819; four copies with eight query probes covered 0.9763. Eight-copy
local partitions read roughly twice as many posting entries as the latter
pair and still missed at least one exact neighbor for 25 audited rows.
This is a candidate-set upper bound, not measured local-graph recall or a
general rejection of overlap. The input and center set fit RAM, and the
audit intentionally includes difficult rows. The script, row-level results,
and literature distinctions are in the `partition-overlap-mnist70k-20261002`
evidence directory of `fastEmbedR-extra`. In particular, DiskANN and
Tagore primarily build navigable ANN search indexes, whereas the MLSys
2026 out-of-core UMAP work targets an all-neighbors graph directly. We
should not add fixed-factor posting duplication without a better
quality-versus-I/O result and bounded per-partition resource estimates.
Progressive posting probes are not a sufficient shortcut either. On the
same 256-row MNIST audit, stopping when the top-30 ID set first remained
unchanged reduced mean logical posting reads to 13,160, but left 30 rows
below 0.99 recall and a worst recall of 0.867. Fixed 64-list search read
16,842 entries on average, with 14 rows below 0.99. A full 256-list
scan matched the exact reference on every sampled row. This is
sample-specific diagnostic evidence, not a timing comparison of native
search. The script and row-level outcomes are in the
`adaptive-postings-mnist70k-20261002` evidence directory of
`fastEmbedR-extra`. Any adaptive production policy still needs an
independent recall gate and a validated stopping criterion.
