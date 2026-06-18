# Metal FFT Roadmap

fastEmbedR's CUDA openTSNE path is fast because it delegates the Fourier
convolutions to cuFFT. The Metal path uses package-native Metal kernels. The
first validated cuFFT-style improvement is a Stockham-512 path for the 512x512
openTSNE/FIt-SNE grids used by the MNIST 70k benchmark.

This document describes the plan for a small internal Metal FFT engine that can
eventually replace the current openTSNE-only FFT kernels.

## Goal

Build a reusable, native C++/Metal FFT layer for the Mac GPU:

- complex-to-complex 1D FFT
- complex-to-complex 2D FFT
- batched 2D FFT for the openTSNE/FIt-SNE convolution grids
- forward and inverse transforms
- float32 internal storage
- reusable plans and scratch buffers
- correctness tests against R/CPU reference FFTs
- microbenchmarks before integration into embedding code

The first target is openTSNE's FFT-grid negative-gradient step. A kernel
variant is only promoted into the default path if it is faster and keeps the
embedding quality unchanged.

## Current Validated Path

The 512x512 Metal openTSNE grid now uses a radix-4 Stockham row/column FFT
kernel implemented using the MIT-licensed AppleSiliconFFT design as a
reference. The standalone
diagnostic compares this fast path against the previous generic Metal
Cooley-Tukey implementation:

```r
fastEmbedR:::metal_fft512_stockham_diagnostic_cpp(seed = 3, inverse = FALSE)
fastEmbedR:::metal_fft512_stockham_diagnostic_cpp(seed = 3, inverse = TRUE)
```

On this Mac, the corrected kernel had forward relative RMS error
`3.16e-7` and inverse relative RMS error `3.17e-7` versus the generic Metal
reference. The earlier failed version used the wrong radix-4 sign for the
forward transform and produced collapsed MNIST embeddings; that version was
not kept.

## Current Bottleneck

Stage timing on a 10k MNIST raw-pixel subset shows that the Metal openTSNE
optimizer spends most GPU time in the custom FFT stages:

| stage | interpretation |
|---|---|
| `fft_forward` | five forward 2D FFTs: mass, mass_x, mass_y, kernel_q, kernel_q2 |
| `fft_convolution` | four frequency-domain products plus four inverse 2D FFTs |

The non-FFT stages are already small: layout statistics, grid scatter, q
normalization, sparse attractive update, and centering.

## Design

### Plan Object

A future `MetalFFTPlan` should own:

- `fft_size`
- `rank` (`1D` or `2D`)
- `batch`
- `inverse`
- twiddle tables
- scratch buffers
- tuned threadgroup geometry
- selected kernel family

The plan must be reusable across repeated openTSNE calls with the same grid
size.

### Kernel Families To Test

1. Existing Cooley-Tukey kernels
   - bit-reversal rows
   - row butterflies
   - bit-reversal columns
   - column butterflies

2. Stockham autosort kernels
   - avoids explicit bit reversal
   - ping-pongs between input and scratch
   - easier to batch safely

3. Radix-4 / radix-8 stages where possible
   - fewer stages for power-of-two FFTs
   - more arithmetic per dispatch
   - must handle fallback radix-2 for remaining stages

4. Tile/shared-memory kernels
   - load row/column tiles into threadgroup memory
   - reduce global memory traffic
   - likely most important for 512 and 1024 grids

5. Fused FIt-SNE convolution kernels
   - combine frequency-domain multiply and inverse preparation
   - avoid separate dispatches where possible

## Rejected Experiments

These were tested and should not be reintroduced as default code:

- private Metal buffers for openTSNE optimizer state: slower on Apple Silicon
- command-buffer batching alone: neutral
- naive multi-buffer FFT batching: slower than separate transforms
- the first direct Stockham-512 row/column port inspired by AppleSiliconFFT:
  faster FFT stages, but incorrect forward-transform sign and collapsed
  openTSNE quality; replaced by the validated signed radix-4 path
- CPU-only grid negative-gradient method: removed

## Acceptance Criteria

For a candidate Metal FFT kernel to replace the current path:

- relative FFT error below `1e-4` for float32-sized test grids
- openTSNE trustworthiness not worse than the current Metal baseline by more
  than benchmark noise
- MNIST70k raw-pixel openTSNE embedding time faster than the current Metal
  FFT-grid path
- no new user-facing parameter
- no CPU fallback reported as Metal

## Development Loop

1. Run `tools/profile_metal_opentsne_fft.R` to record the baseline stage split.
2. Add one Metal FFT kernel family.
3. Validate numerical error against CPU FFT on small grids.
4. Benchmark only the FFT stages.
5. Run openTSNE on MNIST70k raw pixels.
6. Inspect the plot before keeping the change.
