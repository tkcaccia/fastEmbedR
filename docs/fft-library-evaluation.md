# FFT Library Evaluation For Metal openTSNE

This note records the external FFT options considered for speeding up the
native Metal openTSNE FFT-grid path.

## Summary

The most useful direction is a package-native Metal FFT plan with standalone
correctness checks before each kernel family is used by openTSNE. The first
validated improvement is a Stockham-512 row/column FFT path adapted from
AppleSiliconFFT. It is enabled only for 512x512 openTSNE/FIt-SNE grids.

## Libraries

### AppleSiliconFFT

Repository: <https://github.com/aminems/AppleSiliconFFT>

License: MIT.

Relevance: high. The papers and code show the right Apple GPU strategy:

- Stockham autosort
- radix-4/radix-8 butterflies
- threadgroup-memory staging
- cheap barriers but expensive scattered threadgroup access
- 512/4096-friendly kernels

Local test:

- With full Xcode 16.2 selected, `xcrun metal` is available at
  `/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/metal`.
- The AppleSiliconFFT Swift host initially picked up stale SDK headers from
  `/usr/local/include`. It built cleanly when Swift was invoked with an
  isolated SDK include path and `-parse-as-library`.
- The upstream multi-size benchmark passed for N = 256 through 16384 on the
  Apple M3.
- For N = 512, upstream validation reported max absolute error
  `1.239777e-05`, max relative error `1.126102e-05`, and L2 relative error
  `2.077061e-07`.
- For N = 512 with batch = 64, upstream reported about `8.3 us` total,
  `0.13 us/FFT`, and `178.73 GFLOPS`.

Port result:

- Added an internal Stockham-512 row/column fast path for `fft_n == 512`.
- The first version had an incorrect radix-4 forward sign; MNIST70k quality
  collapsed (`trust` around `0.03`), so it was not kept.
- After fixing the forward sign, the standalone diagnostic matched the generic
  Metal Cooley-Tukey reference with relative RMS error around `3.16e-7`.
- On the 10k MNIST profiler, FFT GPU time fell from the previous
  `fft_forward + fft_convolution` baseline of about `0.191s` to about
  `0.109s`.
- On MNIST70k PCA50, the focused Metal openTSNE run produced `NN sec = 7.12`,
  `embed sec = 6.116`, and `trust = 0.332`.

Decision:

- Keep the validated Stockham-512 path as the default for 512x512 Metal
  openTSNE FFT grids.
- Continue using the generic Metal Cooley-Tukey path for other grid sizes until
  they have their own correctness and embedding-quality checks.

### muFFT

Repository: <https://github.com/Themaister/muFFT>

License: MIT for the library core. Its optional test/benchmark binaries use
FFTW and are GPL-related; they are not needed for fastEmbedR.

Relevance: medium. It is CPU/SIMD-oriented, not Metal, but it is useful for
Stockham/radix planning and twiddle scheduling ideas.

Decision:

- Good reference for planner design.
- Not useful as a direct Mac GPU backend.

### clFFT

Repository: <https://github.com/clMathLibraries/clFFT>

License: Apache-2.0.

Relevance: low for this package. It is OpenCL-based. Apple deprecated OpenCL,
and fastEmbedR's Mac GPU backend is native Metal.

Decision:

- Do not add as a dependency or backend.
- Some planner concepts may be useful, but direct integration would move the
  package away from native Metal.

### Apple Accelerate/vDSP FFT

Documentation: <https://developer.apple.com/documentation/accelerate/fast-fourier-transforms>

License/API: system framework.

Relevance: medium as a CPU baseline. vDSP is highly optimized on Apple Silicon,
but it is a CPU framework, not a native Metal GPU FFT. Calling it inside the
Metal openTSNE loop would move FFT work back to CPU and require repeated
GPU/CPU synchronization.

Decision:

- Useful as a correctness/performance reference.
- Not suitable as the default Metal openTSNE FFT engine.

### FINUFFT

Documentation: <https://finufft.readthedocs.io/en/latest/>

Relevance: low for current openTSNE. FINUFFT solves nonuniform FFT problems,
while the FIt-SNE/openTSNE negative-gradient path here already splats points to
a uniform grid and then needs uniform FFT convolution.

Decision:

- Not a replacement for the current uniform grid FFT.
- Could become interesting only if the whole repulsive-force approximation is
  redesigned around nonuniform points, which would change more of the math and
  needs separate validation.

## Next Safe Step

Before another openTSNE integration attempt:

1. Add an internal `metal_fft_test_*` function for 1D/2D complex FFT.
2. Compare Metal output against R/CPU FFT on deterministic small inputs.
3. Benchmark generic Cooley-Tukey vs Stockham variants outside openTSNE.
4. Promote a variant into openTSNE only if it passes correctness and improves
   the MNIST70k plot/trust.
