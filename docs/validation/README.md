# Real-Hardware Evidence

Strict hardware smoke/correctness evidence is produced by
`.github/workflows/hardware-validation.yaml`.
CPU, Metal, and CUDA each produce a separate artifact bound to the exact Git
commit in the artifact name and in `identity.csv`.

These artifacts demonstrate execution and numerical/backend identity on the
named device. They are not full scientific performance benchmarks. A full
benchmark claim additionally requires the release-locked dataset suite,
repetitions, memory and quality measurements, and manifests from the separate
benchmark repository. A compiled architecture that has not executed the
workflow is only a build-level compatibility target.

Every accepted artifact contains:

- `git-commit.txt` and an empty `git-status.txt`;
- `identity.csv` with SHA-256 hashes of the Git archive, source package, and
  installed `fastEmbedR` shared library;
- `SHA256SUMS` covering every retained evidence file;
- `hardware.txt`, `sessionInfo.txt`, and `capabilities.csv`;
- `evidence-scope.csv`, which labels the artifact as strict hardware
  smoke/correctness evidence rather than a performance benchmark;
- build and installation logs;
- the complete installed-package test log;
- `hardware-benchmark.csv` for PCA, KNN, UMAP, t-SNE, and Leiden; and
- `validation-results.rds` containing the layouts and clustering membership.

An artifact is accepted only when all public operations record the requested
backend and the complete test suite succeeds. An unavailable accelerator,
silent CPU fallback, dirty checkout, test failure, or backend identity mismatch
fails the job.

GitHub artifacts are not permanent. For each release, copy the three artifacts
unchanged into the `fastEmbedR-extra` repository under:

```text
hardware-validation/<git-commit>/{cpu,metal,cuda}/
```

Verify each copy with `sha256sum -c SHA256SUMS` on Linux or
`shasum -a 256 -c SHA256SUMS` on macOS. The scientific benchmark archive and
these release correctness artifacts serve different purposes and must retain
separate labels.
