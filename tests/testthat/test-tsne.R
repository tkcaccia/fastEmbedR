test_that("embed_knn runs native openTSNE from supplied neighbours", {
  set.seed(321)
  x <- matrix(rnorm(50L * 5L), 50L, 5L)
  knn <- faissR::nn(x, k = 16L, backend = "cpu")

  layout <- embed_knn(
    knn,
    method = "opentsne",
    perplexity = 5,
    early_exaggeration_iter = 3L,
    n_iter = 4L,
    learning_rate = "auto",
    negative_gradient_method = "fft",
    n_threads = 2L,
    seed = 321L
  )

  expect_equal(dim(layout), c(nrow(x), 2L))
  expect_true(all(is.finite(layout)))
  cfg <- attr(layout, "fastEmbedR_config")
  expect_equal(cfg$method, "opentsne")
  expect_equal(cfg$optimizer, "opentsne_fitsne_fft_grid_sparse_knn")
  expect_equal(cfg$repulsion, "fft_grid")
  expect_equal(cfg$early_exaggeration_iter, 3L)
  expect_equal(cfg$n_iter, 4L)
  expect_equal(cfg$learning_rate, "auto_opt_sne_n_over_early_exaggeration")
  expect_equal(cfg$learning_rate_early, nrow(x) / cfg$early_exaggeration)
  expect_equal(cfg$learning_rate_normal, nrow(x) / cfg$early_exaggeration)
})

test_that("openTSNE auto configuration exposes opt-SNE policy metadata", {
  policy <- fastEmbedR:::tsne_auto_parameters_cpp(
    150L,
    30L,
    NA_real_,
    TRUE,
    "cpu",
    "exact"
  )
  expect_equal(policy$perplexity, 10)
  expect_equal(policy$learning_rate, 150 / policy$early_exaggeration)
  expect_true(policy$auto_kld_stop)
  expect_equal(policy$auto_iter_end, 5000)

  large_fft <- fastEmbedR:::tsne_auto_parameters_cpp(
    70000L,
    90L,
    NA_real_,
    TRUE,
    "cpu",
    "fft"
  )
  expect_equal(large_fft$perplexity, 30)
  expect_false(large_fft$auto_kld_stop)
})

test_that("removed embedding methods fail at the public KNN dispatcher", {
  set.seed(320)
  x <- matrix(rnorm(32L * 4L), 32L, 4L)
  knn <- faissR::nn(x, k = 10L, backend = "cpu")

  expect_error(embed_knn(knn, method = "tsne"), "opentsne", fixed = TRUE)
  expect_error(embed_knn(knn, method = "infotsne"), "opentsne", fixed = TRUE)
})

test_that("openTSNE exposes FFT and exact negative-gradient choices without Barnes-Hut or sampled GPU math", {
  set.seed(312)
  x <- matrix(rnorm(42L * 4L), 42L, 4L)
  knn <- faissR::nn(x, k = 13L, backend = "cpu")

  expect_error(
    embed_knn(
      knn,
      method = "opentsne",
      perplexity = 4,
      negative_gradient_method = "bh",
      early_exaggeration_iter = 2L,
      n_iter = 3L,
      n_threads = 2L,
      seed = 312L
    ),
    "removed"
  )
  expect_error(
    embed_knn(
      knn,
      method = "opentsne",
      perplexity = 4,
      negative_gradient_method = "sampled",
      early_exaggeration_iter = 2L,
      n_iter = 3L,
      n_threads = 2L,
      seed = 312L
    ),
    "changes the optimization mathematics"
  )

  exact <- embed_knn(
    knn,
    method = "opentsne",
    perplexity = 4,
    negative_gradient_method = "exact",
    early_exaggeration_iter = 2L,
    n_iter = 3L,
    n_threads = 2L,
    seed = 312L
  )
  expect_equal(attr(exact, "fastEmbedR_config")$repulsion, "pair_symmetric")

  fft <- embed_knn(
    knn,
    method = "opentsne",
    perplexity = 4,
    negative_gradient_method = "fft",
    early_exaggeration_iter = 2L,
    n_iter = 3L,
    n_threads = 2L,
    seed = 312L
  )
  expect_equal(attr(fft, "fastEmbedR_config")$repulsion, "fft_grid")
  expect_equal(attr(fft, "fastEmbedR_config")$optimizer, "opentsne_fitsne_fft_grid_sparse_knn")
})

test_that("opentsne has direct KNN input functions", {
  set.seed(322)
  x <- matrix(rnorm(54L * 5L), 54L, 5L)
  labels <- rep(1:3, length.out = nrow(x))
  knn <- faissR::nn(x, k = 19L, backend = "cpu")

  layout <- opentsne_knn(
    knn$indices,
    knn$distances,
    n_neighbors = 12L,
    perplexity = 3,
    early_exaggeration_iter = 2L,
    n_iter = 3L,
    n_threads = 2L,
    seed = 322L
  )
  expect_equal(dim(layout), c(nrow(x), 2L))
  expect_true(all(is.finite(layout)))
  cfg <- attr(layout, "fastEmbedR_config")
  expect_equal(cfg$method, "opentsne")
  expect_equal(cfg$n_neighbors, 12L)
  expect_equal(cfg$perplexity, 3)

  fit <- opentsne(
    knn,
    labels = labels,
    n_neighbors = 12L,
    perplexity = 3,
    early_exaggeration_iter = 2L,
    n_iter = 3L,
    silhouette_sample = NULL,
    preserve_sample = NULL,
    seed = 322L
  )
  expect_s3_class(fit, "fastEmbedR_embedding")
  expect_equal(dim(fit$layout), c(nrow(x), 2L))
  expect_equal(fit$parameters$input, "knn")
  expect_equal(fit$metrics$n_neighbors, 12L)
  expect_equal(fit$metrics$preprocess_elapsed, 0)
  expect_equal(fit$metrics$knn_elapsed, 0)
})

test_that("native Metal openTSNE runs FFT-grid without CPU fallback", {
  skip_if_not(fastEmbedR:::embedding_metal_available_cpp())
  skip_if_not(fastEmbedR:::metal_opentsne_native_available())
  skip_if_not(metal_available())

  old_grid <- Sys.getenv("FASTEMBEDR_TSNE_FFT_GRID", unset = NA_character_)
  Sys.setenv(FASTEMBEDR_TSNE_FFT_GRID = "32")
  on.exit({
    if (is.na(old_grid)) {
      Sys.unsetenv("FASTEMBEDR_TSNE_FFT_GRID")
    } else {
      Sys.setenv(FASTEMBEDR_TSNE_FFT_GRID = old_grid)
    }
  }, add = TRUE)

  set.seed(323)
  x <- matrix(rnorm(96L * 5L), 96L, 5L)
  knn <- faissR::nn(x, k = 16L, backend = "cpu")
  metal <- opentsne_knn(
    knn,
    n_neighbors = 15L,
    perplexity = 5,
    early_exaggeration_iter = 1L,
    n_iter = 2L,
    negative_gradient_method = "fft",
    backend = "metal",
    seed = 323L
  )
  expect_equal(dim(metal), c(nrow(x), 2L))
  expect_true(all(is.finite(metal)))
  cfg <- attr(metal, "fastEmbedR_config")
  expect_equal(cfg$backend, "metal")
  expect_equal(cfg$optimizer, "opentsne_fitsne_fft_grid_native_metal")
  expect_equal(cfg$repulsion, "fft_grid_metal")
  expect_equal(cfg$repulsion_block_size, 32L)
})

test_that("openTSNE GPU optimizers are native and fail clearly when unavailable", {
  set.seed(319)
  x <- matrix(rnorm(32L * 4L), 32L, 4L)
  knn <- faissR::nn(x, k = 10L, backend = "cpu")

  if (isTRUE(fastEmbedR:::embedding_metal_available_cpp()) &&
      isTRUE(fastEmbedR:::metal_opentsne_native_available())) {
    metal <- opentsne_knn(
      knn,
      perplexity = 3,
      early_exaggeration_iter = 1L,
      n_iter = 1L,
      negative_gradient_method = "exact",
      backend = "metal"
    )
    cfg <- attr(metal, "fastEmbedR_config")
    expect_equal(cfg$backend, "metal")
    expect_equal(cfg$optimizer, "opentsne_exact_sparse_native_metal")
    expect_equal(cfg$repulsion, "exact_metal")
    expect_equal(cfg$probabilities, "symmetric_sparse_knn_cpu_prepared_for_metal")
  } else {
    expect_error(
      opentsne_knn(
        knn,
        perplexity = 3,
        early_exaggeration_iter = 1L,
        n_iter = 1L,
      backend = "metal"
      ),
      "Native Metal openTSNE optimizer was requested",
      fixed = TRUE
    )
  }
  if (isTRUE(fastEmbedR:::embedding_cuda_available_cpp()) &&
      isTRUE(fastEmbedR:::cuda_opentsne_native_available())) {
    cuda <- opentsne_knn(
      knn,
      perplexity = 3,
      early_exaggeration_iter = 1L,
      n_iter = 1L,
      negative_gradient_method = "fft",
      backend = "cuda"
    )
    cfg <- attr(cuda, "fastEmbedR_config")
    expect_equal(cfg$backend, "cuda")
    expect_equal(cfg$optimizer, "opentsne_fitsne_fft_grid_native_cuda")
    expect_equal(cfg$repulsion, "fft_grid_cuda_cufft")
    expect_true(all(is.finite(cuda)))
  } else {
    expect_error(
      opentsne(
        knn,
        perplexity = 3,
        early_exaggeration_iter = 1L,
        n_iter = 1L,
        backend = "cuda"
      ),
      "CUDA"
    )
  }
})

test_that("opentsne can use cuVS for KNN without labelling the optimizer as GPU", {
  set.seed(313)
  x <- matrix(rnorm(36L * 4L), 36L, 4L)

  if (isTRUE(faissR::cuvs_available())) {
    fit <- opentsne(
      x,
      n_neighbors = 6L,
      perplexity = 2,
      early_exaggeration_iter = 2L,
      n_iter = 3L,
      backend = "cuda_cuvs_bruteforce",
      silhouette_sample = NULL,
      preserve_sample = NULL,
      keep_knn = TRUE
    )
    expect_equal(fit$parameters$method, "opentsne")
    expect_equal(fit$parameters$backend, "cpu")
    expect_equal(fit$parameters$nn_backend, "cuda_cuvs_bruteforce")
    expect_equal(dim(fit$knn$indices), c(nrow(x), 6L))
  } else {
    expect_error(
      opentsne(
        x,
        n_neighbors = 6L,
        perplexity = 2,
        early_exaggeration_iter = 2L,
        n_iter = 3L,
        backend = "cuda_cuvs_bruteforce",
        silhouette_sample = NULL,
        preserve_sample = NULL
      ),
      "cuVS"
    )
  }
})
