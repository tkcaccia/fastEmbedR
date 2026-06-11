make_exact_knn <- function(x, k) {
  d <- as.matrix(stats::dist(x))
  diag(d) <- Inf
  idx <- t(apply(d, 1, order))[, seq_len(k), drop = FALSE]
  dst <- matrix(0, nrow(d), k)
  for (i in seq_len(nrow(d))) {
    dst[i, ] <- d[i, idx[i, ]]
  }
  list(idx = idx, dist = dst)
}

test_that("fast_knn_umap returns a finite layout", {
  set.seed(1)
  x <- matrix(rnorm(60), ncol = 3)
  knn <- make_exact_knn(x, 5L)

  layout <- fastEmbedR:::fast_knn_umap(knn$idx, knn$dist, seed = 7)
  expect_equal(dim(layout), c(nrow(x), 2L))
  expect_true(all(is.finite(layout)))
  expect_equal(colnames(layout), c("UMAP1", "UMAP2"))
})

test_that("masked UMAP refinement keeps unselected rows fixed", {
  set.seed(11)
  x <- matrix(rnorm(48), nrow = 12L)
  knn <- make_exact_knn(x, 4L)
  init <- matrix(rnorm(24), nrow = 12L)
  update_rows <- c(2L, 5L, 9L)

  refined <- fastEmbedR:::knn_umap_refine_masked_cpp(
    knn$idx,
    knn$dist,
    init,
    update_rows,
    8L,
    0.01,
    2L,
    0.1,
    1,
    1L,
    11L,
    FALSE
  )

  fixed_rows <- setdiff(seq_len(nrow(init)), update_rows)
  expect_equal(refined[fixed_rows, ], init[fixed_rows, ])
  expect_equal(dim(refined), dim(init))
  expect_true(all(is.finite(refined)))
})

test_that("row-subset UMAP refinement keeps non-row-id rows fixed", {
  set.seed(12)
  x <- matrix(rnorm(72), nrow = 18L)
  knn <- make_exact_knn(x, 5L)
  init <- matrix(rnorm(36), nrow = 18L)
  row_ids <- c(4L, 10L, 15L)

  refined <- fastEmbedR:::knn_umap_refine_rows_cpp(
    knn$idx[row_ids, , drop = FALSE],
    knn$dist[row_ids, , drop = FALSE],
    as.integer(row_ids),
    init,
    8L,
    0.01,
    2L,
    0.1,
    1,
    1L,
    12L,
    FALSE
  )

  fixed_rows <- setdiff(seq_len(nrow(init)), row_ids)
  expect_equal(refined[fixed_rows, ], init[fixed_rows, ])
  expect_equal(dim(refined), dim(init))
  expect_true(all(is.finite(refined)))
})

test_that("small UMAP defaults follow uwot-compatible settings", {
  set.seed(2)
  x <- matrix(rnorm(36), ncol = 3)
  knn <- make_exact_knn(x, 4L)

  layout <- fastEmbedR:::fast_knn_umap(knn$idx, knn$dist, seed = 8)
  cfg <- attr(layout, "fastEmbedR_config")

  expect_equal(cfg$n_epochs, 500L)
  expect_equal(cfg$min_dist, 0.01)
  expect_equal(cfg$negative_sample_rate, 5L)
  expect_equal(cfg$repulsion_strength, 1)
  expect_equal(cfg$learning_rate, 1)
  expect_equal(cfg$spectral_n_iter, 50L)
  expect_equal(cfg$n_threads, 1L)
  expect_equal(cfg$backend, "cpu")
  expect_equal(cfg$preset, "uwot_default")
  expect_equal(cfg$epoch_source, "uwot_size_rule")
  expect_equal(cfg$graph_storage, "native_csr_float_direct")
  expect_equal(cfg$graph_scales, "4")
  expect_equal(cfg$graph_mid_near_edges_per_point, 0L)
  expect_equal(cfg$graph_mid_near_weight, 0)
  expect_equal(cfg$graph_pruning, "none")
  expect_equal(cfg$graph_mode, "uwot_fuzzy_union")
  expect_equal(cfg$optimizer_math, "uwot_fast_sgd_compatible")
})

test_that("large UMAP defaults restart from uwot_fast_sgd-compatible settings", {
  low_k <- fastEmbedR:::fast_knn_umap_config(50000L, 15L, "cpu")
  high_k <- fastEmbedR:::fast_knn_umap_config(50000L, 50L, "cpu")
  cores <- suppressWarnings(parallel::detectCores(logical = FALSE))
  if (length(cores) != 1L || is.na(cores) || !is.finite(cores)) cores <- 1L
  expected_threads <- max(1L, min(4L, as.integer(cores)))

  for (cfg in list(low_k, high_k)) {
    expect_equal(cfg$preset, "uwot_fast_sgd_compatible")
    expect_equal(cfg$epoch_source, "uwot_fast_sgd_default")
    expect_equal(cfg$n_epochs, 200L)
    expect_equal(cfg$negative_sample_rate, 5L)
    expect_equal(cfg$min_dist, 0.01)
    expect_true(is.na(cfg$init_scale))
    expect_equal(cfg$n_threads, expected_threads)
    expect_equal(cfg$graph_storage, "native_csr_float_direct")
    expect_equal(cfg$graph_mid_near_edges_per_point, 0L)
    expect_equal(cfg$graph_pruning, "none")
    expect_equal(cfg$graph_mode, "uwot_fuzzy_union")
  }

  expect_equal(low_k$spectral_n_iter, 30L)
  expect_equal(high_k$spectral_n_iter, 20L)
})

test_that("CUDA fused UMAP transfer plan avoids init upload", {
  indices <- matrix(1L, nrow = 100L, ncol = 15L)
  distances <- matrix(0.5, nrow = 100L, ncol = 15L)
  plan <- fastEmbedR:::gpu_transfer_plan_knn_optimizer(
    backend = "cuda",
    indices = indices,
    distances = distances,
    init = NULL,
    n = 100L,
    n_components = 2L,
    objective = "umap",
    init_backend = "cuda_fused_spectral",
    graph_prep_backend = "cuda_fused_csr"
  )

  expect_equal(plan$gpu_transfer_host_to_device_count, 2L)
  expect_equal(plan$gpu_transfer_device_to_host_count, 1L)
  expect_false(plan$gpu_transfer_init_uploaded_once)
  expect_true(plan$gpu_transfer_init_computed_on_device)
  expect_true(plan$gpu_transfer_graph_prepared_on_device)
  expect_true(plan$gpu_transfer_embedding_returned_only_at_end)
  expect_match(plan$gpu_transfer_note, "uploads KNN once")
})

test_that("CUDA fused UMAP optimizer mode defaults to atomic but can be deterministic", {
  old_options <- options(fastEmbedR.cuda_optimizer = NULL)
  on.exit(options(old_options), add = TRUE)
  old_env <- Sys.getenv("FASTEMBEDR_CUDA_OPTIMIZER", unset = NA_character_)
  on.exit({
    if (is.na(old_env)) {
      Sys.unsetenv("FASTEMBEDR_CUDA_OPTIMIZER")
    } else {
      Sys.setenv(FASTEMBEDR_CUDA_OPTIMIZER = old_env)
    }
  }, add = TRUE)

  Sys.unsetenv("FASTEMBEDR_CUDA_OPTIMIZER")
  expect_equal(fastEmbedR:::fast_knn_umap_cuda_optimizer_mode(), "atomic")

  options(fastEmbedR.cuda_optimizer = "deterministic")
  expect_equal(fastEmbedR:::fast_knn_umap_cuda_optimizer_mode(), "deterministic")

  options(fastEmbedR.cuda_optimizer = NULL)
  Sys.setenv(FASTEMBEDR_CUDA_OPTIMIZER = "csr")
  expect_equal(fastEmbedR:::fast_knn_umap_cuda_optimizer_mode(), "deterministic")
})

test_that("graph helpers keep a single faithful KNN graph", {
  expect_equal(fastEmbedR:::fast_knn_umap_graph_scales(10L), 10L)
  expect_equal(fastEmbedR:::fast_knn_umap_graph_scales(50L), 50L)
  expect_equal(fastEmbedR:::fast_knn_umap_mid_near_count(50L), 0L)
  expect_equal(fastEmbedR:::fast_knn_umap_mid_near_weight(50L), 0)
  expect_equal(fastEmbedR:::fast_knn_umap_prune_fraction(50L), 0)
})

test_that("large supplied-KNN UMAP can pilot-tune optimizer settings only when requested", {
  old_options <- options(fastEmbedR.knn_pilot = TRUE)
  on.exit(options(old_options), add = TRUE)

  idx <- matrix(1L, nrow = 20000L, ncol = 50L)
  cfg <- fastEmbedR:::fast_knn_umap_config(20000L, 50L, "cpu")

  expect_true(fastEmbedR:::fast_knn_umap_should_auto_pilot(
    cfg = cfg,
    indices = idx,
    config_override = NULL,
    n_epochs = NULL
  ))
  expect_false(fastEmbedR:::fast_knn_umap_should_auto_pilot(
    cfg = cfg,
    indices = idx,
    config_override = list(n_epochs = 80L),
    n_epochs = NULL
  ))
})
