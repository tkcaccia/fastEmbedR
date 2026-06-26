test_that("auto_k chooses bounded size-aware neighborhoods", {
  expect_equal(fastEmbedR:::auto_k(100L), 15L)
  expect_equal(fastEmbedR:::auto_k(100L, include_self = TRUE), 16L)
  expect_equal(fastEmbedR:::auto_k(1000L), 30L)
  expect_equal(fastEmbedR:::auto_k(20000L), 50L)
  expect_equal(fastEmbedR:::auto_k(matrix(0, 8L, 3L)), 7L)
  expect_error(fastEmbedR:::auto_k(1L), "at least two")
})

test_that("automatic embedding K is openTSNE focused", {
  expect_equal(fastEmbedR:::auto_embedding_k(1000L, "opentsne"), 30L)
  expect_equal(fastEmbedR:::auto_embedding_k(1000L, include_self = TRUE), 31L)
})

test_that("preprocessing PCA uses fastPLS-style RSVD", {
  set.seed(39)
  x <- matrix(rnorm(80L * 40L), 80L, 40L)
  pre <- fastEmbedR:::prepare_embedding_data(
    x,
    standardize = TRUE,
    pca_dims = 5L,
    seed = 39L,
    backend = "cpu"
  )

  expect_equal(dim(pre$data), c(80L, 5L))
  expect_true(all(is.finite(pre$data)))
  expect_equal(pre$preprocess$pca_method, "rsvd")
  expect_equal(pre$preprocess$pca_backend, "cpu_rsvd")
})

test_that("float32 matrix input is accepted and preserved without preprocessing", {
  skip_if_not_installed("float")
  set.seed(41)
  x <- float::fl(matrix(rnorm(60L), 20L, 3L))
  pre <- fastEmbedR:::prepare_embedding_data(
    x,
    standardize = FALSE,
    pca_dims = NULL,
    seed = 41L,
    backend = "cpu"
  )

  expect_s4_class(pre$data, "float32")
  expect_equal(dim(pre$data), c(20L, 3L))
  expect_equal(pre$preprocess$preprocess_backend, "none")
  expect_match(pre$preprocess$preprocess_backend_reason, "float32_input_preserved")

  x_bad <- x
  x_bad[1, 1] <- Inf
  expect_error(
    fastEmbedR:::prepare_embedding_data(
      x_bad,
      standardize = FALSE,
      pca_dims = NULL,
      seed = 41L,
      backend = "cpu"
    ),
    "finite"
  )
})

test_that("opentsne and umap accept float32 matrix input", {
  skip_if_not_installed("float")
  set.seed(42)
  x <- float::fl(matrix(rnorm(120L), 30L, 4L))

  fit_tsne <- opentsne(
    x,
    perplexity = 1,
    early_exaggeration_iter = 2L,
    n_iter = 3L,
    seed = 42L,
    n_threads = 2L
  )
  expect_s3_class(fit_tsne, "fastEmbedR_embedding")
  expect_equal(dim(fit_tsne$layout), c(30L, 2L))
  expect_s4_class(fit_tsne$layout, "float32")
  expect_equal(fit_tsne$preprocess$preprocess_backend, "none")

  fit_umap <- umap(
    x,
    n_neighbors = 5L,
    seed = 42L,
    n_threads = 2L
  )
  expect_s3_class(fit_umap, "fastEmbedR_embedding")
  expect_equal(dim(fit_umap$layout), c(30L, 2L))
  expect_s4_class(fit_umap$layout, "float32")
  expect_equal(fit_umap$parameters$preprocess$preprocess_backend, "none")
})

test_that("double matrix input keeps a double layout despite float internal KNN", {
  set.seed(46)
  x <- matrix(rnorm(120L), 30L, 4L)

  fit_tsne <- opentsne(
    x,
    perplexity = 1,
    early_exaggeration_iter = 2L,
    n_iter = 3L,
    seed = 46L,
    n_threads = 2L
  )
  expect_type(fit_tsne$layout, "double")
  expect_false(inherits(fit_tsne$layout, "float32"))

  fit_umap <- umap(
    x,
    n_neighbors = 5L,
    seed = 46L,
    n_threads = 2L
  )
  expect_type(fit_umap$layout, "double")
  expect_false(inherits(fit_umap$layout, "float32"))
})

test_that("layout finalizer preserves requested type and float32 memory footprint", {
  skip_if_not_installed("float")
  set.seed(48)
  raw_layout <- matrix(rnorm(20000L), 10000L, 2L)

  float_layout <- fastEmbedR:::finalize_embedding_layout(
    raw_layout,
    "openTSNE",
    return_float32 = TRUE
  )
  double_layout <- fastEmbedR:::finalize_embedding_layout(
    float_layout,
    "openTSNE",
    return_float32 = FALSE
  )

  expect_s4_class(float_layout, "float32")
  expect_type(double_layout, "double")
  expect_equal(dim(float_layout), dim(double_layout))
  expect_equal(colnames(float_layout), c("openTSNE1", "openTSNE2"))
  expect_equal(colnames(double_layout), c("openTSNE1", "openTSNE2"))
  expect_equal(attr(float_layout, "precision"), "float32")
  expect_equal(attr(double_layout, "precision"), "double")

  float_bytes <- as.numeric(object.size(float_layout))
  double_bytes <- as.numeric(object.size(double_layout))
  expect_lt(float_bytes, double_bytes)
  expect_lt(float_bytes / double_bytes, 0.65)
})

test_that("float32 KNN bridge passes float32 data directly to faissR nn", {
  skip_if_not_installed("float")
  set.seed(44)
  x <- float::fl(matrix(rnorm(40L), 10L, 4L))
  old_nn <- .fastembedr_faissr_cache[["nn"]]
  on.exit({
    if (is.null(old_nn)) {
      rm(list = "nn", envir = .fastembedr_faissr_cache)
    } else {
      .fastembedr_faissr_cache[["nn"]] <- old_nn
    }
  }, add = TRUE)

  captured <- new.env(parent = emptyenv())
  .fastembedr_faissr_cache[["nn"]] <- function(data,
                                               k,
                                               exclude_self = FALSE,
                                               backend = "cpu",
                                               method = "auto",
                                               metric = "euclidean",
                                               output = "double",
                                               n_threads = NULL) {
    captured$class <- class(data)
    captured$is_float32 <- inherits(data, "float32")
    captured$k <- k
    captured$exclude_self <- exclude_self
    captured$output <- output
    idx <- cbind(
      c(2L:10L, 1L),
      c(3L:10L, 1L, 2L),
      c(4L:10L, 1L, 2L, 3L)
    )[, seq_len(k), drop = FALSE]
    dst <- matrix(seq_len(nrow(data) * k) / 100, nrow(data), k)
    if (identical(output, "float")) dst <- float::fl(dst)
    structure(
      list(indices = idx, distances = dst),
      class = "faissR_nn",
      backend = "mock"
    )
  }

  out <- fastembedr_nn_without_self(
    x,
    k = 3L,
    backend = "cpu",
    output = "float",
    n_threads = 2L
  )

  expect_true(captured$is_float32)
  expect_true("float32" %in% captured$class)
  expect_equal(captured$k, 3L)
  expect_true(captured$exclude_self)
  expect_equal(captured$output, "float")
  expect_equal(dim(out$indices), c(10L, 3L))
  expect_s4_class(out$distances, "float32")
})

test_that("one-call CPU and Metal KNN policy uses high-recall faissR HNSW", {
  expect_equal(
    fastembedr_embedding_nn_policy("cpu"),
    list(backend = "cpu", method = "hnsw", tuning = "auto", target_recall = 0.99)
  )
  expect_equal(
    fastembedr_embedding_nn_policy("metal"),
    list(backend = "cpu", method = "hnsw", tuning = "auto", target_recall = 0.99)
  )
  expect_equal(
    fastembedr_embedding_nn_policy("cuda"),
    list(backend = "cuda", method = "auto", tuning = "auto", target_recall = NULL)
  )
})

test_that("KNN bridge forwards HNSW tuning arguments when faissR supports them", {
  set.seed(45)
  x <- matrix(rnorm(40L), 10L, 4L)
  old_nn <- .fastembedr_faissr_cache[["nn"]]
  on.exit({
    if (is.null(old_nn)) {
      rm(list = "nn", envir = .fastembedr_faissr_cache)
    } else {
      .fastembedr_faissr_cache[["nn"]] <- old_nn
    }
  }, add = TRUE)

  captured <- new.env(parent = emptyenv())
  .fastembedr_faissr_cache[["nn"]] <- function(data,
                                               k,
                                               exclude_self = FALSE,
                                               backend = "cpu",
                                               method = "auto",
                                               metric = "euclidean",
                                               tuning = "auto",
                                               target_recall = 0.95,
                                               output = "double",
                                               n_threads = NULL) {
    captured$backend <- backend
    captured$method <- method
    captured$tuning <- tuning
    captured$target_recall <- target_recall
    idx <- cbind(c(2L:10L, 1L), c(3L:10L, 1L, 2L))
    dst <- matrix(seq_len(nrow(data) * k) / 100, nrow(data), k)
    structure(list(indices = idx[, seq_len(k), drop = FALSE], distances = dst), backend = "mock")
  }

  policy <- fastembedr_embedding_nn_policy("metal")
  out <- fastembedr_nn_without_self(
    x,
    k = 2L,
    backend = policy$backend,
    method = policy$method,
    tuning = policy$tuning,
    target_recall = policy$target_recall
  )

  expect_equal(captured$backend, "cpu")
  expect_equal(captured$method, "hnsw")
  expect_equal(captured$tuning, "auto")
  expect_equal(captured$target_recall, 0.99)
  expect_equal(dim(out$indices), c(10L, 2L))
})

test_that("float32 KNN distances use less memory than double distances", {
  skip_if_not_installed("float")
  set.seed(49)
  distances <- matrix(stats::runif(10000L), 1000L, 10L)
  knn_double <- list(
    indices = matrix(sample.int(1000L, 10000L, replace = TRUE), 1000L, 10L),
    distances = distances
  )
  knn_float <- knn_double
  knn_float$distances <- float::fl(distances)

  expect_type(knn_double$distances, "double")
  expect_s4_class(knn_float$distances, "float32")

  double_bytes <- as.numeric(object.size(knn_double$distances))
  float_bytes <- as.numeric(object.size(knn_float$distances))
  expect_lt(float_bytes, double_bytes)
  expect_lt(float_bytes / double_bytes, 0.65)
})

test_that("UMAP CSR graph weights stay float32 through prepared optimizer path", {
  skip_if_not_installed("float")
  idx <- cbind(
    c(2L, 1L, 1L, 1L, 2L, 3L, 4L, 5L, 6L, 7L),
    c(3L, 3L, 2L, 2L, 3L, 4L, 5L, 6L, 7L, 8L),
    c(4L, 4L, 5L, 3L, 4L, 5L, 6L, 7L, 8L, 9L)
  )
  dst <- float::fl(matrix(seq_len(length(idx)) / 100, nrow(idx), ncol(idx)))
  graph <- fastEmbedR:::umap_build_csr_graph(
    idx,
    dst,
    col_start = 0L,
    n_cols = ncol(idx),
    edge_budget = ncol(idx),
    n_threads = 2L,
    graph_mode = "fuzzy"
  )

  expect_s4_class(graph$weights, "float32")
  expect_s4_class(graph$epochs_per_sample, "float32")

  cfg <- fastEmbedR:::fast_knn_umap_config(nrow(idx), ncol(idx), backend = "cpu")
  init <- fastEmbedR:::umap_init_from_csr_graph(
    graph,
    n_components = 2L,
    cfg = cfg,
    seed = 45L
  )
  layout <- fastEmbedR:::fast_knn_umap_csr_init_cpp(
    graph$offsets,
    graph$neighbors,
    graph$weights,
    init,
    2L,
    cfg$min_dist,
    cfg$negative_sample_rate,
    cfg$learning_rate,
    cfg$repulsion_strength,
    2L,
    45L,
    FALSE
  )
  expect_equal(dim(layout), c(nrow(idx), 2L))
  expect_true(all(is.finite(layout)))
})

test_that("opentsne convenience wrapper runs the automatic KNN workflow", {
  set.seed(43)
  x <- rbind(matrix(rnorm(30), 10L, 3L), matrix(rnorm(30, 2), 10L, 3L))
  labels <- rep(1:2, each = 10L)

  fit <- opentsne(
    x,
    perplexity = 1,
    early_exaggeration_iter = 2L,
    n_iter = 3L,
    seed = 43L
  )

  expect_s3_class(fit, "fastEmbedR_embedding")
  expect_equal(fit$parameters$method, "opentsne")
  expect_equal(dim(fit$layout), c(nrow(x), 2L))
  expect_equal(colnames(fit$layout), c("openTSNE1", "openTSNE2"))
})

test_that("high-level embeddings avoid retaining KNN matrices by default", {
  set.seed(47)
  x <- matrix(rnorm(90), 30L, 3L)

  compact <- opentsne(
    x,
    perplexity = 1,
    early_exaggeration_iter = 2L,
    n_iter = 3L,
    seed = 47L
  )
  retained <- opentsne(
    x,
    perplexity = 1,
    early_exaggeration_iter = 2L,
    n_iter = 3L,
    seed = 47L,
    keep_knn = TRUE
  )

  expect_null(compact$knn)
  expect_equal(dim(retained$knn$indices), c(nrow(x), 1L))
  expect_equal(dim(retained$knn$distances), c(nrow(x), 1L))
})
