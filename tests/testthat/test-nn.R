test_that("nn returns exact euclidean neighbors", {
  x <- matrix(c(
    0, 0,
    1, 0,
    0, 2,
    3, 0
  ), ncol = 2, byrow = TRUE)

  out <- nn(x, x, k = 3)
  expect_equal(dim(out$indices), c(4L, 3L))
  expect_equal(dim(out$distances), c(4L, 3L))
  expect_equal(out$indices[, 1], seq_len(nrow(x)))
  expect_equal(out$distances[, 1], rep(0, nrow(x)))

  d <- as.matrix(stats::dist(x))
  expected_idx <- t(apply(d, 1, order))[, 1:3]
  expected_dst <- matrix(0, nrow(x), 3)
  for (i in seq_len(nrow(x))) {
    expected_dst[i, ] <- d[i, expected_idx[i, ]]
  }
  expect_equal(unname(out$indices), unname(expected_idx))
  expect_equal(unname(out$distances), unname(expected_dst))
})

test_that("nn chooses a practical default k and prints clearly", {
  set.seed(101)
  x <- matrix(rnorm(60), nrow = 20L)

  out <- nn(x, backend = "cpu")

  expect_s3_class(out, "fastEmbedR_nn")
  expect_equal(dim(out$indices), c(nrow(x), fastEmbedR:::auto_k(x, include_self = TRUE)))
  expect_equal(attr(out, "backend"), "cpu")
  expect_true(isTRUE(attr(out, "exact")))
  expect_true(isTRUE(attr(out, "self_query")))
  expect_output(print(out), "fastEmbedR KNN")
})

test_that("automatic nn is deterministic", {
  set.seed(12)
  data <- matrix(rnorm(200), ncol = 5)
  points <- matrix(rnorm(75), ncol = 5)

  first <- nn(data, points, k = 4)
  second <- nn(data, points, k = 4)

  expect_equal(second$indices, first$indices)
  expect_equal(second$distances, first$distances)
})

test_that("CPU nn handles non-small Euclidean work", {
  set.seed(121)
  data <- matrix(rnorm(200L * 30L), nrow = 200L)
  points <- matrix(rnorm(200L * 30L), nrow = 200L)

  out <- nn(data, points, k = 12L, backend = "cpu")

  expect_equal(dim(out$indices), c(nrow(points), 12L))
  expect_true(all(is.finite(out$distances)))
  expect_equal(attr(out, "backend"), "cpu")
  expect_true(isTRUE(attr(out, "exact")))
})

test_that("CPU nn row-major distance layout matches column-major fallback", {
  old_row_major <- Sys.getenv("FASTEMBEDR_NN_ROW_MAJOR", unset = NA_character_)
  old_fortran <- Sys.getenv("FASTEMBEDR_USE_FORTRAN_NN", unset = NA_character_)
  on.exit({
    if (is.na(old_row_major)) {
      Sys.unsetenv("FASTEMBEDR_NN_ROW_MAJOR")
    } else {
      Sys.setenv(FASTEMBEDR_NN_ROW_MAJOR = old_row_major)
    }
    if (is.na(old_fortran)) {
      Sys.unsetenv("FASTEMBEDR_USE_FORTRAN_NN")
    } else {
      Sys.setenv(FASTEMBEDR_USE_FORTRAN_NN = old_fortran)
    }
  }, add = TRUE)

  set.seed(1211)
  data <- matrix(rnorm(140L * 11L), nrow = 140L)
  points <- matrix(rnorm(65L * 11L), nrow = 65L)

  Sys.setenv(
    FASTEMBEDR_USE_FORTRAN_NN = "0",
    FASTEMBEDR_NN_ROW_MAJOR = "1"
  )
  row_major <- nn(data, points, k = 9L, backend = "cpu", n_threads = 2L)

  Sys.setenv(FASTEMBEDR_NN_ROW_MAJOR = "0")
  column_major <- nn(data, points, k = 9L, backend = "cpu", n_threads = 2L)

  expect_equal(attr(row_major, "memory_layout"), "row_major_contiguous")
  expect_equal(attr(column_major, "memory_layout"), "r_column_major")
  expect_true(isTRUE(attr(row_major, "row_major_copy")))
  expect_false(isTRUE(attr(column_major, "row_major_copy")))
  expect_equal(row_major$indices, column_major$indices)
  expect_equal(row_major$distances, column_major$distances, tolerance = 1e-12)
})

test_that("clustered self KNN reports approximation and preserves useful neighbors", {
  set.seed(122)
  n_per <- 80L
  labels <- rep(seq_len(3L), each = n_per)
  centers <- c(-4, 0, 4)
  x <- matrix(rnorm(length(labels) * 6L, sd = 0.45), ncol = 6L)
  x <- x + matrix(rep(centers[labels], 6L), ncol = 6L)
  k <- 12L

  exact <- fastEmbedR:::nn_without_self(x, k = k, backend = "cpu")
  clustered <- fastEmbedR:::clustered_self_knn(x, k = k, exclude_self = TRUE, seed = 122L)
  clustered <- fastEmbedR:::finish_nn_result(
    clustered,
    "cpu_clustered",
    k,
    TRUE,
    exact = FALSE
  )

  overlap <- mean(vapply(
    seq_len(nrow(x)),
    function(i) length(intersect(exact$indices[i, ], clustered$indices[i, ])) / k,
    numeric(1)
  ))

  expect_equal(dim(clustered$indices), c(nrow(x), k))
  expect_equal(dim(clustered$distances), c(nrow(x), k))
  expect_equal(attr(clustered, "backend"), "cpu_clustered")
  expect_false(isTRUE(attr(clustered, "exact")))
  expect_gt(overlap, 0.45)
  expect_output(print(clustered), "exact: false")
})

test_that("approximate landmark projection KNN matches exact when window covers all landmarks", {
  set.seed(124)
  landmarks <- matrix(rnorm(50L * 6L), nrow = 50L)
  queries <- matrix(rnorm(35L * 6L), nrow = 35L)
  k <- 7L

  exact <- nn(landmarks, queries, k = k, backend = "cpu")
  approx <- fastEmbedR:::landmark_projection_knn_approx_cpp(
    landmarks,
    queries,
    as.integer(k),
    4L,
    nrow(landmarks),
    124L,
    FALSE,
    1L
  )

  expect_equal(approx$indices, exact$indices)
  expect_equal(approx$distances, exact$distances, tolerance = 1e-6)
  expect_equal(approx$n_projections, 4L)
  expect_equal(approx$window, nrow(landmarks))
  expect_equal(approx$n_threads, 1L)
  expect_equal(approx$score_threads, 1L)
  expect_gt(approx$visited_stamp_mb_per_thread, 0)
})

test_that("subset landmark candidate KNN matches full candidate rows", {
  set.seed(125)
  x <- matrix(rnorm(48L * 5L), nrow = 48L, ncol = 5L)
  landmark_rows <- seq(1L, 48L, length.out = 12L)
  projection <- nn(x[landmark_rows, , drop = FALSE], x, k = 5L, backend = "cpu")
  rows <- c(3L, 11L, 24L, 39L)
  k <- 6L

  full <- fastEmbedR:::landmark_candidate_knn_cpp(
    x,
    projection$indices,
    as.integer(k),
    3L,
    4L,
    FALSE,
    1L
  )
  subset <- fastEmbedR:::landmark_candidate_knn_subset_cpp(
    x,
    projection$indices,
    as.integer(rows),
    as.integer(k),
    3L,
    4L,
    FALSE,
    1L
  )

  expect_equal(subset$row_ids, rows)
  expect_equal(subset$indices, full$indices[rows, , drop = FALSE])
  expect_equal(subset$distances, full$distances[rows, , drop = FALSE], tolerance = 1e-8)
})

test_that("clustered self KNN is not selected automatically", {
  expect_false(fastEmbedR:::should_use_clustered_self_knn(
    backend = "auto",
    self_query = TRUE,
    n = 6000L,
    p = 20L,
    k = 30L,
    work_size = 7.2e8
  ))
  expect_false(fastEmbedR:::should_use_clustered_self_knn(
    backend = "cpu",
    self_query = TRUE,
    n = 6000L,
    p = 20L,
    k = 30L,
    work_size = 7.2e8
  ))
  expect_false(fastEmbedR:::should_use_clustered_self_knn(
    backend = "auto",
    self_query = FALSE,
    n = 6000L,
    p = 20L,
    k = 30L,
    work_size = 7.2e8
  ))
})

test_that("clustered CPU backend is public for self-KNN", {
  set.seed(123)
  x <- rbind(
    matrix(rnorm(120, 0, 0.4), 40L, 3L),
    matrix(rnorm(120, 3, 0.4), 40L, 3L)
  )

  out <- nn(x, k = 10L, backend = "cpu_clustered")

  expect_equal(dim(out$indices), c(nrow(x), 10L))
  expect_equal(out$indices[, 1L], seq_len(nrow(x)))
  expect_equal(out$distances[, 1L], rep(0, nrow(x)))
  expect_equal(attr(out, "backend"), "cpu_clustered")
  expect_false(isTRUE(attr(out, "exact")))
  expect_error(
    fastEmbedR:::nn_compute(
      x,
      x[1:5, , drop = FALSE],
      k = 4L,
      backend = "cpu_clustered",
      points_missing = FALSE,
      exclude_self = FALSE
    ),
    "self-KNN"
  )
})

test_that("NN-descent CPU backend is public, deterministic, and recall-aware", {
  set.seed(124)
  n_per <- 90L
  labels <- rep(seq_len(4L), each = n_per)
  centers <- matrix(rnorm(4L * 8L, sd = 3), 4L, 8L)
  x <- matrix(rnorm(length(labels) * 8L, sd = 0.35), ncol = 8L) +
    centers[labels, , drop = FALSE]
  k <- 15L

  exact <- fastEmbedR:::nn_without_self(x, k = k, backend = "cpu")
  first <- fastEmbedR:::nn_without_self(x, k = k, backend = "cpu_nndescent")
  second <- fastEmbedR:::nn_without_self(x, k = k, backend = "cpu_nndescent")
  recall <- fastEmbedR:::knn_recall(first, exact, k)

  expect_equal(first$indices, second$indices)
  expect_equal(first$distances, second$distances)
  expect_equal(dim(first$indices), c(nrow(x), k))
  expect_equal(attr(first, "backend"), "cpu_nndescent")
  expect_false(isTRUE(attr(first, "exact")))
  expect_gt(recall$recall_at_k, 0.75)
  public <- nn(x, k = 10L, backend = "cpu_nndescent")
  expect_equal(attr(public, "backend"), "cpu_nndescent")
  expect_equal(dim(public$indices), c(nrow(x), 10L))
  expect_equal(public$indices[, 1L], seq_len(nrow(x)))
})

test_that("CPU approximate selector chooses a public native backend", {
  expect_equal(fastEmbedR:::select_cpu_approx_backend(12000L, 30L, 30L), "cpu_nndescent")
  expect_equal(fastEmbedR:::select_cpu_approx_backend(70000L, 50L, 50L), "cpu_nndescent")
  expect_true(fastEmbedR:::should_use_auto_cpu_approx_self_knn(
    self_query = TRUE,
    n = 12000L,
    p = 30L,
    k = 30L,
    work_size = 12000 * 12000 * 30
  ))
  expect_false(fastEmbedR:::should_use_nndescent_self_knn(
    backend = "auto",
    self_query = TRUE,
    n = 12000L,
    p = 30L,
    k = 30L,
    exclude_self = TRUE,
    work_size = 12000 * 12000 * 30
  ))
  expect_true(fastEmbedR:::should_use_nndescent_self_knn(
    backend = "cpu_nndescent",
    self_query = TRUE,
    n = 12000L,
    p = 30L,
    k = 30L,
    exclude_self = TRUE,
    work_size = 12000 * 12000 * 30
  ))
  expect_false(fastEmbedR:::should_use_nndescent_self_knn(
    backend = "cpu",
    self_query = TRUE,
    n = 12000L,
    p = 30L,
    k = 30L,
    exclude_self = TRUE,
    work_size = 12000 * 12000 * 30
  ))
  expect_false(fastEmbedR:::should_use_nndescent_self_knn(
    backend = "auto",
    self_query = FALSE,
    n = 12000L,
    p = 30L,
    k = 30L,
    exclude_self = TRUE,
    work_size = 12000 * 12000 * 30
  ))
})

test_that("native IVF CPU backend is public and recall-aware", {
  old_options <- options(
    fastEmbedR.ivf_nlist = 24L,
    fastEmbedR.ivf_nprobe = 8L
  )
  on.exit(options(old_options), add = TRUE)

  set.seed(126)
  n_per <- 70L
  labels <- rep(seq_len(4L), each = n_per)
  centers <- matrix(rnorm(4L * 10L, sd = 3), 4L, 10L)
  x <- matrix(rnorm(length(labels) * 10L, sd = 0.35), ncol = 10L) +
    centers[labels, , drop = FALSE]
  k <- 12L

  exact <- fastEmbedR:::nn_without_self(x, k = k, backend = "cpu")
  ivf <- fastEmbedR:::nn_without_self(x, k = k, backend = "cpu_ivf")
  recall <- fastEmbedR:::knn_recall(ivf, exact, k)
  public <- nn(x, k = k + 1L, backend = "cpu_ivf")

  expect_equal(dim(ivf$indices), c(nrow(x), k))
  expect_equal(attr(ivf, "backend"), "cpu_ivf")
  expect_false(isTRUE(attr(ivf, "exact")))
  expect_true(is.list(attr(ivf, "approximation")))
  expect_equal(attr(ivf, "approximation")$strategy, "ivf_flat_native")
  expect_gt(recall$recall_at_k, 0.45)
  expect_equal(public$indices[, 1L], seq_len(nrow(x)))
  expect_equal(attr(public, "backend"), "cpu_ivf")
})

test_that("native Annoy-style CPU backend is public and thread-aware", {
  old_options <- options(
    fastEmbedR.annoy_n_trees = 16L,
    fastEmbedR.annoy_leaf_size = 48L,
    fastEmbedR.annoy_search_k = 768L
  )
  on.exit(options(old_options), add = TRUE)

  set.seed(129)
  n_per <- 80L
  labels <- rep(seq_len(4L), each = n_per)
  centers <- matrix(rnorm(4L * 8L, sd = 3), 4L, 8L)
  x <- matrix(rnorm(length(labels) * 8L, sd = 0.45), ncol = 8L) +
    centers[labels, , drop = FALSE]
  k <- 12L

  exact <- fastEmbedR:::nn_without_self(x, k = k, backend = "cpu", n_threads = 4L)
  annoy <- fastEmbedR:::nn_without_self(x, k = k, backend = "cpu_annoy", n_threads = 4L)
  public <- nn(x, k = k + 1L, backend = "cpu_annoy", n_threads = 4L)
  recall <- fastEmbedR:::knn_recall(annoy, exact, k)

  expect_equal(dim(annoy$indices), c(nrow(x), k))
  expect_equal(attr(annoy, "backend"), "cpu_annoy")
  expect_false(isTRUE(attr(annoy, "exact")))
  expect_equal(attr(annoy, "approximation")$strategy, "annoy_style_random_projection_forest_native")
  expect_gt(recall$recall_at_k, 0.35)
  expect_equal(public$indices[, 1L], seq_len(nrow(x)))
  expect_equal(attr(public, "backend"), "cpu_annoy")
})

test_that("FAISS-style IVF backend is implemented natively in package code", {
  old_options <- options(
    fastEmbedR.ivf_nlist = 16L,
    fastEmbedR.ivf_nprobe = 6L
  )
  on.exit(options(old_options), add = TRUE)

  set.seed(127)
  x <- rbind(
    matrix(rnorm(400, -2, 0.4), ncol = 8),
    matrix(rnorm(400, 2, 0.4), ncol = 8)
  )
  out <- nn(x, k = 10L, backend = "cpu_faiss_ivf")

  expect_equal(dim(out$indices), c(nrow(x), 10L))
  expect_equal(out$indices[, 1L], seq_len(nrow(x)))
  expect_equal(attr(out, "backend"), "cpu_faiss_ivf")
  expect_false(isTRUE(attr(out, "exact")))
  expect_equal(attr(out, "approximation")$strategy, "faiss_style_ivf_flat_native")
})

test_that("real FAISS C++ backend is either exact or clearly unavailable", {
  set.seed(137)
  x <- matrix(rnorm(120L * 6L), nrow = 120L)
  k <- 7L

  if (faiss_available()) {
    exact <- fastEmbedR:::nn_without_self(x, k = k, backend = "cpu", n_threads = 2L)
    out <- fastEmbedR:::nn_without_self(x, k = k, backend = "faiss", n_threads = 2L)
    recall <- fastEmbedR:::knn_recall(out, exact, k = k)

    expect_equal(dim(out$indices), c(nrow(x), k))
    expect_equal(attr(out, "backend"), "faiss")
    expect_true(isTRUE(attr(out, "exact")))
    expect_equal(attr(out, "faiss")$index_type, "IndexFlatL2")
    expect_equal(recall$recall_at_k, 1)
  } else {
    expect_error(nn(x, k = k + 1L, backend = "faiss"), "FAISS")
  }
})

test_that("real FAISS IVF backend records approximate index metadata", {
  set.seed(138)
  x <- matrix(rnorm(160L * 5L), nrow = 160L)
  k <- 8L

  if (faiss_available()) {
    old_options <- options(
      fastEmbedR.faiss_nlist = 16L,
      fastEmbedR.faiss_nprobe = 4L
    )
    on.exit(options(old_options), add = TRUE)

    out <- nn(x, k = k + 1L, backend = "faiss_ivf", n_threads = 2L)
    expect_equal(dim(out$indices), c(nrow(x), k + 1L))
    expect_equal(out$indices[, 1L], seq_len(nrow(x)))
    expect_equal(attr(out, "backend"), "faiss_ivf")
    expect_false(isTRUE(attr(out, "exact")))
    expect_equal(attr(out, "approximation")$strategy, "faiss_IndexIVFFlat")
    expect_equal(attr(out, "approximation")$nlist, 16L)
    expect_equal(attr(out, "approximation")$nprobe, 4L)
  } else {
    expect_error(nn(x, k = k + 1L, backend = "faiss_ivf"), "FAISS")
  }
})

test_that("VP-tree CPU backend matches exact self-KNN", {
  set.seed(128)
  x <- matrix(rnorm(90L * 5L), nrow = 90L)
  k <- 8L

  exact <- fastEmbedR:::nn_without_self(x, k = k, backend = "cpu")
  vp <- fastEmbedR:::nn_without_self(x, k = k, backend = "cpu_vptree")
  public <- nn(x, k = k + 1L, backend = "cpu_vptree")

  expect_equal(vp$indices, exact$indices)
  expect_equal(vp$distances, exact$distances, tolerance = 1e-6)
  expect_equal(attr(vp, "backend"), "cpu_vptree")
  expect_true(isTRUE(attr(vp, "exact")))
  expect_equal(public$indices[, 1L], seq_len(nrow(x)))
})

test_that("GPU approximate KNN helpers require explicit backend requests", {
  expect_false(fastEmbedR:::should_use_gpu_approx_self_knn(
    backend = "auto",
    self_query = TRUE,
    n = 100000L,
    p = 20L,
    k = 30L,
    exclude_self = FALSE,
    work_size = 1e9
  ))
  expect_true(fastEmbedR:::should_use_gpu_approx_self_knn(
    backend = "gpu_approx",
    self_query = TRUE,
    n = 1000L,
    p = 20L,
    k = 30L,
    exclude_self = FALSE,
    work_size = 1e6
  ))
  expect_false(fastEmbedR:::should_use_gpu_approx_self_knn(
    backend = "gpu_approx",
    self_query = FALSE,
    n = 100000L,
    p = 20L,
    k = 30L,
    exclude_self = FALSE,
    work_size = 1e9
  ))
  params <- fastEmbedR:::gpu_approx_params(50000L, 30L)
  expect_gte(params$anchors, 128L)
  expect_gte(params$projection_k, 12L)
  expect_gte(params$query_cols, params$bucket_cols)
})

test_that("approximate KNN recall metadata is attached against exact subset", {
  set.seed(131)
  x <- matrix(rnorm(80L * 5L), nrow = 80L)
  exact <- fastEmbedR:::nn_without_self(x, k = 6L, backend = "cpu")
  approx <- fastEmbedR:::finish_nn_result(exact, "test_approx", 6L, TRUE, exact = FALSE)
  approx <- fastEmbedR:::attach_knn_recall_subset(
    approx,
    data = x,
    k = 6L,
    exclude_self = TRUE,
    seed = 131L
  )
  recall <- attr(approx, "recall")
  expect_s3_class(approx, "fastEmbedR_nn")
  expect_true(is.data.frame(recall))
  expect_equal(recall$k, 6L)
  expect_equal(recall$recall_at_k, 1)
  expect_gt(recall$sample_size, 0L)
  expect_output(print(approx), "recall@6")
})

test_that("nn matches brute-force euclidean neighbors for query points", {
  set.seed(13)
  data <- matrix(rnorm(120), ncol = 6)
  points <- matrix(rnorm(60), ncol = 6)

  ours <- nn(data, points, k = 5)
  d <- matrix(0, nrow(points), nrow(data))
  for (i in seq_len(nrow(points))) {
    d[i, ] <- rowSums((data - matrix(points[i, ], nrow(data), ncol(data), byrow = TRUE))^2)
  }
  expected_idx <- t(apply(d, 1L, order))[, 1:5, drop = FALSE]
  expected_dst <- matrix(0, nrow(points), 5L)
  for (i in seq_len(nrow(points))) {
    expected_dst[i, ] <- sqrt(d[i, expected_idx[i, ]])
  }

  expect_equal(ours$indices, expected_idx)
  expect_equal(ours$distances, expected_dst, tolerance = 1e-12)
})

test_that("Fortran CPU nn path matches C++ fallback", {
  old <- Sys.getenv("FASTEMBEDR_USE_FORTRAN_NN", unset = NA_character_)
  on.exit({
    if (is.na(old)) {
      Sys.unsetenv("FASTEMBEDR_USE_FORTRAN_NN")
    } else {
      Sys.setenv(FASTEMBEDR_USE_FORTRAN_NN = old)
    }
  })

  set.seed(130)
  data <- matrix(rnorm(240), nrow = 40L)
  points <- matrix(rnorm(90), nrow = 15L)

  Sys.setenv(FASTEMBEDR_USE_FORTRAN_NN = "1")
  fortran <- nn(data, points, k = 7L, backend = "cpu")
  Sys.setenv(FASTEMBEDR_USE_FORTRAN_NN = "0")
  cpp <- nn(data, points, k = 7L, backend = "cpu")

  expect_equal(fortran$indices, cpp$indices)
  expect_equal(fortran$distances, cpp$distances, tolerance = 1e-12)

  Sys.setenv(FASTEMBEDR_USE_FORTRAN_NN = "1")
  fortran_self <- fastEmbedR:::nn_without_self(data, k = 6L, backend = "cpu")
  Sys.setenv(FASTEMBEDR_USE_FORTRAN_NN = "0")
  cpp_self <- fastEmbedR:::nn_without_self(data, k = 6L, backend = "cpu")

  expect_equal(fortran_self$indices, cpp_self$indices)
  expect_equal(fortran_self$distances, cpp_self$distances, tolerance = 1e-12)
})

test_that("removed nn compatibility options are not accepted", {
  data <- matrix(c(0, 0, 2, 0, 0, 3), ncol = 2, byrow = TRUE)
  point <- matrix(c(0, 0), nrow = 1)

  expect_error(nn(data, point, k = 3, square = TRUE), "unused")
  expect_error(nn(data, point, k = 3, method = "manhattan"), "unused")
})

test_that("metal availability helper returns a logical scalar", {
  expect_type(metal_available(), "logical")
  expect_length(metal_available(), 1L)
})

test_that("cuda availability helper returns a logical scalar", {
  expect_type(cuda_available(), "logical")
  expect_length(cuda_available(), 1L)
})

test_that("faiss availability helper returns a logical scalar", {
  expect_type(faiss_available(), "logical")
  expect_length(faiss_available(), 1L)
})

test_that("cuvs availability helper returns a logical scalar", {
  expect_type(cuvs_available(), "logical")
  expect_length(cuvs_available(), 1L)
})

test_that("backend_info reports native availability without crashing", {
  info <- backend_info()
  expect_s3_class(info, "data.frame")
  expect_true(all(c("cpu", "faiss", "cuvs", "cuda", "metal") %in% info$backend))
  expect_true(all(c(
    "available",
    "knn_available",
    "embedding_available",
    "device",
    "runtime"
  ) %in% names(info)))
  expect_true(isTRUE(info$available[info$backend == "cpu"]))
  expect_false(any(is.na(info$available)))

  cuda_info <- fastEmbedR:::cuda_device_info_json_cpp()
  expect_type(cuda_info, "character")
  expect_length(cuda_info, 1L)
  expect_match(cuda_info, "available")
})

test_that("Metal nn backend matches CPU euclidean results", {
  skip_if_not(metal_available())

  set.seed(14)
  data <- matrix(rnorm(500), ncol = 10)
  points <- matrix(rnorm(230), ncol = 10)

  cpu <- nn(data, points, k = 6, backend = "cpu")
  gpu <- nn(data, points, k = 6, backend = "metal")

  expect_equal(attr(gpu, "backend"), "metal")
  expect_equal(gpu$indices, cpu$indices)
  expect_equal(gpu$distances, cpu$distances, tolerance = 1e-5)
})

test_that("Metal row-major projection KNN matches CPU euclidean results", {
  skip_if_not(metal_available())

  set.seed(141)
  data <- matrix(rnorm(1000 * 10), ncol = 10)
  points <- matrix(rnorm(1000 * 10), ncol = 10)

  cpu <- nn(data, points, k = 8L, backend = "cpu")
  gpu <- nn(data, points, k = 8L, backend = "metal")

  expect_equal(attr(gpu, "backend"), "metal")
  expect_equal(attr(gpu, "metal_kernel"), "row_major_exact")
  expect_equal(gpu$indices, cpu$indices)
  expect_equal(gpu$distances, cpu$distances, tolerance = 1e-5)
})

test_that("Metal approximate self KNN routes to native NN-descent and returns recall metadata", {
  skip_if_not(metal_available())
  old_options <- options(
    fastEmbedR.gpu_approx_recall = TRUE,
    fastEmbedR.gpu_approx_recall_sample = 40L,
    fastEmbedR.gpu_approx_anchors = 80L,
    fastEmbedR.gpu_approx_projection_k = 16L,
    fastEmbedR.metal_nndescent_iters = 1L,
    fastEmbedR.metal_nndescent_sources = 4L,
    fastEmbedR.metal_nndescent_neighbors = 5L
  )
  on.exit(options(old_options), add = TRUE)
  set.seed(132)
  x <- matrix(rnorm(90L * 6L), nrow = 90L)
  out <- nn(x, k = 8L, backend = "metal_approx")
  recall <- attr(out, "recall")

  expect_equal(dim(out$indices), c(nrow(x), 8L))
  expect_equal(out$indices[, 1L], seq_len(nrow(x)))
  expect_equal(attr(out, "backend"), "metal_nndescent")
  expect_equal(attr(out, "metal_kernel"), "row_candidate_knn")
  expect_false(isTRUE(attr(out, "exact")))
  expect_equal(attr(out, "approximation")$strategy, "mlx_vis_adaptive_seeded_nndescent_native_metal")
  expect_true(is.data.frame(recall))
  expect_equal(recall$k, 7L)
  expect_gt(recall$recall_at_k, 0.35)
})

test_that("Metal IVF and FAISS-style IVF use native GPU path", {
  skip_if_not(metal_available())
  old_options <- options(
    fastEmbedR.gpu_approx_recall = TRUE,
    fastEmbedR.gpu_approx_recall_sample = 30L,
    fastEmbedR.gpu_approx_anchors = 64L,
    fastEmbedR.gpu_approx_projection_k = 12L
  )
  on.exit(options(old_options), add = TRUE)

  set.seed(133)
  x <- matrix(rnorm(96L * 6L), nrow = 96L)
  ivf <- nn(x, k = 9L, backend = "metal_ivf")
  faiss <- nn(x, k = 9L, backend = "metal_faiss")

  expect_equal(attr(ivf, "backend"), "metal_ivf")
  expect_equal(attr(ivf, "approximation")$strategy, "ivf_flat_native")
  expect_false(isTRUE(attr(ivf, "exact")))
  expect_equal(attr(faiss, "backend"), "metal_faiss_ivf")
  expect_equal(attr(faiss, "approximation")$strategy, "faiss_style_ivf_flat_native")
  expect_false(isTRUE(attr(faiss, "exact")))
})

test_that("Metal grid KNN uses the native FastGraph-style candidate path", {
  skip_if_not(metal_available())
  old_options <- options(
    fastEmbedR.grid_dims = 4L,
    fastEmbedR.grid_bins = 5L,
    fastEmbedR.grid_radius = 1L
  )
  on.exit(options(old_options), add = TRUE)

  set.seed(134)
  x <- rbind(
    matrix(rnorm(320, -2, 0.25), ncol = 4),
    matrix(rnorm(320, 2, 0.25), ncol = 4)
  )
  k <- 10L
  exact <- fastEmbedR:::nn_without_self(x, k = k, backend = "cpu", n_threads = 4L)
  grid <- fastEmbedR:::nn_without_self(x, k = k, backend = "metal_grid")
  recall <- fastEmbedR:::knn_recall(grid, exact, k)

  expect_equal(dim(grid$indices), c(nrow(x), k))
  expect_equal(attr(grid, "backend"), "metal_grid")
  expect_equal(attr(grid, "metal_kernel"), "grid_bin_candidate")
  expect_equal(attr(grid, "approximation")$strategy, "fastgraph_style_grid_candidate_native")
  expect_false(isTRUE(attr(grid, "exact")))
  expect_gt(recall$recall_at_k, 0.8)
})

test_that("Metal NN-descent refinement uses the native row-candidate kernel", {
  skip_if_not(metal_available())
  old_options <- options(
    fastEmbedR.metal_nndescent_iters = 1L,
    fastEmbedR.metal_nndescent_sources = 4L,
    fastEmbedR.metal_nndescent_neighbors = 5L
  )
  on.exit(options(old_options), add = TRUE)

  set.seed(135)
  x <- rbind(
    matrix(rnorm(300, -1.5, 0.35), ncol = 5),
    matrix(rnorm(300, 1.5, 0.35), ncol = 5)
  )
  k <- 8L
  exact <- fastEmbedR:::nn_without_self(x, k = k, backend = "cpu", n_threads = 4L)
  refined <- fastEmbedR:::nn_without_self(x, k = k, backend = "metal_nndescent")
  recall <- fastEmbedR:::knn_recall(refined, exact, k)

  expect_equal(dim(refined$indices), c(nrow(x), k))
  expect_equal(attr(refined, "backend"), "metal_nndescent")
  expect_equal(attr(refined, "metal_kernel"), "row_candidate_knn")
  expect_equal(
    attr(refined, "approximation")$strategy,
    "mlx_vis_adaptive_seeded_nndescent_native_metal"
  )
  expect_false(isTRUE(attr(refined, "exact")))
  expect_gt(recall$recall_at_k, 0.65)
})

test_that("MLX-inspired NN-descent candidate builder tracks active rows", {
  indices <- matrix(
    c(
      2L, 3L, 4L,
      1L, 3L, 5L,
      2L, 4L, 6L,
      3L, 5L, 1L,
      4L, 6L, 2L,
      5L, 1L, 3L
    ),
    nrow = 6L,
    byrow = TRUE
  )
  flags <- matrix(FALSE, nrow = 6L, ncol = 3L)
  flags[c(1L, 3L, 5L), 1L] <- TRUE

  candidates <- fastEmbedR:::nndescent_candidate_matrix_mlx_cpp(
    indices,
    flags,
    n_sources = 2L,
    n_neighbors = 2L,
    use_reverse = TRUE,
    active_only = TRUE
  )

  expect_equal(nrow(candidates), nrow(indices))
  expect_gte(ncol(candidates), ncol(indices))
  expect_equal(attr(candidates, "active_rows"), 3L)
  expect_true(isTRUE(attr(candidates, "use_reverse")))
  expect_true(isTRUE(attr(candidates, "active_only")))
  expect_equal(attr(candidates, "sources"), 2L)
  expect_equal(attr(candidates, "neighbors"), 2L)
  expect_true(all(candidates[candidates > 0L] >= 1L))
  expect_true(all(candidates[candidates > 0L] <= nrow(indices)))
})

test_that("Metal backend keeps simplified Euclidean API", {
  skip_if_not(metal_available())

  x <- matrix(rnorm(30), ncol = 3)
  expect_error(nn(x, x, k = 2, method = "manhattan", backend = "metal"), "unused")
})

test_that("CUDA nn backend matches CPU euclidean results", {
  skip_if_not(cuda_available())

  set.seed(15)
  data <- matrix(rnorm(500), ncol = 10)
  points <- matrix(rnorm(230), ncol = 10)

  cpu <- nn(data, points, k = 6, backend = "cpu")
  gpu <- nn(data, points, k = 6, backend = "cuda")

  expect_equal(attr(gpu, "backend"), "cuda")
  expect_equal(gpu$indices, cpu$indices)
  expect_equal(gpu$distances, cpu$distances, tolerance = 1e-5)
})

test_that("CUDA NN-descent requests use RAPIDS cuVS", {
  skip_if_not(cuvs_available())

  set.seed(136)
  x <- rbind(
    matrix(rnorm(300, -1.4, 0.35), ncol = 5),
    matrix(rnorm(300, 1.4, 0.35), ncol = 5)
  )
  k <- 8L
  exact <- fastEmbedR:::nn_without_self(x, k = k, backend = "cpu", n_threads = 4L)
  refined <- fastEmbedR:::nn_without_self(x, k = k, backend = "cuda_cuvs_nndescent")
  recall <- fastEmbedR:::knn_recall(refined, exact, k)

  expect_equal(dim(refined$indices), c(nrow(x), k))
  expect_equal(attr(refined, "backend"), "cuda_cuvs_nndescent")
  expect_equal(
    attr(refined, "approximation")$strategy,
    "rapids_cuvs_nndescent"
  )
  expect_false(isTRUE(attr(refined, "exact")))
  expect_gt(recall$recall_at_k, 0.65)
})

test_that("CUDA backend reports unavailable runtime clearly", {
  skip_if(cuda_available())

  x <- matrix(rnorm(30), ncol = 3)
  expect_error(nn(x, x, k = 2, backend = "cuda"), "No CUDA")
  expect_error(nn(x, x, k = 2, backend = "cuda_ivf"), "No CUDA")
  expect_error(nn(x, x, k = 2, backend = "cuda_faiss"), "No CUDA")
})

test_that("cuVS backend reports unavailable runtime clearly", {
  skip_if(cuvs_available())

  x <- matrix(rnorm(30), ncol = 3)
  expect_error(nn(x, x, k = 2, backend = "cuda_cuvs"), "cuVS")
  expect_error(nn(x, x, k = 2, backend = "cuda_cuvs_bruteforce"), "cuVS")
  expect_error(nn(x, x, k = 2, backend = "cuda_cuvs_nndescent"), "cuVS")
  expect_error(nn(x, x, k = 2, backend = "cuda_approx"), "cuVS")
  expect_error(nn(x, x, k = 2, backend = "cuda_nndescent"), "cuVS")
})
