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

test_that("nn returns exact cosine neighbors on CPU", {
  x <- matrix(c(
    1, 0,
    0, 1,
    -1, 0,
    1, 1
  ), ncol = 2, byrow = TRUE)

  out <- nn(x, k = 3L, backend = "cpu", metric = "cosine")

  expect_equal(attr(out, "backend"), "cpu")
  expect_equal(attr(out, "metric"), "cosine")
  expect_true(isTRUE(attr(out, "exact")))
  expect_equal(out$indices[1, ], c(1L, 4L, 2L))
  expect_equal(out$distances[1, ], c(0, 1 - 1 / sqrt(2), 1), tolerance = 1e-12)
})

test_that("nn returns exact correlation neighbors on CPU", {
  x <- matrix(c(
    1, 2, 3,
    1, 3, 5,
    3, 2, 1,
    5, 5, 5
  ), ncol = 3, byrow = TRUE)

  corr_dist <- function(a, b) {
    a <- a - mean(a)
    b <- b - mean(b)
    an <- sqrt(sum(a * a))
    bn <- sqrt(sum(b * b))
    if (an <= 0 && bn <= 0) return(0)
    if (an <= 0 || bn <= 0) return(1)
    1 - sum(a * b) / (an * bn)
  }
  expected <- outer(seq_len(nrow(x)), seq_len(nrow(x)), Vectorize(function(i, j) {
    corr_dist(x[i, ], x[j, ])
  }))

  out <- nn(x, k = 4L, backend = "cpu", metric = "correlation")

  expect_equal(attr(out, "backend"), "cpu")
  expect_equal(attr(out, "metric"), "correlation")
  expect_true(isTRUE(attr(out, "exact")))
  expect_equal(unname(out$indices), unname(t(apply(expected, 1, order))))
  for (i in seq_len(nrow(x))) {
    expect_equal(out$distances[i, ], expected[i, out$indices[i, ]], tolerance = 1e-12)
  }
})

test_that("non-euclidean metrics use only validated backend paths", {
  x <- scale(as.matrix(iris[1:20, 1:4]))

  auto <- nn(x, k = 4L, backend = "auto", metric = "cosine")
  expect_equal(attr(auto, "backend"), "cpu")
  expect_equal(attr(auto, "metric"), "cosine")

  auto_cor <- nn(x, k = 4L, backend = "auto", metric = "correlation")
  expect_equal(attr(auto_cor, "backend"), "cpu")
  expect_equal(attr(auto_cor, "metric"), "correlation")

  expect_error(
    nn(x, k = 4L, backend = "faiss", metric = "cosine"),
    "validated Euclidean-distance semantics only"
  )
  expect_error(
    nn(x, k = 4L, backend = "cuda_cuvs_nndescent", metric = "correlation"),
    "validated Euclidean-distance semantics only"
  )
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

test_that("removed CPU approximation backends are not public nn choices", {
  x <- matrix(rnorm(120L), nrow = 30L)
  expect_error(nn(x, k = 5L, backend = "cpu_clustered"), "should be one of")
  expect_error(nn(x, k = 5L, backend = "cpu_nndescent"), "should be one of")
  expect_error(nn(x, k = 5L, backend = "cpu_ivf"), "should be one of")
  expect_error(nn(x, k = 5L, backend = "cpu_annoy"), "should be one of")
  expect_error(nn(x, k = 5L, backend = "cpu_vptree"), "should be one of")
})

test_that("CPU approximate selector chooses FAISS NN-Descent, RcppHNSW, or exact CPU", {
  selected <- fastEmbedR:::select_cpu_approx_backend(12000L, 30L, 30L)
  expect_true(selected %in% c("faiss_nndescent", "hnsw", "cpu"))
  if (faiss_available()) {
    expect_equal(selected, "faiss_nndescent")
  } else if (requireNamespace("RcppHNSW", quietly = TRUE)) {
    expect_equal(selected, "hnsw")
  }
  expect_true(fastEmbedR:::should_use_auto_cpu_approx_self_knn(
    self_query = TRUE,
    n = 12000L,
    p = 30L,
    k = 30L,
    work_size = 12000 * 12000 * 30
  ))
})

test_that("RcppHNSW backend is public when installed", {
  skip_if_not_installed("RcppHNSW")
  set.seed(127)
  x <- rbind(
    matrix(rnorm(400, -2, 0.4), ncol = 8),
    matrix(rnorm(400, 2, 0.4), ncol = 8)
  )
  out <- nn(x, k = 10L, backend = "hnsw", n_threads = 2L)

  expect_equal(dim(out$indices), c(nrow(x), 10L))
  expect_equal(out$indices[, 1L], seq_len(nrow(x)))
  expect_equal(attr(out, "backend"), "hnsw")
  expect_false(isTRUE(attr(out, "exact")))
  expect_equal(attr(out, "approximation")$strategy, "RcppHNSW_hnswlib")
})

test_that("RcppHNSW backend supports correlation metric", {
  skip_if_not_installed("RcppHNSW")
  set.seed(128)
  x <- matrix(rnorm(80L * 10L), nrow = 80L)

  out <- nn(x, k = 6L, backend = "hnsw", metric = "correlation", n_threads = 2L)

  expect_equal(dim(out$indices), c(nrow(x), 6L))
  expect_equal(attr(out, "backend"), "hnsw")
  expect_equal(attr(out, "metric"), "correlation")
  expect_equal(attr(out, "approximation")$metric, "correlation")
  expect_true(all(is.finite(out$distances)))
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
    backend = "cuda_approx",
    self_query = TRUE,
    n = 1000L,
    p = 20L,
    k = 30L,
    exclude_self = FALSE,
    work_size = 1e6
  ))
  expect_false(fastEmbedR:::should_use_gpu_approx_self_knn(
    backend = "cuda_approx",
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

test_that("Metal KNN backend is not part of the cleaned nn API", {
  x <- matrix(rnorm(30), ncol = 3)
  expect_error(nn(x, x, k = 2, backend = "metal"), "should be one of")
  expect_error(nn(x, x, k = 2, backend = "metal_nndescent"), "should be one of")
  expect_error(nn(x, x, k = 2, backend = "metal_ivf"), "should be one of")
})

test_that("RcppHNSW backend is available when the suggested package is installed", {
  skip_if_not_installed("RcppHNSW")

  set.seed(144)
  x <- matrix(rnorm(120L * 6L), nrow = 120L)
  out <- nn(x, k = 8L, backend = "hnsw", n_threads = 2L)

  expect_equal(dim(out$indices), c(nrow(x), 8L))
  expect_equal(attr(out, "backend"), "hnsw")
  expect_false(isTRUE(attr(out, "exact")))
  expect_equal(attr(out, "approximation")$library, "RcppHNSW")
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
  fallback <- nn(x, x, k = 2, backend = "cuda")
  expect_true(attr(fallback, "backend") %in% c("faiss_nndescent", "hnsw", "cpu"))
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
