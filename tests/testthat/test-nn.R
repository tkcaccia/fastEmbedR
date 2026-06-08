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

test_that("parallel nn matches serial nn", {
  set.seed(12)
  data <- matrix(rnorm(200), ncol = 5)
  points <- matrix(rnorm(75), ncol = 5)

  serial <- nn(data, points, k = 4)
  parallel <- nn(data, points, k = 4, parallel = TRUE, cores = 2)

  expect_equal(parallel$indices, serial$indices)
  expect_equal(parallel$distances, serial$distances)
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

test_that("nn supports squared euclidean and manhattan distances", {
  data <- matrix(c(0, 0, 2, 0, 0, 3), ncol = 2, byrow = TRUE)
  point <- matrix(c(0, 0), nrow = 1)

  sq <- nn(data, point, k = 3, square = TRUE)
  expect_equal(sq$distances[1, ], c(0, 4, 9))

  man <- nn(data, point, k = 3, method = "manhattan")
  expect_equal(man$distances[1, ], c(0, 2, 3))
})

test_that("metal availability helper returns a logical scalar", {
  expect_type(metal_available(), "logical")
  expect_length(metal_available(), 1L)
})

test_that("cuda availability helper returns a logical scalar", {
  expect_type(cuda_available(), "logical")
  expect_length(cuda_available(), 1L)
})

test_that("Metal nn backend matches CPU euclidean results", {
  skip_if_not(metal_available())

  set.seed(14)
  data <- matrix(rnorm(500), ncol = 10)
  points <- matrix(rnorm(230), ncol = 10)

  cpu <- nn(data, points, k = 6, backend = "cpu", parallel = TRUE, cores = 2)
  gpu <- nn(data, points, k = 6, backend = "metal")

  expect_equal(attr(gpu, "backend"), "metal")
  expect_equal(gpu$indices, cpu$indices)
  expect_equal(gpu$distances, cpu$distances, tolerance = 1e-5)
})

test_that("Metal backend rejects unsupported metrics", {
  skip_if_not(metal_available())

  x <- matrix(rnorm(30), ncol = 3)
  expect_error(nn(x, x, k = 2, method = "manhattan", backend = "metal"), "euclidean")
})

test_that("CUDA nn backend matches CPU euclidean results", {
  skip_if_not(cuda_available())

  set.seed(15)
  data <- matrix(rnorm(500), ncol = 10)
  points <- matrix(rnorm(230), ncol = 10)

  cpu <- nn(data, points, k = 6, backend = "cpu", parallel = TRUE, cores = 2)
  gpu <- nn(data, points, k = 6, backend = "cuda")

  expect_equal(attr(gpu, "backend"), "cuda")
  expect_equal(gpu$indices, cpu$indices)
  expect_equal(gpu$distances, cpu$distances, tolerance = 1e-9)
})

test_that("CUDA backend reports unavailable runtime clearly", {
  skip_if(cuda_available())

  x <- matrix(rnorm(30), ncol = 3)
  expect_error(nn(x, x, k = 2, backend = "cuda"), "No CUDA")
})
