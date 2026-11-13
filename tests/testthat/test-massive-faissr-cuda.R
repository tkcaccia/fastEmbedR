test_that("single-device landmark CUDA KNN uses persistent faissR", {
    skip_if_not_installed("float")
    skip_if_not(faissR::faiss_gpu_available())
    set.seed(1147)
    reference <- matrix(rnorm(43L * 6L), ncol = 6L)
    query <- matrix(rnorm(19L * 6L), ncol = 6L)
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    reference_path <- file.path(directory, "reference.f32")
    query_path <- file.path(directory, "query.f32")
    writeBin(as.vector(t(reference)), reference_path,
        size = 4L, endian = "little")
    writeBin(as.vector(t(query)), query_path,
        size = 4L, endian = "little")
    found <- massive_landmark_knn(
        massive_matrix(query_path, 19L, 6L),
        massive_matrix(reference_path, 43L, 6L),
        5L, file.path(directory, "knn"), backend = "cuda",
        method = "exact", chunk_rows = 7L,
        memory_limit = "256MB")
    expected <- test_exact_knn(reference, query, 5L,
        n_threads = 1L, metric = "euclidean")
    observed <- fastEmbedR:::massive_read_knn_rows(found, n = 19L)
    expect_equal(observed$indices, expected$indices)
    expect_equal(observed$distances, expected$distances,
        tolerance = 1e-4)
    expect_identical(found$knn_provider, "faissR")
    expect_identical(found$reference_storage, "R_float32")
    expect_identical(found$query_storage, "R_float32")
    expect_identical(fastEmbedR:::massive_knn_query_storage(
        list(format = "memory"),
        list(method = "exact", backend = "cuda"),
        list(provider = "faissR")), "R_float32")
    expect_true(found$index_reused)
    expect_true(found$recall_audited)
})

test_that("single-device IVF returns distances for its selected IDs", {
    skip_if_not_installed("float")
    skip_if_not(faissR::faiss_gpu_available())
    set.seed(1148)
    reference <- matrix(rnorm(10000L * 8L), ncol = 8L)
    query <- matrix(rnorm(18L * 8L), ncol = 8L)
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    reference_path <- file.path(directory, "reference.f32")
    query_path <- file.path(directory, "query.f32")
    writeBin(as.vector(t(reference)), reference_path,
        size = 4L, endian = "little")
    writeBin(as.vector(t(query)), query_path,
        size = 4L, endian = "little")
    found <- massive_landmark_knn(
        massive_matrix(query_path, 18L, 8L),
        massive_matrix(reference_path, 10000L, 8L),
        5L, file.path(directory, "knn"), backend = "cuda",
        method = "ivf", chunk_rows = 7L,
        memory_limit = "256MB")
    observed <- fastEmbedR:::massive_read_knn_rows(found, n = 18L)
    expect_true(all(observed$indices >= 1L))
    expect_true(all(observed$indices <= nrow(reference)))
    for (row in seq_len(nrow(query))) {
        selected <- reference[observed$indices[row, ], , drop = FALSE]
        distances <- sqrt(rowSums((selected - matrix(query[row, ],
            nrow = 5L, ncol = 8L, byrow = TRUE))^2))
        expect_equal(observed$distances[row, ], distances,
            tolerance = 1e-4)
    }
    expect_identical(found$knn_provider, "faissR")
    expect_identical(found$method, "ivf")
    expect_false(found$recall_audited)
    expect_true(found$nlist > 0L && found$nprobe > 0L)
})
