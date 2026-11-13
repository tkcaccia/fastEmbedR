test_that("persistent exact search uses faissR Flat index", {
    skip_if_not_installed("float")
    skip_if_not_installed("faissR")
    set.seed(611L)
    data <- float::fl(matrix(rnorm(120L * 6L), nrow = 120L))
    query <- data[1:7, , drop = FALSE]
    index <- faissr_flat_index_build_cpp(data, 2L)
    observed <- faissr_flat_index_search_cpp(index, query, 8L, 2L)
    reference <- faissR::nn(
        data, query, k = 8L, backend = "cpu", method = "flat",
        metric = "euclidean", n_threads = 2L
    )
    expect_identical(observed$indices, reference$indices)
    expect_equal(observed$distances, reference$distances)
    expect_identical(observed$backend, "cpu")
    expect_identical(observed$method, "faissr_flat_query_reused")
    expect_true(isTRUE(observed$index_reused))
    expect_error(
        faissr_flat_index_build_cpp(float::dbl(data), 2L),
        "float32"
    )
})
