test_that("automatic HNSW calibration starts at k, explicit ef stays fixed", {
    x <- list(nrow = 100L, ncol = 2L)
    state <- list(index = list(method = "faissr_hnsw_persistent",
        pointer = NULL, efSearch = 150L))
    setup <- list(k = 10L, audit_rows = 0L, workers = 1L,
        target_recall = 0.90, ef_search = 0L)
    received <- integer()
    calibrate <- function(pointer, query, self_ids, k, target,
            initial_ef, max_ef, workers) {
        received <<- c(initial_ef, max_ef)
        list(ef_search = initial_ef, pilot_rows = nrow(query),
            max_ef_search = max_ef,
            pilot_recall = 1, pilot_guard_recall = 0.9,
            pilot_reference = "full_cpu_double_l2", target_met = TRUE,
            target_recall = target)
    }
    read_rows <- function(x, row, count, bytes, exact) {
        matrix(c(row, row + 1), nrow = 1L)
    }
    automatic <- with_mocked_bindings(
        fastEmbedR:::massive_hnsw_calibrate_shard(
            x, state, setup, 1L, 100L),
        massive_read_rows_cpp = read_rows,
        faissr_hnsw_index_calibrate_cpp = calibrate,
        .package = "fastEmbedR")
    expect_identical(received, c(10L, 100L))
    expect_identical(automatic$index$efSearch, 10L)
    setup$ef_search <- 80L
    explicit <- with_mocked_bindings(
        fastEmbedR:::massive_hnsw_calibrate_shard(
            x, state, setup, 1L, 100L),
        massive_read_rows_cpp = read_rows,
        faissr_hnsw_index_calibrate_cpp = calibrate,
        .package = "fastEmbedR")
    expect_identical(received, c(80L, 80L))
    expect_identical(explicit$index$efSearch, 80L)
})

test_that("HNSW bridge calls faissR's precise private calibrator", {
    set.seed(20261116)
    data <- float::fl(matrix(rnorm(200L * 6L), nrow = 200L))
    index <- fastEmbedR:::faissr_hnsw_index_build_cpp(
        data, 200L, 6L, 5L, 1L)
    rows <- seq.int(4L, 190L, by = 11L)
    pilot <- fastEmbedR:::faissr_hnsw_index_calibrate_cpp(
        index$pointer, data[rows, , drop = FALSE], rows,
        5L, 0.90, 10L, 150L, 1L)
    expect_identical(pilot$max_ef_search, 150L)
    expect_identical(pilot$pilot_reference, "full_cpu_double_l2")
    expect_identical(pilot$pilot_rows, length(rows))
    expect_true(pilot$target_met)
})
