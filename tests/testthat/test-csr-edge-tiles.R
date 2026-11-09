test_that("CSR tiles preserve rows across a 32-bit boundary", {
    compiler <- Sys.which("c++")
    skip_if(!nzchar(compiler), "C++ compiler unavailable")
    source <- testthat::test_path("..", "test_csr_edge_tiles.cpp")
    skip_if(!file.exists(source), "native CSR fixture unavailable")
    output <- tempfile("csr-tiles-")
    on.exit(unlink(output), add = TRUE)
    build <- suppressWarnings(system2(compiler,
        c("-std=c++17", "-O2", shQuote(source), "-o",
            shQuote(output)), stdout = TRUE, stderr = TRUE))
    build_status <- attr(build, "status")
    if (is.null(build_status)) build_status <- 0L
    expect_identical(build_status, 0L,
        info = paste(build, collapse = "\n"))
    if (!file.exists(output)) return(invisible())
    run <- suppressWarnings(system2(output, stdout = TRUE,
        stderr = TRUE))
    run_status <- attr(run, "status")
    if (is.null(run_status)) run_status <- 0L
    expect_identical(run_status, 0L,
        info = paste(run, collapse = "\n"))
})

test_that("file-backed CSR validation crosses a tile within one row", {
    skip_if_not_installed("fastEmbedR")
    prefix <- tempfile("csr-wide-row-")
    paths <- paste0(prefix, c(".offsets.u64", ".indices.u32",
        ".weights.f32"))
    on.exit(unlink(paths), add = TRUE)
    counts <- c(0L, rep(70000L, 70001L))
    words <- as.vector(rbind(counts, integer(length(counts))))
    writeBin(words, paths[[1L]], size = 4L, endian = "little")
    writeBin(2:70001, paths[[2L]], size = 4L,
        endian = "little")
    writeBin(rep(0.5, 70000L), paths[[3L]], size = 4L,
        endian = "little")
    checked <- fastEmbedR:::massive_validate_csr_graph_cpp(
        paths[[1L]], paths[[2L]], paths[[3L]], 70001)
    expect_equal(checked$n_edges, 70000)
    expect_equal(checked$max_degree, 70000)
    expect_true(checked$validated)
})

test_that("32-bit resident and 64-bit file CSR have the same edges", {
    indices <- outer(seq_len(12L), seq_len(4L),
        function(i, j) (i + j - 1L) %% 12L + 1L)
    distances <- matrix(as.double(rep(seq_len(4L), each = 12L)),
        12L, 4L)
    graph <- fastEmbedR:::umap_build_csr_graph(indices,
        distances, 0L, 4L, 4L, 1L)
    offsets <- as.double(graph$offsets)
    expect_equal(offsets[[length(offsets)]],
        length(graph$neighbors))
    prefix <- tempfile("csr-index-parity-")
    paths <- paste0(prefix, c(".offsets.u64", ".indices.u32",
        ".weights.f32"))
    on.exit(unlink(paths), add = TRUE)
    words <- as.integer(c(rbind(offsets, rep(0, length(offsets)))))
    writeBin(words, paths[[1L]], size = 4L, endian = "little")
    writeBin(as.integer(graph$neighbors + 1L), paths[[2L]],
        size = 4L, endian = "little")
    writeBin(as.numeric(graph$weights), paths[[3L]],
        size = 4L, endian = "little")
    observed <- fastEmbedR:::massive_read_csr_graph_edges_cpp(
        paths[[1L]], paths[[2L]], paths[[3L]], 12,
        "stream", 1, 12L, 128e6)
    expected_from <- rep(seq_len(12L), diff(graph$offsets))
    expect_equal(observed$from, as.double(expected_from))
    expect_equal(observed$to, graph$neighbors + 1L)
    expect_equal(observed$weight, as.numeric(graph$weights),
        tolerance = 1e-6)
})
