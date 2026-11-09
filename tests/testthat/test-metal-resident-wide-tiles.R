test_that("wide Metal UMAP processes rows split across edge tiles", {
    skip_if_not(fastEmbedR:::embedding_metal_available_cpp())
    skip_if_not_installed("float")
    offsets <- c(0, 9, 9, 9, 9, 9, 9)
    neighbors <- as.integer(rep(1:5, length.out = 9L))
    weights <- float::fl(rep(1, 9L))
    init <- cbind(seq_len(6L) / 20, seq_len(6L) / 30)
    run <- function(capacity) {
        fastEmbedR:::umap_metal_optimize_csr_wide_cpp(
            offsets, neighbors, weights, init, 2L, 0L,
            0.1, 0.1, 1, 42L, capacity
        )
    }
    split <- run(3L)
    whole <- run(9L)
    expect_s4_class(split, "float32")
    expect_identical(dim(split), c(6L, 2L))
    expect_true(all(is.finite(as.numeric(split))))
    expect_identical(attr(split, "csr_offset_dtype"), "uint64")
    expect_identical(attr(split, "metal_positive_edges"), "9")
    expect_identical(attr(whole, "metal_positive_edges"), "9")
    expect_identical(attr(split, "metal_negative_samples"), "0")
    expect_identical(attr(whole, "metal_negative_samples"), "0")
})

test_that("wide Metal UMAP rejects invalid global CSR offsets", {
    skip_if_not(fastEmbedR:::embedding_metal_available_cpp())
    skip_if_not_installed("float")
    offsets <- c(0, 2, 1, 2)
    init <- cbind(seq_len(3L) / 10, seq_len(3L) / 20)
    expect_error(fastEmbedR:::umap_metal_optimize_csr_wide_cpp(
        offsets, as.integer(c(1, 2)), float::fl(c(1, 1)),
        init, 2L, 0L, 0.1, 0.1, 1, 42L, 1L
    ), "offsets are not monotone")
})

test_that("wide binary 3D Metal UMAP visits every graph edge", {
    skip_if_not(fastEmbedR:::embedding_metal_available_cpp())
    skip_if_not_installed("float")
    rows <- 12L
    width <- 5L
    indices <- outer(seq_len(rows), seq_len(width),
        function(i, j) (i + j - 1L) %% rows + 1L)
    graph <- fastEmbedR:::umap_build_csr_graph(
        indices, NULL, 0L, width, width, 1L,
        graph_mode = "binary", allow_wide = TRUE,
        force_wide = TRUE
    )
    init <- cbind(seq_len(rows) / 20, seq_len(rows) / 30,
        seq_len(rows) / 40)
    fit <- fastEmbedR:::umap_metal_optimize_csr_wide_cpp(
        graph$offsets, graph$neighbors, graph$weights,
        init, 2L, 0L, 0.1, 0.1, 1, 42L, 3L
    )
    expect_identical(dim(fit), c(rows, 3L))
    expect_true(all(is.finite(as.numeric(fit))))
    expect_identical(attr(fit, "metal_positive_edges"),
        as.character(length(graph$neighbors)))
})
