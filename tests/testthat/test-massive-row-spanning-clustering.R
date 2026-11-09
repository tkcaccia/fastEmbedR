rowspan_graph_fixture <- function(directory) {
    n <- 72L
    hub <- 37L
    neighbors <- lapply(seq_len(n), function(row) {
        if (row == hub) setdiff(seq_len(n), hub) else hub
    })
    offsets <- c(0L, cumsum(lengths(neighbors)))
    paths <- file.path(directory, c("offsets.u64", "indices.u32",
        "weights.f32", "parent.u32"))
    writeBin(as.integer(rbind(offsets, 0L)), paths[[1L]],
        size = 4L, endian = "little")
    writeBin(as.integer(unlist(neighbors)), paths[[2L]],
        size = 4L, endian = "little")
    writeBin(rep(0.5, tail(offsets, 1L)), paths[[3L]],
        size = 4L, endian = "little")
    writeBin(rep(1L, n), paths[[4L]], size = 4L,
        endian = "little")
    paths
}

rowspan_compare <- function(directory, paths, backend, method) {
    n <- 72L
    fits <- lapply(c(16L, 80L), function(budget) {
        output <- file.path(directory, paste0(backend, method,
            budget, ".u32"))
        if (method == "louvain") {
            native <- if (backend == "metal") {
                fastEmbedR:::massive_louvain_metal_level_cpp
            } else fastEmbedR:::massive_louvain_cuda_level_cpp
            return(native(paths[[1L]], paths[[2L]], paths[[3L]],
                n, output, 4L, 1, 8L, budget, 4L))
        }
        native <- if (backend == "metal") {
            fastEmbedR:::massive_leiden_metal_refine_cpp
        } else fastEmbedR:::massive_leiden_cuda_refine_cpp
        native(paths[[1L]], paths[[2L]], paths[[3L]],
            paths[[4L]], n, 1L, output, 1, 8L,
            budget, 4L, FALSE)
    })
    labels <- lapply(fits, function(fit) {
        readBin(fit$membership_path, integer(), n = n,
            size = 4L, endian = "little")
    })
    expect_identical(labels[[1L]], labels[[2L]])
    expect_identical(fits[[1L]]$moves, fits[[2L]]$moves)
    expect_identical(fits[[1L]]$backend, backend)
    if (method == "louvain") {
        expect_equal(fits[[1L]]$modularity_final,
            fits[[2L]]$modularity_final, tolerance = 1e-6)
        audited <- fastEmbedR:::massive_graph_modularity_cpp(
            paths[[1L]], paths[[2L]], paths[[3L]],
            fits[[1L]]$membership_path, n,
            fits[[1L]]$n_communities, 1, 8L, 16L, FALSE)
        expect_equal(audited$modularity,
            fits[[1L]]$modularity_final, tolerance = 1e-6)
    }
}

test_that("Metal clustering finalizes a row after its last upload", {
    skip_if_not(fastEmbedR:::graph_clustering_metal_available_cpp())
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    paths <- rowspan_graph_fixture(directory)
    for (method in c("louvain", "leiden")) {
        rowspan_compare(directory, paths, "metal", method)
    }
})

test_that("CUDA clustering finalizes a row after its last upload", {
    skip_if_not(fastEmbedR:::graph_clustering_cuda_available_cpp())
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    paths <- rowspan_graph_fixture(directory)
    for (method in c("louvain", "leiden")) {
        rowspan_compare(directory, paths, "cuda", method)
    }
})
