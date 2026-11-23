test_that("Metal file spectral uses the saved CSR and preserves the layout", {
    skip_if_not(Sys.info()[["sysname"]] == "Darwin")
    skip_if_not(embedding_metal_available_cpp())
    work <- tempfile("spectral-csr-")
    dir.create(work)
    on.exit(unlink(work, recursive = TRUE))
    files <- file.path(work, c("offsets", "neighbors", "weights"))
    n <- 37L
    edges <- lapply(seq_len(n), function(i) {
        sort(as.integer(1L + (i - 1L + c(-3L:-1L, 1L:3L)) %% n))
    })
    weights <- unlist(lapply(seq_len(n), function(i) {
        .2 + ((i + edges[[i]]) %% 17L) / 25
    }), use.names = FALSE)
    offsets <- as.integer(c(0L, cumsum(lengths(edges))))
    writeBin(as.integer(rbind(offsets, 0L)), files[[1L]],
        size = 4L, endian = "little")
    writeBin(unlist(edges, use.names = FALSE), files[[2L]],
        size = 4L, endian = "little")
    writeBin(weights, files[[3L]], size = 4L, endian = "little")
    run <- function(output, dim, metal, budget = 64 * 1024^2) {
        massive_spectral_init_stream_cpp(files[[1L]], files[[2L]],
            files[[3L]], n, file.path(work, output), dim, 12L, 19L,
            32L, budget, metal)
    }
    layout <- function(fit) {
        matrix(readBin(fit$output, "numeric", n * fit$dimensions,
            size = 4L, endian = "little"), nrow = n, byrow = TRUE)
    }
    for (dim in c(2L, 3L)) {
        cpu <- run(paste0("cpu-", dim), dim, FALSE)
        metal <- run(paste0("metal-", dim), dim, TRUE)
        expect_identical(metal$backend_used, "native_metal_mapped_spectral")
        expect_identical(metal$graph_access, "read_only_mmap")
        expect_identical(metal$cpu_graph_passes, 1L)
        expect_gt(metal$metal_owned_workspace_bytes, 0)
        expect_identical(dim(layout(metal)), c(n, dim))
        expect_equal(as.vector(dist(layout(metal))),
            as.vector(dist(layout(cpu))), tolerance = 1e-4)
    }
    expect_error(run("insufficient", 2L, TRUE, budget = 1024),
        "exceeds RAM budget")
    expect_false(file.exists(file.path(work, "insufficient")))
})
