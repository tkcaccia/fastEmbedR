check_accelerated_file_spectral <- function(backend) {
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
    run <- function(output, dim, engine, budget = 64 * 1024^2) {
        massive_spectral_init_stream_cpp(files[[1L]], files[[2L]],
            files[[3L]], n, file.path(work, output), dim, 12L, 19L,
            32L, budget, use_metal = engine == "metal",
            use_cuda = engine == "cuda")
    }
    layout <- function(fit) {
        matrix(readBin(fit$output, "numeric", n * fit$dimensions,
            size = 4L, endian = "little"), nrow = n, byrow = TRUE)
    }
    for (dim in c(2L, 3L)) {
        cpu <- run(paste0("cpu-", dim), dim, "cpu")
        fit <- run(paste0(backend, "-", dim), dim, backend)
        if (backend == "metal") {
            expect_identical(fit$backend_used, "native_metal_mapped_spectral")
            expect_identical(fit$graph_access, "read_only_mmap")
            expect_identical(fit$cpu_graph_passes, 1L)
            expect_gt(fit$metal_owned_workspace_bytes, 0)
        } else {
            expect_identical(fit$backend_used, "native_cuda_cached_spectral")
            expect_identical(fit$graph_access, "device_cached")
            expect_identical(fit$cpu_graph_passes, 2L)
            expect_gt(fit$cuda_workspace_estimate_bytes, 0)
        }
        expect_identical(dim(layout(fit)), c(n, dim))
        expect_equal(as.vector(dist(layout(fit))),
            as.vector(dist(layout(cpu))), tolerance = 1e-4)
    }
    expect_error(run("insufficient", 2L, backend, budget = 1024),
        "exceeds RAM budget")
    expect_false(file.exists(file.path(work, "insufficient")))
}

test_that("Metal file spectral uses the saved CSR and preserves the layout", {
    skip_if_not(Sys.info()[["sysname"]] == "Darwin")
    skip_if_not(embedding_metal_available_cpp())
    check_accelerated_file_spectral("metal")
})

test_that("CUDA file spectral caches the saved CSR and preserves the layout", {
    skip_if_not(embedding_cuda_available_cpp())
    check_accelerated_file_spectral("cuda")
})

test_that("CUDA spectral admission accounts for graph and basis allocations", {
    testthat::local_mocked_bindings(massive_cuda_memory_cpp = function() {
        list(free_bytes = 15 * 1024^3)
    })
    graph <- list(n_vertices = 1e7, n_edges = 5.15e8)
    choose <- function() full_backed_umap_spectral_backend(
        graph, 2L, "cuda", NULL, 22 * 1024^3)
    expect_identical(choose(), "cuda")
    graph <- list(n_vertices = 5e7, n_edges = 2.6e9)
    expect_identical(choose(), "cpu")
})
