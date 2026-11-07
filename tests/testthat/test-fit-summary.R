test_that("fit summaries have a stable bounded-output schema", {
    fields <- c("backend", "backend_engine", "devices",
        "device_identity", "scope", "stage_backend",
        "stage_seconds", "memory_use", "output_rows")
    embedding <- structure(list(layout = matrix(0, 4L, 2L),
        parameters = list(backend = "cpu", landmark = FALSE),
        timings = matrix(c(0.2, 0.3), ncol = 1L,
            dimnames = list(c("knn", "fit"), "elapsed"))),
        class = "fastEmbedR_embedding")
    resident <- summary(embedding)
    expect_identical(names(resident), fields)
    expect_identical(resident$backend, "cpu")
    expect_identical(resident$backend_engine, "cpu")
    expect_identical(resident$scope, "full")
    expect_identical(names(resident$stage_seconds), c("knn", "fit"))
    expect_identical(resident$memory_use$ram_basis, "not_recorded")
    cuda <- embedding
    cuda$parameters$backend <- "cuda"
    cuda$gpu_devices <- c(0L, 1L)
    cuda$gpu_identity <- c("gpu-a", "gpu-b")
    gpu_summary <- summary(cuda)
    expect_identical(gpu_summary$devices, c(0L, 1L))
    expect_identical(gpu_summary$device_identity,
        c("gpu-a", "gpu-b"))
    backed <- structure(list(method = "umap", backend = "metal",
        nrow = 100000000, ncol = 2L,
        pipeline = list(fit_scope = "all_rows_optimized",
            stage_seconds = c(knn = 1, graph = 2,
                init = 3, optimize = 4)),
        resources = list(peak_ram_bytes = 1024)),
        class = "fastEmbedR_massive_matrix")
    file_summary <- summary(backed)
    expect_identical(names(file_summary), fields)
    expect_identical(file_summary$memory_use$ram_basis, "estimate")
    expect_identical(file_summary$memory_use$component_estimates_bytes,
        list())
    expect_match(file_summary$output_rows, "massive_read_rows")
    expect_null(file_summary$layout)
    input <- structure(list(nrow = 100000000, ncol = 2L),
        class = "fastEmbedR_massive_matrix")
    expect_error(summary(input), "input data, not a fit")
    detailed <- backed
    detailed$backend <- "native_metal_fft_graph_mmap"
    detailed$resources <- list(layout_ram_bytes = 1600,
        graph_bytes = 3200, delta_ram_bytes = 800)
    detailed$pipeline$stage_backend <- c(knn = "metal", graph = "cpu",
        init = "metal", optimize = "metal")
    info <- summary(detailed)
    expect_identical(info$backend, "metal")
    expect_identical(info$backend_engine,
        "native_metal_fft_graph_mmap")
    expect_identical(info$stage_backend[["graph"]], "cpu")
    expect_true(is.na(info$memory_use$ram_bytes))
    expect_identical(info$memory_use$component_estimates_bytes,
        list(layout_ram_bytes = 1600, graph_bytes = 3200,
            delta_ram_bytes = 800))
    pca_fit <- structure(list(resources = list(peak_ram_bytes = 512)),
        class = c("fastEmbedR_massive_pca", "fastEmbedR_pca"))
    expect_identical(summary(pca_fit)$output_rows,
        "massive_read_rows(fit$scores, first, n)")
})

test_that("file-backed PCA summary reads bounded score rows", {
    skip_if(.Platform$OS.type != "unix")
    x <- as.matrix(iris[, 1:4])
    work <- tempfile("summary-pca-")
    dir.create(work)
    on.exit(unlink(work, recursive = TRUE), add = TRUE)
    input_path <- file.path(work, "input.f32")
    writeBin(as.numeric(t(x)), input_path, size = 4L,
        endian = "little")
    input <- massive_matrix(input_path, nrow(x), ncol(x))
    fit <- pca(input, ncomp = 2L, backend = "cpu", n.cores = 1L,
        massive = "out_of_core", output = file.path(work, "scores.f32"),
        memory_limit = "512MB")
    info <- summary(fit)
    expect_identical(info$backend, "cpu")
    expect_identical(info$scope, "full")
    expect_identical(info$output_rows,
        "massive_read_rows(fit$scores, first, n)")
    rows <- massive_read_rows(fit$scores, 1L, 5L)
    expect_identical(dim(rows), c(5L, 2L))
    expect_true(all(is.finite(rows)))
})
