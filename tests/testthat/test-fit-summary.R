test_that("fit summaries have a stable bounded-output schema", {
    fields <- c("backend", "devices", "device_identity", "scope",
        "stage_seconds", "memory_use", "output_rows")
    embedding <- structure(list(layout = matrix(0, 4L, 2L),
        parameters = list(backend = "cpu", landmark = FALSE),
        timings = matrix(c(0.2, 0.3), ncol = 1L,
            dimnames = list(c("knn", "fit"), "elapsed"))),
        class = "fastEmbedR_embedding")
    resident <- summary(embedding)
    expect_identical(names(resident), fields)
    expect_identical(resident$backend, "cpu")
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
    expect_match(file_summary$output_rows, "massive_read_rows")
    expect_null(file_summary$layout)
    input <- structure(list(nrow = 100000000, ncol = 2L),
        class = "fastEmbedR_massive_matrix")
    expect_error(summary(input), "input data, not a fit")
})
