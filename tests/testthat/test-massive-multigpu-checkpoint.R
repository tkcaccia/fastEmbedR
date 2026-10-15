test_that("coordinated UMAP checkpoints bind source and device mapping", {
    root <- tempfile("umap_gpu_checkpoint_")
    dir.create(root)
    on.exit(unlink(root, recursive = TRUE), add = TRUE)
    source <- file.path(root, "source.bin")
    init <- file.path(root, "init.f32")
    writeBin(as.raw(1:8), source)
    writeBin(rep(0, 8), init, size = 4L)
    plan <- list(output = file.path(root, "layout.f32"),
        source_paths = c(source, init), init_format = "f32",
        layout_bytes = 32, n_epochs = 4L, backend = "cuda",
        parameters = list(devices = c(0L, 1L),
            gpu_identity = c("GPU-A", "GPU-B"),
            layout_storage = "memory"))
    identity <- fastEmbedR:::massive_checkpoint_input_files_identity
    plan$source_identity <- identity(plan$source_paths)
    prepare <- fastEmbedR:::massive_umap_checkpoint_prepare
    saved <- prepare(plan, TRUE, FALSE, 1L)
    snapshot <- paste0(plan$output, ".epoch_1.f32")
    writeBin(rep(1, 8), snapshot, size = 4L)
    saved$progress(1L, snapshot, 4, 2, 8)
    expect_equal(prepare(plan, TRUE, TRUE, 1L)$epoch, 1L)
    changed <- plan
    changed$parameters$gpu_identity[[2L]] <- "GPU-C"
    expect_error(prepare(changed, TRUE, TRUE, 1L), "does not match")
    changed <- plan
    changed$parameters$devices <- c(1L, 0L)
    expect_error(prepare(changed, TRUE, TRUE, 1L), "does not match")
    con <- file(source, "ab")
    writeBin(as.raw(9), con)
    close(con)
    changed <- plan
    changed$source_identity <- identity(changed$source_paths)
    expect_error(prepare(changed, TRUE, TRUE, 1L), "does not match")
})

test_that("coordinated t-SNE checkpoints bind source and device mapping", {
    root <- tempfile("tsne_gpu_checkpoint_")
    dir.create(root)
    on.exit(unlink(root, recursive = TRUE), add = TRUE)
    source <- file.path(root, "source.bin")
    init <- file.path(root, "init.f32")
    writeBin(as.raw(1:8), source)
    writeBin(rep(0, 8), init, size = 4L)
    plan <- list(output = file.path(root, "layout.f32"),
        source_paths = c(source, init), state_bytes = 96,
        n_vertices = 4L, dimensions = 2L,
        early_iter = 1L, normal_iter = 3L,
        parameters = list(devices = c(0L, 1L),
            gpu_identity = c("GPU-A", "GPU-B")))
    identity <- fastEmbedR:::massive_checkpoint_input_files_identity
    plan$source_identity <- identity(plan$source_paths)
    prepare <- fastEmbedR:::massive_tsne_checkpoint_prepare
    saved <- prepare(plan, TRUE, FALSE, 1L)
    snapshot <- paste0(plan$output, ".iter_1.state.f32")
    writeBin(rep(1, 24), snapshot, size = 4L)
    saved$progress(1L, snapshot, 0.1)
    expect_equal(prepare(plan, TRUE, TRUE, 1L)$iteration, 1L)
    changed <- plan
    changed$parameters$gpu_identity[[2L]] <- "GPU-C"
    expect_error(prepare(changed, TRUE, TRUE, 1L), "does not match")
    changed <- plan
    changed$parameters$devices <- c(1L, 0L)
    expect_error(prepare(changed, TRUE, TRUE, 1L), "does not match")
    con <- file(source, "ab")
    writeBin(as.raw(9), con)
    close(con)
    changed <- plan
    changed$source_identity <- identity(changed$source_paths)
    expect_error(prepare(changed, TRUE, TRUE, 1L), "does not match")
})
