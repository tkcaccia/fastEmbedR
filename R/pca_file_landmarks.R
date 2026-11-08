pca_file_landmark_count <- function(landmarks, rows, rank) {
    if (identical(landmarks, TRUE)) {
        count <- min(rows, max(300, ceiling(7 * sqrt(rows))))
    } else if (is.numeric(landmarks) && length(landmarks) == 1L &&
        is.finite(landmarks) && landmarks > 0) {
        count <- if (landmarks < 1) ceiling(rows * landmarks) else
            landmarks
    } else {
        stop("File-backed PCA landmarks need TRUE, a fraction, or a ",
            "whole-number count; explicit row IDs are not yet supported.",
            call. = FALSE)
    }
    if (!is.finite(count) || count != floor(count) ||
        count > .Machine$integer.max || count <= rank || count > rows) {
        stop("File-backed PCA needs more landmarks than components, ",
            "at most the input row count and at most 2^31 - 1.",
            call. = FALSE)
    }
    as.integer(count)
}

pca_file_landmarks <- function(x, ncomp, xtest, center, scale,
        backend, n.cores, seed, tsne_init, devices, landmarks,
        output, chunk_rows, memory_limit, checkpoint, resume) {
    if (!fastembedr_file_backed_input(x)) stop(
        "File-backed PCA landmarks need a disk-backed massive_matrix().",
        call. = FALSE)
    massive_checkpoint_validate_controls(checkpoint, resume)
    validated <- validate_massive_pca(x, ncomp, xtest, center, scale,
        backend, tsne_init)
    count <- pca_file_landmark_count(landmarks, x$nrow,
        validated$rank)
    if (count == x$nrow) {
        fit <- run_massive_pca(x, ncomp, xtest, center, scale,
            validated$backend, n.cores, seed, tsne_init, output,
            chunk_rows, memory_limit, checkpoint, resume, devices)
        fit$fit_scope <- "all_rows_optimized"
        massive_checkpoint_write(list(version = 1L, fit = fit),
            fit$manifest_path)
        return(fit)
    }
    if (checkpoint || resume) stop(
        "File-backed landmark PCA checkpoint and resume are not yet ",
        "supported; no calculation was started.", call. = FALSE)
    workers <- normalize_pca_threads(n.cores)
    if (validated$backend == "metal" && workers != 1L) stop(
        "Metal file-backed PCA needs `n.cores = 1`.", call. = FALSE)
    massive_validate_devices(devices, validated$backend, x$nrow)
    if (!is.character(output) || length(output) != 1L ||
        is.na(output) || tolower(tools::file_ext(output)) != "f32") {
        stop("File-backed landmark PCA needs one .f32 output path.",
            call. = FALSE)
    }
    path <- file.path(normalizePath(dirname(output), winslash = "/",
        mustWork = TRUE), basename(output))
    stem <- sub("\\.f32$", "", path, ignore.case = TRUE)
    selected_path <- paste0(stem, ".landmarks.f32")
    reference_path <- paste0(stem, ".reference-scores.f32")
    auxiliary <- c(path, paste0(path, ".part"),
        paste0(path, ".manifest.rds"), selected_path,
        reference_path, paste0(reference_path, ".part"),
        paste0(reference_path, ".manifest.rds"))
    if (any(file.exists(auxiliary))) stop(
        "PCA output or landmark intermediate already exists; use a new ",
        "output path.", call. = FALSE)
    init_elements <- if (tsne_init) x$nrow * validated$rank else 0
    test_elements <- if (is.null(xtest)) 0 else
        xtest$nrow * validated$rank
    required <- 4 * (x$nrow * validated$rank +
        count * (x$ncol + validated$rank) + init_elements +
        test_elements)
    if (!is.finite(required) || required + 1024^2 >
        0.8 * massive_disk_available_cpp(path)) stop(
        "PCA landmark outputs exceed the conservative free-disk budget.",
        call. = FALSE)
    total_started <- proc.time()[[3L]]
    selection_started <- proc.time()[[3L]]
    selection <- massive_select_landmarks(x, count, selected_path,
        seed = seed, chunk_rows = chunk_rows,
        memory_limit = memory_limit)
    selection_seconds <- unname(proc.time()[[3L]] - selection_started)
    reference_started <- proc.time()[[3L]]
    fit <- run_massive_pca(selection$data, ncomp, NULL, center,
        scale, validated$backend, n.cores, seed, FALSE,
        reference_path, chunk_rows, memory_limit, FALSE, FALSE,
        devices)
    reference_seconds <- unname(proc.time()[[3L]] - reference_started)
    projection_started <- proc.time()[[3L]]
    projected <- massive_project_pca(fit, x, path,
        backend = validated$backend, n.cores = n.cores,
        chunk_rows = chunk_rows, memory_limit = memory_limit,
        devices = devices, landmarks = selection)
    projection_seconds <- unname(proc.time()[[3L]] - projection_started)
    fit$scores_reference <- fit$scores
    fit$reference_score_identity <- fit$score_identity
    fit$reference_manifest_path <- fit$manifest_path
    fit$scores <- projected$scores
    fit$score_identity <- projected$score_identity
    fit$source_identity <- projected$source_identity
    fit$manifest_path <- paste0(path, ".manifest.rds")
    fit$landmarks <- projected$landmarks
    fit$resources$peak_ram_bytes <- max(
        fit$resources$peak_ram_bytes,
        selection$resources$peak_ram_bytes,
        projected$resources$peak_ram_bytes)
    fit$resources$peak_vram_bytes <- if (
        any(is.finite(c(fit$resources$peak_vram_bytes,
            projected$resources$peak_vram_bytes)))) max(c(
        fit$resources$peak_vram_bytes,
        projected$resources$peak_vram_bytes), na.rm = TRUE) else NA_real_
    fit$fit_scope <- "landmark_fit_plus_projection"
    fit$projection_backend <- projected$backend
    fit$stage_seconds <- list(selection = selection_seconds,
        reference_fit = reference_seconds,
        projection = projection_seconds)
    if (!is.null(xtest)) {
        test_started <- proc.time()[[3L]]
        fit <- massive_pca_project_test(fit, xtest, path,
            normalize_pca_threads(n.cores), chunk_rows,
            memory_limit, devices, FALSE)
        fit$stage_seconds$test_projection <- unname(
            proc.time()[[3L]] - test_started)
    }
    if (tsne_init) {
        init_started <- proc.time()[[3L]]
        fit <- massive_pca_attach_init(fit, TRUE, path)
        fit$stage_seconds$tsne_init <- unname(
            proc.time()[[3L]] - init_started)
    }
    fit$elapsed_sec <- unname(proc.time()[[3L]] - total_started)
    massive_checkpoint_write(list(version = 1L, fit = fit),
        fit$manifest_path)
    fit
}
