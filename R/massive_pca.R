massive_memory_bytes <- function(value) {
    if (is.numeric(value) && length(value) == 1L &&
        is.finite(value) && value > 0) return(as.numeric(value))
    if (!is.character(value) || length(value) != 1L || is.na(value) ||
        !grepl("^[0-9]+(\\.[0-9]+)?(KB|MB|GB)$", value)) {
        stop("`memory_limit` must be positive bytes or KB/MB/GB.",
            call. = FALSE)
    }
    unit <- sub("^.*(KB|MB|GB)$", "\\1", value)
    amount <- as.numeric(sub("(KB|MB|GB)$", "", value))
    if (amount <= 0) {
        stop("`memory_limit` must be positive.", call. = FALSE)
    }
    amount * switch(unit, KB = 1024, MB = 1024^2, GB = 1024^3)
}

massive_linux_available_ram <- function(meminfo = "/proc/meminfo",
        v2 = "/sys/fs/cgroup", v1 = "/sys/fs/cgroup/memory") {
    lines <- tryCatch(readLines(meminfo, warn = FALSE),
        error = function(e) character())
    found <- grep("^MemAvailable:[[:space:]]+[0-9]+", lines,
        value = TRUE)
    if (length(found) != 1L) return(NA_real_)
    available <- as.numeric(sub(
        "^MemAvailable:[[:space:]]+([0-9]+).*", "\\1", found)) * 1024
    read_number <- function(path) {
        if (!file.exists(path)) return(NA_real_)
        value <- tryCatch(readLines(path, n = 1L, warn = FALSE),
            error = function(e) character())
        if (length(value) != 1L) return(NA_real_)
        suppressWarnings(as.numeric(value))
    }
    limit <- read_number(file.path(v2, "memory.max"))
    used <- read_number(file.path(v2, "memory.current"))
    if (!is.finite(limit) || !is.finite(used)) {
        limit <- read_number(file.path(v1, "memory.limit_in_bytes"))
        used <- read_number(file.path(v1, "memory.usage_in_bytes"))
    }
    if (is.finite(limit) && is.finite(used)) {
        available <- min(available, max(0, limit - used))
    }
    available
}

massive_macos_available_ram <- function(report = NULL) {
    if (is.null(report)) report <- tryCatch(suppressWarnings(
        system2("memory_pressure", "-Q", stdout = TRUE,
            stderr = FALSE)), error = function(e) character())
    total <- grep("^The system has [0-9]+", report, value = TRUE)
    free <- grep("^System-wide memory free percentage: [0-9]+%",
        report, value = TRUE)
    if (length(total) != 1L || length(free) != 1L) return(NA_real_)
    bytes <- as.numeric(sub("^The system has ([0-9]+).*", "\\1", total))
    fraction <- as.numeric(sub("^.*: ([0-9]+)%.*", "\\1", free)) / 100
    bytes * fraction
}

massive_windows_available_ram <- function(report = NULL) {
    if (is.null(report)) report <- tryCatch(suppressWarnings(
        system2("powershell.exe", c("-NoProfile", "-NonInteractive",
            "-Command", shQuote(paste0(
                "(Get-CimInstance Win32_OperatingSystem).",
                "FreePhysicalMemory"))), stdout = TRUE, stderr = FALSE)),
        error = function(e) character())
    number <- grep("^[[:space:]]*[0-9]+[[:space:]]*$", report,
        value = TRUE)
    if (length(number) != 1L) return(NA_real_)
    as.numeric(trimws(number)) * 1024
}

massive_available_ram_bytes <- function() {
    system <- Sys.info()[["sysname"]]
    if (identical(system, "Linux")) return(massive_linux_available_ram())
    if (identical(system, "Darwin")) return(massive_macos_available_ram())
    if (identical(system, "Windows")) return(massive_windows_available_ram())
    NA_real_
}

massive_auto_memory_limit <- function(memory_limit,
        available = massive_available_ram_bytes()) {
    if (length(available) != 1L || !is.finite(available) ||
        available <= 0) stop(
        "Auto mode cannot determine available RAM; use an explicit ",
        "massive mode and memory_limit.", call. = FALSE)
    min(massive_memory_bytes(memory_limit), 0.8 * available)
}

massive_pca_auto_gpu_ok <- function(rows, cols, backend) {
    if (!identical(backend, "cuda")) return(TRUE)
    if (!isTRUE(embedding_cuda_available_cpp())) {
        stop("EXPERIMENTAL CUDA auto PCA is unavailable; ",
            "no CPU fallback was used.", call. = FALSE)
    }
    needed <- 128 * 1024^2 + 16 * rows * cols + 8 * cols^2
    needed <= 0.5 * massive_cuda_memory_cpp()$free_bytes
}

massive_pca_auto_route <- function(x, memory_limit, output,
                                    backend = NULL, devices = NULL) {
    limit <- massive_auto_memory_limit(memory_limit)
    source <- inherits(x, "fastEmbedR_massive_matrix")
    if (!source && !is.matrix(x)) {
        stop("EXPERIMENTAL auto PCA requires a matrix or ",
            "massive_matrix() source.", call. = FALSE)
    }
    rows <- if (source) x$nrow else nrow(x)
    cols <- if (source) x$ncol else ncol(x)
    if (!is.null(devices)) {
        devices <- massive_validate_devices(devices, backend, rows)
        previous <- massive_cuda_memory_cpp()$device
        on.exit(massive_cuda_select_cpp(previous))
        massive_cuda_select_cpp(devices[[1L]])
    }
    elements <- rows * cols
    estimate <- 128 * 1024^2 + 32 * elements + 8 * cols^2
    resident <- !source || x$format == "memory"
    materializable <- !source ||
        (rows <= .Machine$integer.max &&
            cols <= .Machine$integer.max)
    if (is.finite(estimate) && estimate <= 0.5 * limit &&
        materializable &&
        massive_pca_auto_gpu_ok(rows, cols, backend)) {
        if (!is.null(output)) {
            stop("Auto PCA fits in memory; remove `output` or use ",
                "`massive = \"out_of_core\"`.", call. = FALSE)
        }
        message("EXPERIMENTAL auto PCA selected in-memory PCA; ",
            "estimated peak=", round(estimate / 1024^2), " MB",
            if (source) "; materializing source" else "")
        return(list(mode = "off", data = if (source) {
            massive_auto_materialize(x)
        } else x, estimate = estimate))
    }
    if (resident) {
        stop("Auto PCA exceeds its safe in-memory budget; supply a ",
            "file-backed massive_matrix() source.", call. = FALSE)
    }
    if (is.null(output)) {
        stop("Auto PCA selected out-of-core processing; provide ",
            "a persistent `.f32` `output` path.", call. = FALSE)
    }
    message("EXPERIMENTAL auto PCA selected out-of-core PCA; ",
        "estimated in-memory peak=", round(estimate / 1024^2), " MB")
    list(mode = "out_of_core", data = x, estimate = estimate)
}

run_massive_auto_pca <- function(x, ncomp, xtest, center, scale,
        backend, n.cores, seed, tsne_init, output, chunk_rows,
        memory_limit, checkpoint, resume, devices) {
    backend <- validate_pca_backend(resolve_embedding_backend(backend))
    route <- massive_pca_auto_route(x, memory_limit, output,
        backend, devices)
    if (route$mode == "off" && !is.null(devices)) stop(
        "Auto PCA selected in-memory processing; `devices` requires ",
        "out-of-core PCA.", call. = FALSE)
    if (route$mode == "off" && !is.null(chunk_rows)) {
        stop("Auto PCA selected in-memory processing; `chunk_rows` ",
            "is not used.", call. = FALSE)
    }
    fit <- pca(route$data, ncomp, xtest, center, scale, backend,
        n.cores, seed, tsne_init, massive = route$mode,
        output = output, chunk_rows = chunk_rows,
        memory_limit = memory_limit, checkpoint = checkpoint,
        resume = resume, devices = devices)
    fit$massive_auto <- if (route$mode == "off") {
        "in_memory"
    } else "out_of_core"
    fit$massive_auto_estimated_ram_bytes <- route$estimate
    fit
}

massive_pca_covariance_route <- function(p, backend) {
    p <= 1024 || (backend == "cuda" && p <= 2048)
}

massive_pca_fixed_bytes <- function(source, rank, n.cores, backend) {
    p <- source$ncol
    source_bytes <- if (source$format == "memory") source$nrow * p * 8 else 0
    if (!massive_pca_covariance_route(p, backend)) {
        width <- min(p, rank + 16)
        source_bytes + 64 * 1024^2 +
            (n.cores + 8) * p * width * 8 + n.cores * p * 8
    } else {
        source_bytes + (n.cores + 5) * p * p * 8 + 8 * p * 8
    }
}

massive_pca_gpu_fixed_bytes <- function(p, rank) {
    width <- if (!massive_pca_covariance_route(p, "cuda")) {
        min(p, rank + 16)
    } else rank
    if (!massive_pca_covariance_route(p, "cuda")) {
        return((2 * p * width + 3 * p) * 4 + 128 * 1024^2)
    }
    (p * p + p * rank + 3 * p) * 4 + 128 * 1024^2
}

massive_pca_resources <- function(source, rank, n.cores,
    chunk_rows, memory_limit, backend) {
    p <- source$ncol
    limit <- massive_memory_bytes(memory_limit)
    source_bytes <- if (source$format == "memory") source$nrow * p * 8 else 0
    fixed <- massive_pca_fixed_bytes(source, rank, n.cores, backend)
    per_row <- 4 * (p + rank)
    maximum <- floor((0.7 * limit - fixed) / per_row)
    if (!is.finite(maximum) || maximum < 1) {
        stop("PCA buffers exceed `memory_limit`; reduce `n.cores` ",
            "or use a larger limit.", call. = FALSE)
    }
    requested <- chunk_rows %||% 250000L
    if (length(requested) != 1L || !is.finite(requested) ||
        requested < 1 || requested != floor(requested) ||
        requested > .Machine$integer.max) {
        stop("`chunk_rows` must be one positive integer.",
            call. = FALSE)
    }
    rows <- as.integer(min(requested, maximum, source$nrow))
    gpu <- NULL
    if (backend == "cuda") {
        gpu <- massive_cuda_memory_cpp()
        width <- if (massive_pca_covariance_route(p, backend)) rank else
            min(p, rank + 16)
        gpu_fixed <- massive_pca_gpu_fixed_bytes(p, rank)
        gpu_per_row <- 4 * (p + width)
        gpu_max <- floor((0.65 * gpu$free_bytes - gpu_fixed) /
            gpu_per_row)
        if (!is.finite(gpu_max) || gpu_max < 1) {
            stop("CUDA PCA buffers exceed the free-VRAM budget.",
                call. = FALSE)
        }
        rows <- as.integer(min(rows, gpu_max))
    }
    result <- list(
        chunk_rows = rows, peak_ram_bytes = fixed + per_row * rows,
        output_bytes = source$nrow * rank * 4,
        input_buffer_bytes = rows * p * 4,
        input_resident_bytes = source_bytes,
        memory_limit_bytes = limit
    )
    if (!is.null(gpu)) {
        result$peak_vram_bytes <- gpu_fixed + gpu_per_row * rows
        result$free_vram_bytes <- gpu$free_bytes
        result$gpu_device <- gpu$device
    }
    result
}

massive_float_output <- function(output, bytes, resume = FALSE) {
    if (!is.character(output) || length(output) != 1L ||
        is.na(output) || !nzchar(output)) {
        stop("`output` must be one .f32 path.", call. = FALSE)
    }
    if (tolower(tools::file_ext(output)) != "f32") {
        stop("Output must have a .f32 extension.", call. = FALSE)
    }
    parent <- normalizePath(dirname(output), mustWork = TRUE)
    path <- file.path(parent, basename(output))
    part <- paste0(path, ".part")
    if (file.exists(path) || (file.exists(part) && !resume)) {
        stop("Output or its .part file already exists.",
            call. = FALSE)
    }
    if (resume && !file.exists(part)) {
        stop("No .part file exists to resume.", call. = FALSE)
    }
    remaining <- bytes - if (resume) file.info(part)$size else 0
    if (!is.finite(remaining) || remaining < 0 ||
        remaining > 0.9 * massive_disk_available_cpp(path)) {
        stop("Output exceeds the conservative free-disk budget.",
            call. = FALSE)
    }
    path
}

massive_pca_output <- function(output, score_bytes, columns, rank,
        resume) {
    model_bytes <- 8 * columns * (rank + 2) + 1024^2
    path <- massive_float_output(output, score_bytes + model_bytes,
        resume = resume && file.exists(paste0(output, ".part")))
    if (file.exists(paste0(path, ".manifest.rds"))) stop(
        "PCA completion manifest already exists.", call. = FALSE)
    path
}

massive_pca_decomposition <- function(source, rank, center, scale,
        resources, n.cores, backend, seed, saved_stats = NULL) {
    if (!massive_pca_covariance_route(source$ncol, backend)) {
        return(massive_pca_wide_decomposition(source, rank, center,
            scale, resources, n.cores, seed, backend, saved_stats))
    }
    covariance_fn <- if (backend == "cuda") {
        massive_pca_covariance_cuda_cpp
    } else massive_pca_covariance_cpp
    stats <- if (is.list(saved_stats)) saved_stats else covariance_fn(
        source, resources$chunk_rows, n.cores, center, saved_stats)
    deviations <- if (scale) sqrt(diag(stats$covariance)) else {
        rep(1, source$ncol)
    }
    if (any(!is.finite(deviations)) || any(deviations <= 0)) {
        stop("PCA scaling requires positive variance in every column.",
            call. = FALSE)
    }
    covariance <- stats$covariance / outer(deviations, deviations)
    eig <- eigen(covariance, symmetric = TRUE)
    list(
        center = stats$center, scale = deviations,
        loadings = eig$vectors[, seq_len(rank), drop = FALSE],
        singular_values = sqrt(pmax(eig$values[seq_len(rank)], 0) *
            (source$nrow - 1))
    )
}

massive_pca_wide_decomposition <- function(source, rank, center,
        scale, resources, n.cores, seed, backend, saved_moments = NULL) {
    width <- as.integer(min(source$ncol, rank + 16))
    moments <- saved_moments %||% massive_pca_wide_moments_cpp(
        source, resources$chunk_rows, n.cores, center, scale)
    seed <- integer_scalar(seed)
    if (is.na(seed)) stop("`seed` must be an integer.", call. = FALSE)
    restore <- set_local_seed(seed)
    on.exit(restore())
    omega <- matrix(stats::rnorm(source$ncol * width),
        nrow = source$ncol)
    action_fn <- if (backend == "cuda") {
        massive_pca_wide_action_cuda_cpp
    } else massive_pca_wide_action_cpp
    args <- list(source, moments$center, moments$scale,
        omega, resources$chunk_rows)
    if (backend == "cpu") args <- c(args, list(n.cores))
    action <- do.call(action_fn, args)
    factor <- qr(action)
    if (factor$rank < rank) {
        stop("Wide PCA sketch is rank deficient.", call. = FALSE)
    }
    basis <- qr.Q(factor)
    args[[4L]] <- basis
    projected <- do.call(action_fn, args)
    small <- crossprod(basis, projected) / (source$nrow - 1)
    eig <- eigen((small + t(small)) / 2, symmetric = TRUE)
    list(center = moments$center, scale = moments$scale,
        loadings = basis %*% eig$vectors[, seq_len(rank), drop = FALSE],
        singular_values = sqrt(pmax(eig$values[seq_len(rank)], 0) *
            (source$nrow - 1)), sketch_size = width,
        moments_backend = "cpu", sketch_backend = backend)
}

validate_massive_pca <- function(x, ncomp, xtest, center, scale,
                                    backend, tsne_init) {
    if (!inherits(x, "fastEmbedR_massive_matrix")) {
        stop("EXPERIMENTAL out-of-core PCA requires massive_matrix(x).",
            call. = FALSE)
    }
    backend <- resolve_embedding_backend(backend)
    if (backend == "metal") {
        stop("EXPERIMENTAL out-of-core PCA has no Metal route; ",
            "no CPU fallback was used.", call. = FALSE)
    }
    if (backend == "cuda" && !isTRUE(
            embedding_cuda_available_cpp())) {
        stop("EXPERIMENTAL out-of-core CUDA PCA requires native CUDA; ",
            "no CPU fallback was used.", call. = FALSE)
    }
    if (!identical(tsne_init, FALSE) && !identical(tsne_init, TRUE)) {
        stop("`tsne_init` must be TRUE or FALSE.", call. = FALSE)
    }
    if (!is.null(xtest) || tsne_init) {
        stop("Out-of-core `xtest` and `tsne_init` are not implemented.",
            call. = FALSE)
    }
    if (!is.logical(center) || !is.logical(scale) ||
        length(center) != 1L || length(scale) != 1L ||
        is.na(center) || is.na(scale)) {
        stop("`center` and `scale` must be TRUE or FALSE.",
            call. = FALSE)
    }
    rank <- integer_scalar(ncomp)
    if (is.na(rank) || rank < 1L || rank > x$ncol ||
        rank >= x$nrow || x$ncol > .Machine$integer.max) {
        stop("Streamed PCA needs rank < nrow and ",
            "rank <= ncol <= 2^31 - 1.", call. = FALSE)
    }
    list(rank = rank, backend = backend)
}

finish_massive_pca <- function(fit, path, x, rank, workers, seed,
                                resources, started, backend) {
    fit$scores <- massive_matrix(
        path, nrow = x$nrow, ncol = rank, format = "f32"
    )
    fit$ncomp <- rank
    fit$method <- if (massive_pca_covariance_route(x$ncol, backend)) {
        "streamed_covariance"
    } else "streamed_randomized"
    fit$backend <- backend
    fit$n_threads <- workers
    fit$seed <- seed
    fit$resources <- resources
    if (!is.null(resources$gpu_devices)) {
        fit$gpu_devices <- resources$gpu_devices
    }
    fit$experimental <- TRUE
    fit$elapsed_sec <- unname(proc.time()[[3L]] - started)
    fit$score_identity <- massive_checkpoint_file_identity(path)
    fit$manifest_path <- paste0(path, ".manifest.rds")
    class(fit) <- c("fastEmbedR_massive_pca", "fastEmbedR_pca")
    massive_checkpoint_write(list(version = 1L, fit = fit),
        fit$manifest_path)
    fit
}

#' EXPERIMENTAL reopen of completed file-backed PCA
#'
#' Reuses a saved streamed PCA fit and float32 scores without reloading the
#' input matrix. The saved score file must still match its completion record.
#'
#' @param output The `.f32` path supplied to `pca()` in out-of-core mode.
#' @return The lightweight PCA fit with file-backed `scores`.
#' @export
massive_open_pca <- function(output) {
    if (!is.character(output) || length(output) != 1L ||
        is.na(output) || tolower(tools::file_ext(output)) != "f32") {
        stop("`output` must be one .f32 PCA path.", call. = FALSE)
    }
    path <- file.path(normalizePath(dirname(output), winslash = "/",
        mustWork = TRUE),
        basename(output))
    manifest <- paste0(path, ".manifest.rds")
    if (!file.exists(manifest)) stop(
        "PCA completion manifest is missing.", call. = FALSE)
    saved <- tryCatch(suppressWarnings(readRDS(manifest)),
        error = function(e) NULL)
    if (!is.list(saved)) stop("PCA completion manifest is invalid.",
        call. = FALSE)
    fit <- saved$fit
    valid <- is.list(saved) && identical(saved$version, 1L) &&
        inherits(fit, "fastEmbedR_massive_pca") &&
        inherits(fit$scores, "fastEmbedR_massive_matrix") &&
        massive_checkpoint_same_paths(fit$scores$path, path) &&
        identical(fit$scores$format, "f32") &&
        massive_checkpoint_same_paths(fit$manifest_path, manifest) &&
        isTRUE(fit$experimental) &&
        isTRUE(fit$backend %in% c("cpu", "cuda")) &&
        is.numeric(fit$ncomp) && length(fit$ncomp) == 1L &&
        is.finite(fit$ncomp) && fit$ncomp >= 1 &&
        identical(as.numeric(fit$scores$ncol),
            as.numeric(fit$ncomp)) &&
        is.matrix(fit$loadings) &&
        ncol(fit$loadings) == fit$ncomp &&
        nrow(fit$loadings) == length(fit$center) &&
        length(fit$scale) == nrow(fit$loadings) &&
        isTRUE(all(is.finite(fit$loadings))) &&
        isTRUE(all(is.finite(fit$center))) &&
        isTRUE(all(is.finite(fit$scale)))
    if (!isTRUE(valid)) stop("PCA completion manifest is invalid.",
        call. = FALSE)
    identity <- tryCatch(massive_checkpoint_file_identity(path),
        error = function(e) NULL)
    if (!identical(identity, fit$score_identity) ||
        !identical(identity$bytes,
            as.numeric(4 * fit$scores$nrow * fit$ncomp))) stop(
        "PCA score file changed since completion.", call. = FALSE)
    fit
}

massive_pca_checkpoint_signature <- function(x, path, rank, center,
        scale, backend, workers, seed, resources) {
    massive_checkpoint_signature(list(version = 5L,
        source = massive_checkpoint_source_identity(x),
        output = path, rank = rank, center = center, scale = scale,
        backend = backend, workers = workers, seed = seed,
        covariance = massive_pca_covariance_route(x$ncol, backend),
        chunk_rows = resources$chunk_rows,
        gpu_devices = resources$gpu_devices %||% NULL))
}

massive_pca_checkpoint_stats_valid <- function(state, x, resources,
        workers, backend) {
    if (state$stage == "pending") return(TRUE)
    if (state$stage == "mean_partial") {
        rows <- state$mean_rows
        sums <- state$mean_sums
        return(massive_pca_covariance_route(x$ncol, backend) &&
            is.numeric(rows) &&
            length(rows) == 1L && is.finite(rows) &&
            rows == floor(rows) && rows >= 0 && rows <= x$nrow &&
            (rows == x$nrow || rows %% resources$chunk_rows == 0) &&
            is.matrix(sums) && is.numeric(sums) &&
            nrow(sums) == min(workers, resources$chunk_rows) &&
            ncol(sums) == x$ncol &&
            all(is.finite(sums)) && identical(state$mean_hash,
                massive_checkpoint_signature(list(rows = rows,
                    sums = sums))))
    }
    if (state$stage %in% c("means", "covariance_partial")) {
        valid <- massive_pca_covariance_route(x$ncol, backend) &&
        is.numeric(state$means) &&
        length(state$means) == x$ncol &&
        all(is.finite(state$means)) &&
        identical(state$means_hash,
            massive_checkpoint_signature(state$means))
        if (state$stage == "means") return(valid)
        return(valid && massive_pca_cross_state_valid(state, x,
            resources, workers, backend))
    }
    if (state$stage == "moments") return(
        !massive_pca_covariance_route(x$ncol, backend) &&
        is.list(state$moments) &&
        is.numeric(state$moments$center) &&
        is.numeric(state$moments$scale) &&
        length(state$moments$center) == x$ncol &&
        length(state$moments$scale) == x$ncol &&
        all(is.finite(c(state$moments$center,
            state$moments$scale))) &&
        all(state$moments$scale > 0) &&
        identical(state$moments_hash,
            massive_checkpoint_signature(state$moments)))
    FALSE
}

massive_pca_cross_state_valid <- function(state, x, resources,
        workers, backend) {
    rows <- state$cross_rows
    sums <- state$cross_sums
    columns <- if (backend == "cuda") 1L else
        min(workers, resources$chunk_rows)
    is.numeric(rows) && length(rows) == 1L && is.finite(rows) &&
        rows == floor(rows) && rows >= 0 && rows <= x$nrow &&
        (rows == x$nrow || rows %% resources$chunk_rows == 0) &&
        is.matrix(sums) && is.numeric(sums) &&
        nrow(sums) == x$ncol^2 && ncol(sums) == columns &&
        all(is.finite(sums)) && identical(state$cross_hash,
            massive_checkpoint_signature(list(rows = rows,
                sums = sums)))
}

massive_pca_checkpoint_load <- function(sidecar, signature, x,
        rank, resources, part, workers, backend) {
    state <- tryCatch(readRDS(sidecar), error = function(e) NULL)
    if (!is.list(state) || !identical(state$signature, signature) ||
        !is.character(state$stage) || length(state$stage) != 1L ||
        !state$stage %in% c("pending", "mean_partial", "means",
            "covariance_partial", "moments", "fit")) {
        stop("PCA checkpoint does not match the current inputs.",
            call. = FALSE)
    }
    rows <- state$completed_rows
    if (!is.numeric(rows) || length(rows) != 1L ||
        !is.finite(rows) || rows != floor(rows) ||
        rows < 0 || rows > x$nrow) stop(
        "PCA checkpoint does not match the current inputs.",
        call. = FALSE)
    if (state$stage != "fit") {
        valid <- rows == 0 && !file.exists(part) &&
            massive_pca_checkpoint_stats_valid(state, x, resources,
                workers, backend)
        if (!valid) stop("PCA statistics checkpoint is invalid.", call. = FALSE)
        return(state)
    }
    if (!is.list(state$fit) || !identical(state$fit_hash,
        massive_checkpoint_signature(state$fit))) stop(
        "PCA checkpoint does not match the current inputs.",
        call. = FALSE)
    size <- if (file.exists(part)) file.info(part)$size else 0
    if ((rows != x$nrow && rows %% resources$chunk_rows != 0) ||
        is.na(size) || size < rows * rank * 4 ||
        size > resources$output_bytes) {
        stop("PCA partial scores do not match the checkpoint.",
            call. = FALSE)
    }
    state
}

massive_pca_checkpoint_means <- function(x, center, workers,
        resources, state, sidecar) {
    if (state$stage == "pending" && center) {
        state$mean_rows <- 0
        state$mean_sums <- matrix(0,
            min(workers, resources$chunk_rows), x$ncol)
        state$mean_hash <- massive_checkpoint_signature(list(
            rows = state$mean_rows, sums = state$mean_sums))
        state$stage <- "mean_partial"
        massive_checkpoint_write(state, sidecar)
    }
    while (state$stage == "mean_partial" &&
        state$mean_rows < x$nrow) {
        rows <- as.integer(min(resources$chunk_rows,
            x$nrow - state$mean_rows))
        sums <- massive_pca_mean_chunk_cpp(x,
            state$mean_rows + 1, rows, nrow(state$mean_sums),
            state$mean_sums)
        state$mean_sums <- sums
        state$mean_rows <- state$mean_rows + rows
        state$mean_hash <- massive_checkpoint_signature(list(
            rows = state$mean_rows, sums = state$mean_sums))
        massive_checkpoint_write(state, sidecar)
    }
    state$means <- numeric(x$ncol)
    if (center) for (worker in seq_len(nrow(state$mean_sums))) {
        state$means <- state$means +
            state$mean_sums[worker, ] / x$nrow
    }
    state$means_hash <- massive_checkpoint_signature(state$means)
    state$mean_sums <- NULL
    state$mean_rows <- NULL
    state$mean_hash <- NULL
    state$stage <- "means"
    massive_checkpoint_write(state, sidecar)
    state
}

massive_pca_checkpoint_covariance <- function(x, center, backend,
        workers, resources, state, sidecar) {
    if (state$stage == "means") {
        columns <- if (backend == "cuda") 1L else
            min(workers, resources$chunk_rows)
        state$cross_rows <- 0
        state$cross_sums <- matrix(0, x$ncol^2, columns)
        state$cross_hash <- massive_checkpoint_signature(list(
            rows = state$cross_rows, sums = state$cross_sums))
        state$stage <- "covariance_partial"
        massive_checkpoint_write(state, sidecar)
    }
    state_bytes <- 8 * length(state$cross_sums)
    block_bytes <- 4 * x$ncol * resources$chunk_rows
    every <- if (state_bytes <= 1024^2) 16L else
        as.integer(min(256, max(16,
            ceiling(100 * state_bytes / block_bytes))))
    progress <- function(rows, sums) {
        current <- state
        current$cross_rows <- rows
        current$cross_sums <- sums
        current$cross_hash <- massive_checkpoint_signature(list(
            rows = rows, sums = sums))
        massive_checkpoint_write(current, sidecar)
    }
    covariance_fn <- if (backend == "cuda") {
        massive_pca_covariance_cuda_cpp
    } else massive_pca_covariance_cpp
    stats <- covariance_fn(x, resources$chunk_rows, workers,
        center, state$means, state$cross_rows, state$cross_sums,
        progress, every)
    list(state = state, stats = stats)
}

massive_pca_checkpoint_fit <- function(x, rank, center, scale,
        backend, workers, seed, resources, state, sidecar) {
    if (state$stage == "fit") return(state)
    if (state$stage %in% c("pending", "mean_partial") &&
        massive_pca_covariance_route(x$ncol, backend)) {
        state <- massive_pca_checkpoint_means(
            x, center, workers, resources, state, sidecar)
    }
    if (state$stage == "pending" &&
        !massive_pca_covariance_route(x$ncol, backend)) {
        state$moments <- massive_pca_wide_moments_cpp(x,
            resources$chunk_rows, workers, center, scale)
        state$moments_hash <- massive_checkpoint_signature(state$moments)
        state$stage <- "moments"
        massive_checkpoint_write(state, sidecar)
    }
    saved <- state$moments %||% NULL
    if (massive_pca_covariance_route(x$ncol, backend)) {
        covariance <- massive_pca_checkpoint_covariance(x, center,
            backend, workers, resources, state, sidecar)
        state <- covariance$state
        saved <- covariance$stats
    }
    fit <- massive_pca_decomposition(x, rank, center, scale,
        resources, workers, backend, seed, saved)
    state$fit <- fit
    state$fit_hash <- massive_checkpoint_signature(fit)
    state$means <- NULL
    state$means_hash <- NULL
    state$cross_rows <- NULL
    state$cross_sums <- NULL
    state$cross_hash <- NULL
    state$moments <- NULL
    state$moments_hash <- NULL
    state$stage <- "fit"
    massive_checkpoint_write(state, sidecar)
    state
}

massive_pca_prepare <- function(x, path, rank, center, scale,
        backend, workers, seed, resources, checkpoint, resume) {
    if (!checkpoint) {
        fit <- massive_pca_decomposition(x, rank, center, scale,
            resources, workers, backend, seed)
        return(list(fit = fit, rows = 0, progress = NULL,
            resume_part = FALSE, sidecar = NULL))
    }
    if (is.null(x$path) && x$format != "synthetic") {
        stop("PCA checkpointing requires a file-backed source.",
            call. = FALSE)
    }
    sidecar <- paste0(path, ".checkpoint.rds")
    signature <- massive_pca_checkpoint_signature(x, path, rank,
        center, scale, backend, workers, seed, resources)
    if (resume) {
        state <- massive_pca_checkpoint_load(sidecar, signature, x,
            rank, resources, paste0(path, ".part"), workers,
            backend)
    } else {
        if (file.exists(sidecar)) {
            stop("PCA checkpoint already exists; use `resume = TRUE`.",
                call. = FALSE)
        }
        state <- list(signature = signature, stage = "pending",
            completed_rows = 0)
        massive_checkpoint_write(state, sidecar)
    }
    state <- massive_pca_checkpoint_fit(x, rank, center, scale,
        backend, workers, seed, resources, state, sidecar)
    progress <- function(rows) {
        current <- state
        current$completed_rows <- rows
        massive_checkpoint_write(current, sidecar)
    }
    list(fit = state$fit, rows = state$completed_rows,
        progress = progress,
        resume_part = resume && file.exists(paste0(path, ".part")),
        sidecar = sidecar)
}

massive_pca_project <- function(x, path, fit, resources, workers,
        backend, prepared, devices, memory_limit, checkpoint, resume) {
    if (length(devices) > 1L) {
        resources <- massive_pca_project_devices(x, path, fit,
            resources, memory_limit, devices, checkpoint, resume)
    } else if (backend == "cuda") {
        massive_pca_project_cuda_cpp(x, path, fit$loadings,
            fit$center, fit$scale, resources$chunk_rows,
            prepared$rows, prepared$progress, prepared$resume_part)
    } else {
        massive_pca_project_cpp(x, path, fit$loadings,
            fit$center, fit$scale, resources$chunk_rows, workers,
            prepared$rows, prepared$progress, prepared$resume_part)
    }
    resources
}

run_massive_pca <- function(x, ncomp, xtest, center, scale, backend,
                            n.cores, seed, tsne_init, output,
                            chunk_rows, memory_limit, checkpoint, resume,
                            devices) {
    massive_checkpoint_validate_controls(checkpoint, resume)
    validated <- validate_massive_pca(
        x, ncomp, xtest, center, scale, backend, tsne_init
    )
    rank <- validated$rank
    backend <- validated$backend
    devices <- massive_validate_devices(devices, backend, x$nrow)
    if (length(devices)) {
        previous <- massive_cuda_memory_cpp()$device
        on.exit(massive_cuda_select_cpp(previous))
        massive_cuda_select_cpp(devices[[1L]])
    }
    workers <- normalize_pca_threads(n.cores)
    resources <- massive_pca_resources(
        x, rank, workers, chunk_rows, memory_limit, backend
    )
    if (length(devices) > 1L) resources$gpu_devices <- devices
    path <- massive_pca_output(output, resources$output_bytes,
        x$ncol, rank, resume)
    if (length(devices) > 1L) massive_pca_shards_check(x, path,
        rank, resources$output_bytes, devices, resume)
    message("EXPERIMENTAL out-of-core PCA: ", x$nrow, " x ", x$ncol,
        "; method=", if (massive_pca_covariance_route(
            x$ncol, backend)) "streamed_covariance" else {
            "streamed_randomized"
        },
        "; backend=", backend, "; CPU workers=", workers,
        "; chunk rows=", resources$chunk_rows,
        "; estimated algorithm RAM=",
        round(resources$peak_ram_bytes / 1024^2),
        " MB; scores=", round(resources$output_bytes / 1024^2), " MB")
    started <- proc.time()[[3L]]
    prepared <- massive_pca_prepare(x, path, rank, center, scale,
        backend, workers, seed, resources, checkpoint, resume)
    resources <- massive_pca_project(x, path, prepared$fit,
        resources, workers, backend, prepared, devices, memory_limit,
        checkpoint, resume)
    if (length(devices) == 1L) resources$gpu_devices <- devices
    fit <- finish_massive_pca(
        prepared$fit, path, x, rank, workers, seed, resources, started,
        backend
    )
    if (!is.null(prepared$sidecar)) unlink(prepared$sidecar)
    fit
}
