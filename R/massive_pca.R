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

massive_cgroup_number <- function(path) {
    if (!file.exists(path)) return(NA_real_)
    value <- tryCatch(readLines(path, n = 1L, warn = FALSE),
        error = function(e) character())
    if (length(value) != 1L) return(NA_real_)
    suppressWarnings(as.numeric(value))
}

massive_linux_cgroup_v2_available <- function(v2, cgroup) {
    lines <- tryCatch(readLines(cgroup, warn = FALSE),
        error = function(e) character())
    found <- grep("^0::/", lines, value = TRUE)
    if (length(found) != 1L) return(NA_real_)
    relative <- sub("^0::/", "", found)
    segments <- strsplit(relative, "/", fixed = TRUE)[[1L]]
    if (any(segments %in% c(".", ".."))) return(NA_real_)
    available <- NA_real_
    repeat {
        directory <- file.path(v2, relative)
        limit <- massive_cgroup_number(file.path(directory, "memory.max"))
        used <- massive_cgroup_number(file.path(directory,
            "memory.current"))
        if (is.finite(limit) && is.finite(used)) {
            free <- max(0, limit - used)
            available <- if (is.na(available)) free else
                min(available, free)
        }
        if (!nzchar(relative)) break
        relative <- dirname(relative)
        if (identical(relative, ".")) relative <- ""
    }
    available
}

massive_linux_available_ram <- function(meminfo = "/proc/meminfo",
        v2 = "/sys/fs/cgroup", v1 = "/sys/fs/cgroup/memory",
        cgroup = "/proc/self/cgroup") {
    lines <- tryCatch(readLines(meminfo, warn = FALSE),
        error = function(e) character())
    found <- grep("^MemAvailable:[[:space:]]+[0-9]+", lines,
        value = TRUE)
    if (length(found) != 1L) return(NA_real_)
    available <- as.numeric(sub(
        "^MemAvailable:[[:space:]]+([0-9]+).*", "\\1", found)) * 1024
    nested <- massive_linux_cgroup_v2_available(v2, cgroup)
    if (is.finite(nested)) available <- min(available, nested)
    roots <- list(
        c(file.path(v2, "memory.max"),
            file.path(v2, "memory.current")),
        c(file.path(v1, "memory.limit_in_bytes"),
            file.path(v1, "memory.usage_in_bytes")))
    for (paths in roots) {
        limit <- massive_cgroup_number(paths[[1L]])
        used <- massive_cgroup_number(paths[[2L]])
        if (is.finite(limit) && is.finite(used))
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
    per_row <- if (backend == "metal") 48 * p + 16 * rank else
        4 * (p + rank)
    maximum <- floor((0.7 * limit - fixed) / per_row)
    if (!is.finite(maximum) || maximum < 1)
        stop("PCA buffers exceed `memory_limit`; reduce `n.cores` ",
            "or use a larger limit.", call. = FALSE)
    requested <- chunk_rows %||% 250000L
    if (length(requested) != 1L || !is.finite(requested) ||
        requested < 1 || requested != floor(requested) ||
        requested > .Machine$integer.max)
        stop("`chunk_rows` must be one positive integer.", call. = FALSE)
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
        if (!is.finite(gpu_max) || gpu_max < 1)
            stop("CUDA PCA buffers exceed the free-VRAM budget.",
                call. = FALSE)
        rows <- as.integer(min(rows, gpu_max))
    }
    result <- list(chunk_rows = rows,
        peak_ram_bytes = fixed + per_row * rows,
        output_bytes = source$nrow * rank * 4,
        input_buffer_bytes = rows * p * 4,
        input_resident_bytes = source_bytes,
        memory_limit_bytes = limit)
    if (!is.null(gpu)) {
        result$peak_vram_bytes <- gpu_fixed + gpu_per_row * rows
        result$free_vram_bytes <- gpu$free_bytes
        result$gpu_device <- gpu$device
    }
    if (backend == "metal") {
        result$peak_vram_bytes <- NA_real_
        result$metal_memory_model <- "unified_shared"
        result$metal_buffer_bytes_estimate <- 8 * rows * (p + rank)
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

massive_pca_covariance_metal <- function(source, chunk_rows, center) {
    p <- source$ncol
    cross <- matrix(0, p, p)
    means <- numeric(p)
    completed <- 0
    allocated <- 0
    started <- proc.time()[[3L]]
    for (first in seq.int(1, source$nrow, by = chunk_rows)) {
        rows <- as.integer(min(chunk_rows, source$nrow - first + 1))
        block <- massive_read_rows_cpp(source, first, rows,
            4 * rows * p, TRUE, TRUE)
        moment <- pca_crossprod_block_metal_cpp(block, center)
        observed <- moment$metal_allocated_bytes_observed
        if (length(observed) != 1L || !is.finite(observed) ||
            observed <= 0) stop("Metal allocation was not observed")
        allocated <- max(allocated, observed)
        block_mean <- moment$center
        block_cross <- moment$cross
        updated <- completed + rows
        if (center) {
            delta <- block_mean - means
            cross <- cross + block_cross +
                tcrossprod(delta) * (completed * rows / updated)
            means <- means + delta * (rows / updated)
        } else cross <- cross + block_cross
        completed <- updated
    }
    covariance <- cross / (source$nrow - 1)
    if (any(!is.finite(covariance))) stop(
        "Metal PCA covariance contains nonfinite values.",
        call. = FALSE)
    list(center = means, covariance = (covariance + t(covariance)) / 2,
        fit_statistics_seconds = unname(proc.time()[[3L]] - started),
        metal_allocated_bytes_observed = allocated)
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
    stats <- if (is.list(saved_stats)) saved_stats else if (
            length(resources$gpu_devices %||% integer()) > 1L) {
        massive_pca_covariance_devices(source, rank, center, resources)
    } else if (backend == "metal") {
        massive_pca_covariance_metal(source, resources$chunk_rows,
            center)
    } else covariance_fn(source, resources$chunk_rows, n.cores,
        center, saved_stats)
    deviations <- if (scale) sqrt(diag(stats$covariance)) else {
        rep(1, source$ncol)
    }
    if (any(!is.finite(deviations)) || any(deviations <= 0)) {
        stop("PCA scaling requires positive variance in every column.",
            call. = FALSE)
    }
    covariance <- stats$covariance / outer(deviations, deviations)
    eig <- eigen(covariance, symmetric = TRUE)
    fit <- list(
        center = stats$center, scale = deviations,
        decomposition_backend = "cpu_eigen",
        loadings = eig$vectors[, seq_len(rank), drop = FALSE],
        singular_values = sqrt(pmax(eig$values[seq_len(rank)], 0) *
            (source$nrow - 1))
    )
    if (!is.null(stats$fit_devices)) {
        fit$fit_devices <- stats$fit_devices
        fit$fit_per_device <- stats$fit_per_device
        fit$fit_statistics_backend <- "distributed_cuda_covariance"
        fit$fit_statistics_seconds <- stats$fit_statistics_seconds
    }
    if (backend == "metal") {
        fit$fit_statistics_backend <- "metal_streamed_crossprod"
        fit$fit_statistics_seconds <- stats$fit_statistics_seconds
        fit$metal_allocated_bytes_observed <-
            stats$metal_allocated_bytes_observed
    }
    fit
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
    first <- massive_pca_wide_action_run(source, moments, omega,
        resources, rank, n.cores, backend)
    factor <- qr(first$value)
    if (factor$rank < rank) {
        stop("Wide PCA sketch is rank deficient.", call. = FALSE)
    }
    basis <- qr.Q(factor)
    second <- massive_pca_wide_action_run(source, moments, basis,
        resources, rank, n.cores, backend)
    small <- crossprod(basis, second$value) / (source$nrow - 1)
    eig <- eigen((small + t(small)) / 2, symmetric = TRUE)
    fit <- list(center = moments$center, scale = moments$scale,
        decomposition_backend = "cpu_qr_eigen",
        loadings = basis %*% eig$vectors[, seq_len(rank), drop = FALSE],
        singular_values = sqrt(pmax(eig$values[seq_len(rank)], 0) *
            (source$nrow - 1)), sketch_size = width,
        moments_backend = "cpu", sketch_backend = backend)
    if (!is.null(first$per_device)) {
        fit$fit_devices <- resources$gpu_devices
        fit$fit_sketch_backend <- "distributed_cuda_action"
        fit$fit_sketch_seconds <- first$seconds + second$seconds
        fit$fit_per_device <- Map(function(a, b) {
            a$second_action_seconds <- b$seconds
            a
        }, first$per_device, second$per_device)
    }
    fit
}

massive_pca_wide_action_run <- function(source, moments, basis,
        resources, rank, n.cores, backend) {
    if (length(resources$gpu_devices %||% integer()) > 1L) return(
        massive_pca_wide_action_devices(source, rank,
            moments$center, moments$scale, basis, resources))
    action_fn <- if (backend == "cuda") {
        massive_pca_wide_action_cuda_cpp
    } else massive_pca_wide_action_cpp
    args <- list(source, moments$center, moments$scale,
        basis, resources$chunk_rows)
    if (backend == "cpu") args <- c(args, list(n.cores))
    list(value = do.call(action_fn, args), per_device = NULL)
}

validate_massive_pca <- function(x, ncomp, xtest, center, scale,
                                    backend, tsne_init) {
    if (!inherits(x, "fastEmbedR_massive_matrix")) {
        stop("EXPERIMENTAL out-of-core PCA requires massive_matrix(x).",
            call. = FALSE)
    }
    backend <- resolve_embedding_backend(backend)
    if (backend == "metal" &&
        (!isTRUE(embedding_metal_available_cpp()) ||
            x$ncol > 1024L)) stop(
        "Metal streamed PCA needs native Metal and <=1024 columns; ",
        "no CPU fallback was used.", call. = FALSE)
    if (backend == "cuda" && !isTRUE(
            embedding_cuda_available_cpp())) {
        stop("EXPERIMENTAL out-of-core CUDA PCA requires native CUDA; ",
            "no CPU fallback was used.", call. = FALSE)
    }
    if (!identical(tsne_init, FALSE) && !identical(tsne_init, TRUE)) {
        stop("`tsne_init` must be TRUE or FALSE.", call. = FALSE)
    }
    massive_pca_validate_xtest(x, xtest)
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
    if (tsne_init && !rank %in% c(2L, 3L)) stop(
        "Out-of-core t-SNE initialization needs two or three components.",
        call. = FALSE)
    list(rank = rank, backend = backend)
}

massive_pca_validate_xtest <- function(x, xtest) {
    if (is.null(xtest)) return(invisible(NULL))
    if (!inherits(xtest, "fastEmbedR_massive_matrix") ||
        xtest$ncol != x$ncol || xtest$nrow < 1L ||
        xtest$format == "synthetic") stop(
        "Out-of-core `xtest` needs a nonempty matching massive_matrix().",
        call. = FALSE)
    invisible(NULL)
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
    if (!is.null(fit$fit_per_device)) {
        worker_ram <- sum(vapply(fit$fit_per_device,
            `[[`, 0, "estimated_ram_bytes"))
        parent <- resources$parent_peak_ram_bytes %||%
            resources$peak_ram_bytes
        fit$resources$parent_peak_ram_bytes <- parent
        fit$resources$fit_peak_ram_bytes <- parent + worker_ram
        fit$resources$peak_ram_bytes <- if (
                is.na(resources$peak_ram_bytes)) NA_real_ else
            max(resources$peak_ram_bytes,
                fit$resources$fit_peak_ram_bytes)
        fit$resources$peak_ram_scope <- "process_tree_estimate"
    }
    if (!is.null(resources$gpu_devices)) {
        fit$gpu_devices <- resources$gpu_devices
        if (length(resources$gpu_devices) > 1L)
            fit$gpu_identity <- massive_cuda_device_uuids(
                resources$gpu_devices)
    }
    fit$experimental <- TRUE
    fit$implementation_identity <-
        massive_checkpoint_implementation_identity()
    fit$elapsed_sec <- unname(proc.time()[[3L]] - started)
    fit$source_identity <- if (x$format == "memory") NULL else
        massive_checkpoint_source_identity(x)
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
#' @keywords internal
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
        isTRUE(fit$backend %in% c("cpu", "cuda", "metal")) &&
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
    massive_pca_check_test_scores(fit)
    massive_pca_check_init(fit)
    fit
}

massive_pca_check_test_scores <- function(fit) {
    if (is.null(fit$test_projection)) return(invisible(NULL))
    test <- fit$test_projection
    if (!inherits(test$scores, "fastEmbedR_massive_matrix") ||
        !inherits(fit$scores_test, "fastEmbedR_massive_matrix")) stop(
        "PCA test scores changed since completion.", call. = FALSE)
    identity <- tryCatch(
        massive_checkpoint_file_identity(test$scores$path),
        error = function(e) NULL)
    if (!identical(identity, test$score_identity) ||
        !identical(test$scores$path, fit$scores_test$path)) stop(
        "PCA test scores changed since completion.", call. = FALSE)
    invisible(NULL)
}

massive_pca_test_path <- function(output) {
    sub("\\.f32$", ".test.f32", output, ignore.case = TRUE)
}

massive_pca_init_path <- function(output) {
    sub("\\.f32$", ".init.f32", output, ignore.case = TRUE)
}

massive_pca_check_init <- function(fit) {
    if (is.null(fit$tsne_init)) return(invisible(NULL))
    init <- fit$tsne_init
    identity <- tryCatch(massive_checkpoint_file_identity(init$path),
        error = function(e) NULL)
    if (!inherits(init, "fastEmbedR_massive_matrix") ||
        !identical(identity, fit$init_identity) ||
        init$nrow != fit$scores$nrow || init$ncol != fit$ncomp) stop(
        "PCA t-SNE initialization changed since completion.",
        call. = FALSE)
    invisible(NULL)
}

massive_pca_preflight_test <- function(xtest, path, rank, checkpoint) {
    if (is.null(xtest)) return(invisible(NULL))
    if (checkpoint && xtest$format == "memory") stop(
        "Checkpointed PCA test projection needs a file-backed `xtest`.",
        call. = FALSE)
    massive_float_output(massive_pca_test_path(path),
        4 * xtest$nrow * rank)
    invisible(NULL)
}

massive_pca_preflight_init <- function(path, rows, rank, requested) {
    if (requested) massive_float_output(massive_pca_init_path(path),
        4 * rows * rank)
    invisible(NULL)
}

massive_pca_report <- function(x, rank, backend, workers, resources) {
    method <- if (massive_pca_covariance_route(x$ncol, backend))
        "streamed_covariance" else "streamed_randomized"
    message("EXPERIMENTAL out-of-core PCA: ", x$nrow, " x ", x$ncol,
        "; method=", method, "; backend=", backend,
        "; CPU workers=", workers,
        "; chunk rows=", resources$chunk_rows,
        "; estimated algorithm RAM=",
        round(resources$peak_ram_bytes / 1024^2),
        " MB; scores=", round(resources$output_bytes / 1024^2), " MB")
}

massive_pca_project_test <- function(fit, xtest, output, workers,
        chunk_rows, memory_limit, devices, checkpoint) {
    if (is.null(xtest)) return(fit)
    test_path <- massive_pca_test_path(output)
    projected <- massive_project_pca(fit, xtest, test_path,
        backend = fit$backend, n.cores = workers,
        chunk_rows = chunk_rows, memory_limit = memory_limit,
        devices = devices, checkpoint = checkpoint)
    fit$scores_test <- projected$scores
    fit$test_projection <- projected
    massive_checkpoint_write(list(version = 1L, fit = fit),
        fit$manifest_path)
    fit
}

massive_pca_init_stats <- function(scores, chunk_rows) {
    count <- 0
    mean <- numeric(scores$ncol)
    squares <- numeric(scores$ncol)
    first <- 1
    while (first <= scores$nrow) {
        rows <- as.integer(min(chunk_rows, scores$nrow - first + 1))
        block <- massive_read_rows_cpp(scores, first, rows,
            8 * rows * scores$ncol)
        if (any(!is.finite(block))) stop(
            "PCA scores contain nonfinite values.", call. = FALSE)
        block_mean <- colMeans(block)
        next_count <- count + rows
        delta <- block_mean - mean
        squares <- squares + colSums(sweep(block, 2L,
            block_mean)^2) + delta^2 * count * rows / next_count
        mean <- mean + delta * rows / next_count
        count <- next_count
        first <- first + rows
    }
    deviation <- sqrt(max(squares / (count - 1)))
    list(center = mean, scale = if (deviation > 0) 1e-4 / deviation
        else 1)
}

massive_pca_write_init <- function(scores, output, stats, chunk_rows) {
    part <- paste0(output, ".part")
    con <- file(part, "wb")
    closed <- FALSE
    on.exit({
        if (!closed) close(con)
        if (!closed) unlink(part)
    })
    first <- 1
    while (first <= scores$nrow) {
        rows <- as.integer(min(chunk_rows, scores$nrow - first + 1))
        block <- massive_read_rows_cpp(scores, first, rows,
            8 * rows * scores$ncol)
        block <- sweep(block, 2L, stats$center)
        block <- block * stats$scale
        writeBin(as.numeric(t(block)), con, size = 4L,
            endian = "little")
        first <- first + rows
    }
    close(con)
    closed <- TRUE
    if (file.info(part)$size != 4 * scores$nrow * scores$ncol ||
        !file.rename(part, output)) stop(
        "PCA t-SNE initialization did not finalize.", call. = FALSE)
    massive_matrix(output, nrow = scores$nrow, ncol = scores$ncol)
}

#' Create file-backed t-SNE initialization from streamed PCA
#'
#' Centers completed PCA scores and scales their largest sample standard
#' deviation to `1e-4`, matching resident PCA initialization. The two
#' bounded passes run on CPU and write a new row-major `.f32` file.
#'
#' @param fit A completed out-of-core PCA fit with two or three components.
#' @param output New `.f32` initialization path.
#' @param chunk_rows Maximum score rows read per pass, capped at 16 MB.
#' @param resume If `TRUE`, replace an interrupted `.part` file and restart
#'   both bounded passes from the verified PCA scores.
#' @return A file-backed [massive_matrix()] descriptor for
#'   [massive_tsne_optimize()].
#' @keywords internal
massive_pca_tsne_init <- function(fit, output,
        chunk_rows = 65536L, resume = FALSE) {
    if (!inherits(fit, "fastEmbedR_massive_pca") ||
        !fit$ncomp %in% c(2L, 3L) ||
        !identical(massive_checkpoint_file_identity(
            fit$scores$path), fit$score_identity)) stop(
        "A verified file-backed 2D/3D PCA fit is required.",
        call. = FALSE)
    rows <- integer_scalar(chunk_rows)
    if (is.na(rows) || rows < 1L) stop(
        "`chunk_rows` must be positive.", call. = FALSE)
    rows <- as.integer(min(rows, floor(16e6 / (8 * fit$ncomp))))
    if (!is.logical(resume) || length(resume) != 1L || is.na(resume))
        stop("`resume` must be TRUE or FALSE.", call. = FALSE)
    path <- massive_float_output(output,
        4 * fit$scores$nrow * fit$ncomp, resume = resume)
    stats <- massive_pca_init_stats(fit$scores, rows)
    result <- massive_pca_write_init(fit$scores, path, stats, rows)
    if (!identical(massive_checkpoint_file_identity(
            fit$scores$path), fit$score_identity)) stop(
        "PCA score file changed during initialization.",
        call. = FALSE)
    result$normalization_backend <- "cpu"
    result$center <- stats$center
    result$scale <- stats$scale
    result
}

massive_pca_attach_init <- function(fit, requested, path) {
    if (!requested) return(fit)
    init <- massive_pca_tsne_init(fit, massive_pca_init_path(path))
    fit$tsne_init <- init
    fit$init_identity <- massive_checkpoint_file_identity(init$path)
    massive_checkpoint_write(list(version = 1L, fit = fit),
        fit$manifest_path)
    fit
}

massive_pca_checkpoint_signature <- function(x, path, rank, center,
        scale, backend, workers, seed, resources) {
    version <- if (length(resources$gpu_devices %||% integer()) > 1L)
        7L else 5L
    massive_checkpoint_signature(list(version = version,
        implementation = massive_checkpoint_implementation_identity(),
        source = massive_checkpoint_source_identity(x),
        output = path, rank = rank, center = center, scale = scale,
        backend = backend, workers = workers, seed = seed,
        covariance = massive_pca_covariance_route(x$ncol, backend),
        chunk_rows = resources$chunk_rows,
        gpu_devices = resources$gpu_devices %||% NULL,
        gpu_identity = if (version == 7L) massive_cuda_device_uuids(
            resources$gpu_devices) else NULL))
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
    if (length(resources$gpu_devices %||% integer()) > 1L &&
        !massive_pca_fit_devices_valid(state$fit, x,
            resources$gpu_devices)) stop(
        "PCA checkpoint fit has invalid GPU row ownership.",
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

massive_pca_fit_devices_valid <- function(fit, x, devices) {
    if (!is.list(fit) || !identical(fit$fit_devices, devices) ||
        !is.list(fit$fit_per_device) ||
        length(fit$fit_per_device) != length(devices)) return(FALSE)
    tasks <- massive_pca_fit_tasks(x, devices)
    all(vapply(seq_along(tasks), function(i) {
        part <- fit$fit_per_device[[i]]
        is.list(part) && identical(part$device,
            tasks[[i]]$device) && identical(part$row_range,
            tasks[[i]]$row_range) && identical(part$rows,
                as.double(diff(tasks[[i]]$row_range) + 1))
    }, logical(1L)))
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
    if (backend == "metal") {
        state$fit <- massive_pca_decomposition(x, rank, center,
            scale, resources, workers, backend, seed)
        state$fit_hash <- massive_checkpoint_signature(state$fit)
        state$stage <- "fit"
        massive_checkpoint_write(state, sidecar)
        return(state)
    }
    if (length(resources$gpu_devices %||% integer()) > 1L) return(
        massive_pca_checkpoint_fit_devices(x, rank, center, scale,
            backend, workers, seed, resources, state, sidecar))
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

massive_pca_checkpoint_fit_devices <- function(x, rank, center,
        scale, backend, workers, seed, resources, state, sidecar) {
    fit <- massive_pca_decomposition(x, rank, center, scale,
        resources, workers, backend, seed)
    if (!massive_pca_fit_devices_valid(fit, x,
        resources$gpu_devices)) stop(
        "Multi-GPU PCA fit has invalid GPU row ownership.",
        call. = FALSE)
    state$fit <- fit
    state$fit_hash <- massive_checkpoint_signature(fit)
    state$stage <- "fit"
    massive_checkpoint_write(state, sidecar)
    state
}

massive_pca_prepare <- function(x, path, rank, center, scale,
        backend, workers, seed, resources, checkpoint, resume) {
    if (!checkpoint) {
        fit <- massive_pca_decomposition(x, rank, center, scale,
            resources, workers, backend, seed)
        if (length(resources$gpu_devices %||% integer()) > 1L &&
            !massive_pca_fit_devices_valid(fit, x,
                resources$gpu_devices)) stop(
            "Multi-GPU PCA fit has invalid GPU row ownership.",
            call. = FALSE)
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
    } else if (backend == "metal") {
        resources$metal_projection_allocated_bytes_observed <-
            massive_pca_project_metal(x, path, fit, resources,
            prepared, checkpoint)
    } else {
        massive_pca_project_cpp(x, path, fit$loadings,
            fit$center, fit$scale, resources$chunk_rows, workers,
            prepared$rows, prepared$progress, prepared$resume_part)
    }
    resources
}

massive_pca_project_metal <- function(x, path, fit, resources,
        prepared, checkpoint) {
    part <- paste0(path, ".part")
    completed <- prepared$rows
    allocated <- 0
    con <- file(part, if (prepared$resume_part) "r+b" else "wb")
    closed <- FALSE
    on.exit({
        if (!closed) try(close(con), silent = TRUE)
        if (!checkpoint && !closed) unlink(part)
    })
    if (completed > 0) seek(con, where =
        completed * ncol(fit$loadings) * 4,
        origin = "start", rw = "write")
    first <- completed + 1
    while (first <= x$nrow) {
        rows <- as.integer(min(resources$chunk_rows, x$nrow - first + 1))
        block <- massive_read_rows_cpp(x, first, rows,
            4 * rows * x$ncol, TRUE, TRUE)
        scores <- pca_project_block_metal_cpp(block, fit$loadings,
            fit$center, fit$scale)
        observed <- attr(scores, "metal_allocated_bytes_observed")
        if (length(observed) != 1L || !is.finite(observed) ||
            observed <= 0) stop("Metal allocation was not observed")
        allocated <- max(allocated, observed)
        writeBin(as.numeric(t(scores)), con, size = 4L,
            endian = "little")
        flush(con)
        if (!is.null(prepared$progress))
            prepared$progress(first + rows - 1)
        first <- first + rows
    }
    truncate(con)
    close(con)
    closed <- TRUE
    if (file.info(part)$size != resources$output_bytes ||
        !file.rename(part, path)) stop(
        "Metal PCA scores did not finalize; .part retained.",
        call. = FALSE)
    allocated
}

massive_pca_projection_backend <- function(fit, x, backend, devices,
        workers) {
    if (!inherits(fit, "fastEmbedR_massive_pca") ||
        !is.matrix(fit$loadings) || !is.numeric(fit$loadings) ||
        ncol(fit$loadings) < 1L ||
        !inherits(x, "fastEmbedR_massive_matrix") ||
        nrow(fit$loadings) != x$ncol ||
        !is.numeric(fit$center) || !is.numeric(fit$scale) ||
        length(fit$center) != x$ncol ||
        length(fit$scale) != x$ncol ||
        !all(is.finite(c(fit$loadings, fit$center, fit$scale))) ||
        any(fit$scale <= 0) ||
        !isTRUE(fit$backend %in% c("cpu", "cuda", "metal"))) stop(
        "Projection needs a valid streamed PCA fit and ",
        "matching matrix columns.",
        call. = FALSE)
    backend <- validate_pca_backend(backend %||% fit$backend)
    if (backend == "cuda" &&
        !isTRUE(embedding_cuda_available_cpp())) stop(
        "CUDA PCA projection needs native CUDA; no fallback was used.",
        call. = FALSE)
    if (backend == "metal" &&
        (!isTRUE(embedding_metal_available_cpp()) ||
            x$ncol > 1024L)) stop(
        "Metal PCA projection needs native Metal and <=1024 columns; ",
        "no fallback was used.", call. = FALSE)
    if (backend == "metal" && workers != 1L) stop(
        "Metal PCA projection requires `n.cores = 1`.",
        call. = FALSE)
    list(backend = backend,
        devices = massive_validate_devices(devices, backend, x$nrow))
}

massive_pca_projection_landmarks <- function(fit, x, selection) {
    if (is.null(selection)) return(NULL)
    if (!is.list(selection) ||
        !inherits(selection$data, "fastEmbedR_massive_matrix") ||
        is.null(fit$source_identity) ||
        is.null(selection$source_identity) ||
        !identical(fit$source_identity,
            massive_checkpoint_source_identity(selection$data)) ||
        !identical(selection$source_identity,
            massive_checkpoint_source_identity(x))) stop(
        "PCA landmark selection does not match fit and source.",
        call. = FALSE)
    indices <- selection$indices
    if (!is.numeric(indices) ||
        length(indices) != selection$data$nrow ||
        any(!is.finite(indices)) || any(indices < 1) ||
        any(indices > x$nrow) || any(indices != floor(indices)) ||
        is.unsorted(indices, strictly = TRUE) ||
        !identical(selection$indices_signature,
            massive_checkpoint_signature(indices))) stop(
        "PCA landmark row IDs changed since selection.",
        call. = FALSE)
    list(indices = indices, count = length(indices),
        method = selection$method, seed = selection$seed,
        indices_signature = selection$indices_signature)
}

massive_pca_projection_checkpoint <- function(x, path, model, selected,
        resources, backend, workers, devices, checkpoint, resume) {
    massive_checkpoint_validate_controls(checkpoint, resume)
    sidecar <- paste0(path, ".checkpoint.rds")
    if (!checkpoint) {
        if (file.exists(sidecar)) stop("PCA projection checkpoint exists.")
        return(list(rows = 0, progress = NULL,
            resume_part = FALSE, sidecar = NULL))
    }
    if (x$format == "memory") stop(
        "PCA projection checkpointing needs a file-backed source.",
        call. = FALSE)
    signature <- massive_checkpoint_signature(list(version = 1L,
        implementation = massive_checkpoint_implementation_identity(),
        source = massive_checkpoint_source_identity(x),
        model = model, landmarks = selected$indices_signature %||% NULL,
        output = path, backend = backend,
        workers = workers, chunk_rows = resources$chunk_rows,
        devices = devices, gpu_identity = if (length(devices) > 1L)
            massive_cuda_device_uuids(devices) else NULL))
    part <- paste0(path, ".part")
    if (resume) {
        state <- tryCatch(readRDS(sidecar), error = function(e) NULL)
        rows <- if (is.list(state)) state$completed_rows else NULL
        size <- if (file.exists(part)) file.info(part)$size else 0
        valid <- is.list(state) &&
            identical(state$signature, signature) &&
            is.numeric(rows) && length(rows) == 1L &&
            is.finite(rows) && rows == floor(rows) &&
            rows >= 0 && rows <= x$nrow &&
            (rows == x$nrow || rows %% resources$chunk_rows == 0) &&
            (length(devices) <= 1L || rows == 0) &&
            is.finite(size) &&
            size >= rows * ncol(model$loadings) * 4 &&
            size <= resources$output_bytes
        if (!isTRUE(valid)) stop(
            "PCA projection checkpoint does not match this run.",
            call. = FALSE)
    } else {
        if (file.exists(sidecar)) stop("PCA projection checkpoint exists.")
        state <- list(signature = signature, completed_rows = 0)
        massive_checkpoint_write(state, sidecar)
    }
    progress <- function(rows) {
        state$completed_rows <- rows
        massive_checkpoint_write(state, sidecar)
    }
    list(rows = state$completed_rows, progress = progress,
        resume_part = resume && file.exists(part), sidecar = sidecar)
}

#' Experimental file-backed PCA projection
#'
#' Projects a file-backed matrix through a completed streamed PCA fit.
#' Fit PCA on a sampled landmark file with [pca()], then use this call to
#' project the full source without loading all rows into RAM. The selected
#' backend writes row-major float32 scores to a new `.f32` file.
#'
#' @param fit A completed out-of-core PCA fit.
#' @param x A [massive_matrix()] descriptor with matching columns.
#' @param output New `.f32` score-file path.
#' @param backend Projection backend; defaults to the fit backend.
#' @param n.cores CPU worker limit; Metal requires one.
#' @param chunk_rows Maximum projection rows per block.
#' @param memory_limit Conservative algorithm RAM budget.
#' @param devices Optional distinct CUDA device IDs for row sharding.
#' @param landmarks Optional [massive_select_landmarks()] result. When
#'   supplied, its row IDs and source identity must match the fit and input.
#' @param checkpoint Save partial score batches for recovery.
#' @param resume Continue from a matching checkpoint and partial score file.
#' @return A list with file-backed `scores`, backend, worker, resource,
#'   input, model, and output identities, plus any verified landmark rows.
#'   This projection is experimental.
#' @export
#' @keywords internal
massive_project_pca <- function(fit, x, output, backend = NULL,
        n.cores = NULL, chunk_rows = NULL, memory_limit = "512MB",
        devices = NULL, landmarks = NULL, checkpoint = FALSE,
        resume = FALSE) {
    workers <- normalize_pca_threads(n.cores)
    route <- massive_pca_projection_backend(fit, x, backend,
        devices, workers)
    selected <- massive_pca_projection_landmarks(fit, x, landmarks)
    backend <- route$backend
    devices <- route$devices
    if (length(devices)) {
        previous <- massive_cuda_memory_cpp()$device
        on.exit(massive_cuda_select_cpp(previous))
        massive_cuda_select_cpp(devices[[1L]])
    }
    rank <- ncol(fit$loadings)
    resources <- massive_pca_resources(x, rank, workers,
        chunk_rows, memory_limit, backend)
    path <- massive_float_output(output, resources$output_bytes,
        resume = resume && file.exists(paste0(output, ".part")))
    if (length(devices) > 1L) massive_pca_shards_check(x, path,
        rank, resources$output_bytes, devices, resume)
    input <- massive_checkpoint_source_identity(x)
    model <- massive_checkpoint_signature(list(
        loadings = fit$loadings, center = fit$center,
        scale = fit$scale, source = fit$source_identity))
    started <- proc.time()[[3L]]
    prepared <- massive_pca_projection_checkpoint(x, path, fit, selected,
        resources, backend, workers, devices, checkpoint, resume)
    resources <- massive_pca_project(x, path, fit, resources,
        workers, backend, prepared, devices, memory_limit,
        checkpoint, resume)
    if (!identical(input, massive_checkpoint_source_identity(x))) stop(
        "PCA projection input changed during calculation.",
        call. = FALSE)
    result <- list(scores = massive_matrix(path, nrow = x$nrow,
        ncol = rank), backend = backend, n_threads = workers,
        resources = resources, source_identity = input,
        model_identity = model,
        mode = if (is.null(selected)) "projection" else "landmark",
        landmarks = selected,
        score_identity = massive_checkpoint_file_identity(path),
        elapsed_sec = unname(proc.time()[[3L]] - started),
        experimental = TRUE)
    class(result) <- "fastEmbedR_massive_pca_projection"
    if (!is.null(prepared$sidecar)) unlink(prepared$sidecar)
    result
}

run_massive_pca <- function(x, ncomp, xtest, center, scale, backend,
                            n.cores, seed, tsne_init, output,
                            chunk_rows, memory_limit, checkpoint, resume,
                            devices) {
    massive_checkpoint_validate_controls(checkpoint, resume)
    validated <- validate_massive_pca(x, ncomp, xtest, center,
        scale, backend, tsne_init)
    rank <- validated$rank
    backend <- validated$backend
    devices <- massive_validate_devices(devices, backend, x$nrow)
    if (length(devices)) {
        previous <- massive_cuda_memory_cpp()$device
        on.exit(massive_cuda_select_cpp(previous))
        massive_cuda_select_cpp(devices[[1L]])
    }
    workers <- normalize_pca_threads(n.cores)
    if (backend == "metal" && workers != 1L) stop(
        "Metal streamed PCA requires `n.cores = 1`.",
        call. = FALSE)
    resources <- massive_pca_resources(x, rank, workers,
        chunk_rows, memory_limit, backend)
    if (length(devices) > 1L) resources$gpu_devices <- devices
    path <- massive_pca_output(output, resources$output_bytes,
        x$ncol, rank, resume)
    massive_pca_preflight_test(xtest, path, rank, checkpoint)
    massive_pca_preflight_init(path, x$nrow, rank, tsne_init)
    if (length(devices) > 1L) massive_pca_shards_check(x, path,
        rank, resources$output_bytes, devices, resume)
    massive_pca_report(x, rank, backend, workers, resources)
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
    fit <- massive_pca_project_test(fit, xtest, path, workers,
        chunk_rows, memory_limit, devices, checkpoint)
    massive_pca_attach_init(fit, tsne_init, path)
}
