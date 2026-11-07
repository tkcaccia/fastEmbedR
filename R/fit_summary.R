fastembedr_summary_stages <- function(x) {
    if (!is.null(x$pipeline$stage_seconds))
        return(as.list(x$pipeline$stage_seconds))
    if (!is.null(x$stage_seconds))
        return(as.list(x$stage_seconds))
    if (is.matrix(x$timings) && "elapsed" %in%
        colnames(x$timings))
        return(as.list(x$timings[, "elapsed"]))
    if (is.list(x$timing) || is.numeric(x$timing))
        return(as.list(x$timing))
    total <- x$elapsed_sec %||% x$elapsed_seconds %||%
        x$parameters$elapsed_sec %||% x$metrics$elapsed
    if (is.null(total)) return(list())
    list(total = as.numeric(total[[1L]]))
}

fastembedr_summary_memory <- function(x) {
    resources <- x$resources %||% list()
    ram <- resources$peak_ram_bytes %||%
        resources$estimated_peak_ram_bytes %||% NA_real_
    vram <- resources$peak_vram_bytes %||%
        resources$estimated_peak_vram_bytes %||% NA_real_
    observed <- x$metal_allocated_bytes_observed
    if (is.null(observed)) observed <- NA_real_
    list(ram_bytes = ram, vram_bytes = vram,
        metal_allocated_bytes_observed = observed,
        ram_basis = if (is.na(ram)) "not_recorded" else "estimate",
        vram_basis = if (is.na(vram)) "not_recorded" else "estimate")
}

fastembedr_summary_scope <- function(x) {
    scope <- x$fit_scope %||% x$pipeline$fit_scope
    if (!is.null(scope)) return(if (grepl("landmark", scope))
        "landmark" else "full")
    if (isTRUE(x$parameters$landmark) ||
        !is.null(x$n_reference))
        return("landmark")
    "full"
}

fastembedr_summary_fit <- function(x) {
    backed <- inherits(x, "fastEmbedR_massive_matrix") ||
        inherits(x, c("fastEmbedR_massive_pca",
            "fastEmbedR_massive_clusters", "fastEmbedR_massive_leiden",
            "fastEmbedR_massive_louvain_level",
            "fastEmbedR_massive_walktrap"))
    list(backend = x$backend %||% x$parameters$backend %||%
            NA_character_,
        devices = x$gpu_devices %||% x$parameters$devices,
        device_identity = x$gpu_identity,
        scope = fastembedr_summary_scope(x),
        stage_seconds = fastembedr_summary_stages(x),
        memory_use = fastembedr_summary_memory(x),
        output_rows = if (backed) "massive_read_rows(fit, first, n)"
            else "fit$layout or fit$scores or fit$membership")
}

#' Summarize a fastEmbedR fit without loading file-backed output
#'
#' Reports the actual recorded backend and devices, fit scope, stage times,
#' and available memory estimates. Unknown memory is `NA`, never inferred
#' from the input size. Use [massive_read_rows()] with explicit row bounds
#' for file-backed output.
#'
#' @param object A result from `pca()`, `tsne()`, `umap()`, or
#'   `graph_cluster()`.
#' @param ... Unused.
#' @return A list with `backend`, `devices`, `device_identity`, `scope`,
#'   `stage_seconds`, `memory_use`, and `output_rows`.
#' @name fastembedr_fit_summary
NULL

#' @rdname fastembedr_fit_summary
#' @export
summary.fastEmbedR_pca <- function(object, ...) {
    fastembedr_summary_fit(object)
}

#' @rdname fastembedr_fit_summary
#' @export
summary.fastEmbedR_embedding <- function(object, ...) {
    fastembedr_summary_fit(object)
}

#' @rdname fastembedr_fit_summary
#' @export
summary.fastEmbedR_graph_cluster <- function(object, ...) {
    fastembedr_summary_fit(object)
}

#' @rdname fastembedr_fit_summary
#' @export
summary.fastEmbedR_landmark_clusters <- function(object, ...) {
    fastembedr_summary_fit(object)
}

#' @rdname fastembedr_fit_summary
#' @export
summary.fastEmbedR_massive_matrix <- function(object, ...) {
    if (is.null(object$method)) stop(
        "A matrix descriptor is input data, not a fit.", call. = FALSE)
    fastembedr_summary_fit(object)
}

#' @rdname fastembedr_fit_summary
#' @export
summary.fastEmbedR_massive_clusters <- function(object, ...) {
    fastembedr_summary_fit(object)
}

#' @rdname fastembedr_fit_summary
#' @export
summary.fastEmbedR_massive_leiden <- function(object, ...) {
    fastembedr_summary_fit(object)
}

#' @rdname fastembedr_fit_summary
#' @export
summary.fastEmbedR_massive_louvain_level <- function(object, ...) {
    fastembedr_summary_fit(object)
}

#' @rdname fastembedr_fit_summary
#' @export
summary.fastEmbedR_massive_walktrap <- function(object, ...) {
    fastembedr_summary_fit(object)
}
