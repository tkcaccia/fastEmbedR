#' Cluster a resident landmark graph and assign every observation
#'
#' Selects reference observations, builds one KNN graph among them,
#' fits native Leiden or Louvain, then assigns remaining rows by a
#' distance-weighted vote over reference labels. Reference labels remain
#' fixed. This is landmark approximation, not a full-data graph fit.
#'
#' @param data Resident numeric matrix, data frame, or float32 matrix.
#' @param landmarks Fraction, count, or explicit reference row indices.
#' @param method `"leiden"` or `"louvain"`.
#' @param k Number of reference graph neighbors per landmark.
#' @param vote_k Number of reference neighbors used to assign query rows.
#' @param backend `"cpu"`, `"cuda"`, or `"metal"` for KNN and graph fit.
#' @param metric KNN distance metric.
#' @param weight Reference graph edge weighting.
#' @param resolution Modularity resolution.
#' @param n_iterations Maximum local-moving passes.
#' @param n_runs Number of clustering starts.
#' @param seed Selection and clustering seed.
#' @param n.cores CPU worker limit for selection, fit, and voting.
#' @param devices Optional coordinated CUDA devices for the graph fit.
#' @param parallel.mode CPU fit or restart parallelism.
#' @return A `fastEmbedR_landmark_clusters` list with all-row membership,
#'   confidence, selection, reference fit, stage times, and backend labels.
#'   Graph assembly and label voting are CPU stages on every backend.
#' @keywords internal
cluster_landmarks <- function(data, landmarks, method = c("leiden",
        "louvain"), k = 30L, vote_k = k, backend = NULL,
        metric = "euclidean", weight = "snn", resolution = 1,
        n_iterations = 10L, n_runs = 1L, seed = 1L,
        n.cores = 1L, devices = NULL,
        parallel.mode = c("restarts", "fit")) {
    started <- proc.time()[[3L]]
    method <- match.arg(method)
    backend <- resolve_embedding_backend(backend)
    x <- coerce_embedding_data(data)$data
    selected <- select_landmarks(x, landmarks, seed, n.cores)
    k <- validate_precompute_k(k, selected$n_landmarks - 1L,
        "nrow(reference) - 1")
    vote_k <- validate_precompute_k(vote_k, selected$n_landmarks,
        "nrow(reference)")
    if (selected$n_landmarks >= nrow(x)) stop(
        "Landmark clustering needs fewer reference than input rows.",
        call. = FALSE)
    selection_time <- proc.time()[[3L]] - started
    reference <- x[selected$indices, , drop = FALSE]
    graph <- knn_graph(reference, k = k, backend = backend,
        metric = metric, weight = weight, n.cores = n.cores)
    graph_time <- proc.time()[[3L]] - started - selection_time
    fit <- graph_cluster(graph, method = method, backend = backend,
        resolution = resolution, n_iterations = n_iterations,
        n_runs = n_runs, seed = seed, n.cores = if (backend == "cpu")
            n.cores else 1L, devices = devices,
        parallel.mode = parallel.mode)
    if (!identical(fit$backend, backend)) stop(
        "Landmark graph fit changed backend.", call. = FALSE)
    fit_time <- proc.time()[[3L]] - started - selection_time - graph_time
    assigned <- cluster_landmarks_vote(x, selected, reference,
        fit$membership, vote_k, metric, backend, n.cores)
    times <- c(selection = selection_time, graph = graph_time,
        fit = fit_time, vote = proc.time()[[3L]] - started -
            selection_time - graph_time - fit_time)
    out <- c(assigned, list(selection = selected,
        reference_fit = fit, fit_scope = "landmark_fit_plus_projection",
        method = method, backend = backend,
        gpu_devices = fit$gpu_devices,
        gpu_identity = fit$gpu_identity,
        n_reference = selected$n_landmarks,
        n_query = length(selected$query_indices),
        graph_assembly_backend = "cpu", vote_backend = "cpu",
        stage_seconds = times))
    class(out) <- c("fastEmbedR_landmark_clusters", "list")
    out
}

cluster_landmarks_vote <- function(x, selected, reference, labels,
        vote_k, metric, backend, n.cores) {
    labels <- as.integer(labels)
    if (length(labels) != selected$n_landmarks || anyNA(labels) ||
        any(labels < 1L)) stop("Reference clustering labels changed.",
        call. = FALSE)
    query <- x[selected$query_indices, , drop = FALSE]
    knn <- precompute_query_knn(reference, query, k = vote_k,
        metric = metric, backend = backend, n.cores = n.cores)
    source_gpu <- fastembedr_is_gpu_knn(knn)
    if (source_gpu) knn <- fastembedr_gpu_knn_to_host(knn)
    voted <- massive_vote_landmarks_cpp(knn$indices,
        embedding_dense_double_matrix(knn$distances), labels,
        normalize_nn_threads(n.cores))
    membership <- integer(nrow(x))
    confidence <- numeric(nrow(x))
    membership[selected$indices] <- labels
    membership[selected$query_indices] <- voted$membership
    confidence[selected$indices] <- 1
    confidence[selected$query_indices] <- voted$confidence
    list(membership = membership, confidence = confidence,
        query_knn_backend = if (source_gpu) "cuda_to_host" else
            attr(knn, "backend") %||% knn$backend %||% backend)
}
