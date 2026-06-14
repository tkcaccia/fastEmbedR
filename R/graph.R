#' Build an igraph nearest-neighbour graph
#'
#' `knn_graph()` turns a data matrix, a precomputed [nn()] result, or a
#' `fastEmbedR_embedding` object returned by [opentsne()] or [umap()] into an
#' undirected `igraph` graph. This keeps the workflow simple: pass the [nn()]
#' result for a graph on the original data space, or pass the [opentsne()] /
#' [umap()] result for a graph on the visible embedding layout.
#'
#' @param data Numeric matrix/data frame, a KNN object returned by [nn()], or a
#'   `fastEmbedR_embedding` object returned by [opentsne()] or [umap()].
#' @param knn Optional precomputed KNN object returned by [nn()]. If supplied,
#'   `data` is ignored for neighbour search.
#' @param k Number of non-self neighbours used in the graph.
#' @param backend KNN backend passed to [nn()] when `knn` is not supplied.
#'   Use `"auto"` for the fastest graph-KNN default: cuVS NN-descent on CUDA
#'   when available, then FAISS NN-descent, then RcppHNSW, then exact CPU.
#' @param weight Graph weighting. `"auto"` uses SNN/Jaccard weights for input
#'   space and distance weights for embedding space. `"snn"` builds
#'   full shared-nearest-neighbour Jaccard weights between all rows sharing at
#'   least one neighbour. `"adaptive"` uses
#'   `exp(-d_ij^2 / (sigma_i * sigma_j))`, where each `sigma` is the local
#'   neighbourhood radius. `"distance"` uses `1 / (1 + distance)`. `"binary"`
#'   gives every edge weight 1.
#' @param mutual If `TRUE`, keep only reciprocal nearest-neighbour edges. This
#'   can sharpen cluster boundaries on embedding-layout graphs.
#' @param prune Drop edges with weight less than or equal to this value.
#' @param n_threads CPU threads passed to [nn()] when KNN is computed here.
#' @return An undirected `igraph` graph with edge attribute `weight`.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' if (requireNamespace("igraph", quietly = TRUE)) {
#'   g <- knn_graph(x, k = 15, backend = "cpu")
#'   cl <- igraph::cluster_louvain(g, weights = igraph::E(g)$weight)
#'   table(igraph::membership(cl))
#' }
#' @export
knn_graph <- function(data,
                      knn = NULL,
                      k = 50L,
                      backend = "auto",
                      weight = c("auto", "snn", "adaptive", "distance", "binary"),
                      mutual = FALSE,
                      prune = 0,
                      n_threads = NULL) {
  if (!requireNamespace("igraph", quietly = TRUE)) {
    stop(
      "`knn_graph()` requires the optional package `igraph`. ",
      "Install it with `install.packages(\"igraph\")`.",
      call. = FALSE
    )
  }
  weight <- match.arg(weight)
  k <- as.integer(k)
  if (length(k) != 1L || is.na(k) || !is.finite(k) || k < 1L) {
    stop("`k` must be a positive integer.", call. = FALSE)
  }
  mutual <- isTRUE(mutual)
  prune <- suppressWarnings(as.numeric(prune))
  if (length(prune) != 1L || is.na(prune) || !is.finite(prune) || prune < 0) {
    stop("`prune` must be a non-negative number.", call. = FALSE)
  }

  input_backend <- NA_character_
  graph_space <- "input"
  input_method <- NA_character_
  if (is.null(knn)) {
    if (missing(data) || is.null(data)) {
      stop("Provide either `data` or `knn`.", call. = FALSE)
    }
    if (is_knn_input(data)) {
      knn <- data
      graph_space <- "input"
      input_method <- "nn"
    } else if (is_fastembedr_embedding(data)) {
      input_method <- as.character(data$method %||% "embedding")[1L]
      if (!is.matrix(data$layout)) {
        stop("Embedding objects passed to `knn_graph()` must contain a matrix `layout`.", call. = FALSE)
      }
      graph_space <- "embedding"
      graph_backend <- resolve_knn_graph_backend(as.character(backend)[1L])
      knn <- nn_without_self(
        data$layout,
        k = k,
        backend = graph_backend,
        n_threads = n_threads
      )
      input_backend <- attr(knn, "backend") %||% graph_backend
    } else {
      graph_backend <- resolve_knn_graph_backend(as.character(backend)[1L])
      knn <- nn_without_self(
        data,
        k = k,
        backend = graph_backend,
        n_threads = n_threads
      )
      input_backend <- attr(knn, "backend") %||% graph_backend
    }
  }

  knn_input <- coerce_knn_input(knn, arg_name = "knn")
  if (identical(weight, "auto")) {
    weight <- if (identical(graph_space, "embedding")) "distance" else "snn"
  }
  if (!is.na(knn_input$input_backend)) input_backend <- knn_input$input_backend
  if (knn_input$n_neighbors < k) {
    k <- knn_input$n_neighbors
  }
  cols <- seq_len(k)
  indices <- knn_input$indices[, cols, drop = FALSE]
  distances <- knn_input$distances[, cols, drop = FALSE]

  edges <- knn_graph_edges_cpp(indices, distances, weight, prune, mutual)
  graph <- igraph::make_empty_graph(n = edges$n_vertices, directed = FALSE)
  if (length(edges$from) > 0L) {
    graph <- igraph::add_edges(graph, as.vector(rbind(edges$from, edges$to)))
    igraph::E(graph)$weight <- edges$weight
  }
  attr(graph, "fastEmbedR_graph") <- list(
    k = as.integer(k),
    space = graph_space,
    weight = weight,
    mutual = mutual,
    prune = prune,
    nn_backend = input_backend,
    input_method = input_method,
    n_vertices = igraph::vcount(graph),
    n_edges = igraph::ecount(graph)
  )
  graph
}

is_fastembedr_embedding <- function(x) {
  inherits(x, "fastEmbedR_embedding") ||
    (is.list(x) && is.matrix(x$layout) && !is.null(x$method))
}

resolve_knn_graph_backend <- function(backend) {
  backend <- as.character(backend)[1L]
  if (is.na(backend) || !nzchar(backend)) backend <- "auto"
  if (!identical(backend, "auto")) return(backend)
  if (isTRUE(cuvs_available())) return("cuda_cuvs_nndescent")
  if (isTRUE(faiss_available())) return("faiss_nndescent")
  if (isTRUE(requireNamespace("RcppHNSW", quietly = TRUE))) return("hnsw")
  "cpu"
}
