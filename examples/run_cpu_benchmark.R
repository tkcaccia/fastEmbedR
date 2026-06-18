library(fastEmbedR)

results_dir <- file.path("results", "cpu_example")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

make_synthetic <- function(n = 500L, p = 12L, groups = 4L, seed = 44L) {
  set.seed(seed)
  labels <- rep(seq_len(groups), length.out = n)
  centers <- matrix(stats::rnorm(groups * p, sd = 3), groups, p)
  x <- matrix(0, n, p)
  for (g in seq_len(groups)) {
    idx <- which(labels == g)
    x[idx, ] <- matrix(stats::rnorm(length(idx) * p, sd = 0.7), length(idx), p) +
      matrix(rep(centers[g, ], each = length(idx)), length(idx), p)
  }
  list(x = x, y = factor(labels))
}

datasets <- list(
  iris = list(x = as.matrix(iris[, 1:4]), y = iris$Species),
  synthetic_500 = make_synthetic()
)

run_fit <- function(dataset_name, method, seed) {
  item <- datasets[[dataset_name]]
  fit <- switch(
    method,
    umap = umap(
      item$x,
      n_neighbors = 15L,
      seed = seed,
      backend = "cpu"
    ),
    opentsne = opentsne(
      item$x,
      perplexity = 15,
      seed = seed,
      backend = "cpu"
    ),
    stop("unknown method")
  )
  quality <- evaluate_embedding(
    item$x,
    fit$layout,
    labels = item$y,
    k = 15L,
    sample_size_for_global_metrics = min(1000L, nrow(item$x)),
    sample_size_for_local_metrics = min(1000L, nrow(item$x)),
    n_threads = 2L
  )
  data.frame(
    dataset = dataset_name,
    method = method,
    seed = seed,
    n = nrow(item$x),
    p = ncol(item$x),
    elapsed_sec = fit$metrics$elapsed,
    silhouette = quality$silhouette,
    knn_preservation = quality$knn_preservation,
    stringsAsFactors = FALSE
  )
}

rows <- do.call(rbind, lapply(names(datasets), function(dataset_name) {
  do.call(rbind, lapply(c("umap", "opentsne"), function(method) {
    do.call(rbind, lapply(c(4L, 5L), function(seed) {
      run_fit(dataset_name, method, seed)
    }))
  }))
}))

csv_path <- file.path(results_dir, "core_cpu_embeddings.csv")
utils::write.csv(rows, csv_path, row.names = FALSE)
print(rows, row.names = FALSE)
cat("Wrote:", csv_path, "\n")
