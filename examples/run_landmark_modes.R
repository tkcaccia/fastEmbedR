library(fastEmbedR)

results_dir <- file.path("results", "landmark_modes")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

make_synthetic <- function(n = 1500L, p = 20L, groups = 5L, seed = 101L) {
  set.seed(seed)
  labels <- rep(seq_len(groups), length.out = n)
  centers <- matrix(stats::rnorm(groups * p, sd = 4), groups, p)
  x <- matrix(0, n, p)
  for (g in seq_len(groups)) {
    idx <- which(labels == g)
    x[idx, ] <- matrix(stats::rnorm(length(idx) * p, sd = 0.7), length(idx), p) +
      matrix(rep(centers[g, ], each = length(idx)), length(idx), p)
  }
  list(name = "synthetic", x = x, y = factor(labels))
}

dataset <- make_synthetic()
x <- dataset$x
labels <- dataset$y
n_neighbors <- 15L
seed <- 13L

run_one <- function(name, ...) {
  gc(FALSE)
  fit <- if (identical(name, "full")) {
    umap(
      x,
      n_neighbors = n_neighbors,
      seed = seed,
      backend = "cpu"
    )
  } else {
    landmark_umap(
      x,
      n_neighbors = n_neighbors,
      seed = seed,
      backend = "cpu",
      ...
    )
  }
  quality <- evaluate_embedding(
    x,
    fit$layout,
    labels = labels,
    k = n_neighbors,
    sample_size_for_global_metrics = min(1000L, nrow(x)),
    sample_size_for_local_metrics = min(1000L, nrow(x)),
    n_threads = 2L
  )
  row <- data.frame(
    dataset = dataset$name,
    n = nrow(x),
    p = ncol(x),
    mode = name,
    elapsed_sec = fit$metrics$elapsed,
    n_landmarks = if ("n_landmarks" %in% names(fit$metrics)) fit$metrics$n_landmarks else nrow(x),
    silhouette = quality$silhouette,
    knn_preservation = quality$knn_preservation,
    stringsAsFactors = FALSE
  )
  list(fit = fit, row = row)
}

runs <- list(
  full = run_one("full"),
  landmark_50 = run_one("landmark_50", landmarks = 0.5),
  landmark_25 = run_one("landmark_25", landmarks = 0.25)
)

metrics <- do.call(rbind, lapply(runs, `[[`, "row"))
metrics$speedup_vs_full <- metrics$elapsed_sec[metrics$mode == "full"][1] / metrics$elapsed_sec

csv_path <- file.path(results_dir, paste0(dataset$name, "_landmark_modes.csv"))
utils::write.csv(metrics, csv_path, row.names = FALSE)
print(metrics, row.names = FALSE)
cat("Wrote metrics:", csv_path, "\n")

png_path <- file.path(results_dir, paste0(dataset$name, "_landmark_modes.png"))
grDevices::png(png_path, width = 1200, height = 1000, res = 140)
old_par <- graphics::par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
on.exit(graphics::par(old_par), add = TRUE)
for (name in names(runs)) {
  fit <- runs[[name]]$fit
  title <- paste0(
    name,
    "\n",
    round(fit$metrics$elapsed, 3), " sec"
  )
  plot(fit, main = title)
}
grDevices::dev.off()
cat("Wrote plot:", png_path, "\n")
