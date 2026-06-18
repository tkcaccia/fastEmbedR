library(fastEmbedR)

if (!metal_available()) {
  message("Metal native backend is not available for this build/runtime.")
} else {
  x <- as.matrix(iris[, 1:4])
  labels <- iris$Species
  fit <- umap(
    x,
    n_neighbors = 15L,
    backend = "metal"
  )
  print(fit$metrics)
}
