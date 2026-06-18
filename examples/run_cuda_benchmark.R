library(fastEmbedR)

if (!cuda_available()) {
  message("CUDA native backend is not available for this build/runtime.")
} else {
  x <- as.matrix(iris[, 1:4])
  labels <- iris$Species
  fit <- umap(
    x,
    n_neighbors = 15L,
    backend = "cuda"
  )
  print(fit$metrics)
}
