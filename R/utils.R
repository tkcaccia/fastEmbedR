cache_file <- function(cache_dir, prefix, dataset, n, p, key) {
  safe_dataset <- gsub("[^A-Za-z0-9_.-]+", "_", dataset)
  file.path(cache_dir, sprintf("%s_%s_n%s_p%s_%s.rds", prefix, safe_dataset, n, p, key))
}
