silhouette_score <- function(labels, layout) {
  layout <- as.matrix(layout)
  labels <- as.factor(labels)
  if (length(labels) != nrow(layout)) {
    stop("`labels` must have one entry per row of `layout`.", call. = FALSE)
  }

  keep <- !is.na(labels) & apply(layout, 1L, function(row) all(is.finite(row)))
  layout <- layout[keep, , drop = FALSE]
  labels <- droplevels(labels[keep])
  n <- nrow(layout)
  if (n < 2L || length(levels(labels)) < 2L) {
    return(NA_real_)
  }

  d <- as.matrix(stats::dist(layout))
  groups <- split(seq_len(n), labels)
  values <- numeric(n)
  for (i in seq_len(n)) {
    own_name <- as.character(labels[i])
    own <- groups[[own_name]]
    same <- own[own != i]
    a <- if (length(same) == 0L) 0 else mean(d[i, same])

    other <- groups[names(groups) != own_name]
    if (length(other) == 0L) {
      values[i] <- 0
      next
    }
    b <- min(vapply(other, function(idx) mean(d[i, idx]), numeric(1)))
    denom <- max(a, b)
    values[i] <- if (denom > 0) (b - a) / denom else 0
  }
  mean(values)
}

sample_indices <- function(n, sample_size = NULL, seed = 4L) {
  if (is.null(sample_size) || sample_size >= n) {
    return(seq_len(n))
  }
  set.seed(seed)
  sort(sample.int(n, as.integer(sample_size)))
}
