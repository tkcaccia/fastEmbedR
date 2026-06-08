#' Fast landmark UMAP from precomputed nearest neighbors
#'
#' @param indices Integer KNN index matrix without the self-neighbor column.
#' @param distances Numeric KNN distance matrix matching `indices`.
#' @param n_landmarks Number of landmark/hub points to select.
#' @param landmark_ratio Landmark fraction used when `n_landmarks` is `NULL`.
#' @param landmark_k Maximum landmark neighbors retained per point.
#' @param selection Landmark selection strategy.
#' @param mode Embedding mode passed to `fast_knn_umap`.
#' @param n_epochs Number of optimization epochs.
#' @param spectral_n_iter Number of spectral power iterations.
#' @param seed Random seed.
#' @param ... Additional arguments passed to `fast_knn_umap`.
#' @return A numeric embedding matrix with an attribute `landmarks`.
#' @export
landmark_knn_umap <- function(indices,
                              distances,
                              n_landmarks = NULL,
                              landmark_ratio = 0.1,
                              landmark_k = 10L,
                              local_k = 5L,
                              selection = c("hub", "random"),
                              mode = c("hybrid", "sgd", "spectral"),
                              n_epochs = 100L,
                              spectral_n_iter = 25L,
                              seed = 4L,
                              ...) {
  selection <- match.arg(selection)
  mode <- match.arg(mode)
  graph <- landmark_knn_graph(
    indices = indices,
    distances = distances,
    n_landmarks = n_landmarks,
    landmark_ratio = landmark_ratio,
    landmark_k = landmark_k,
    local_k = local_k,
    selection = selection,
    seed = seed
  )

  layout <- fast_knn_umap(
    graph$indices,
    graph$distances,
    mode = mode,
    n_epochs = n_epochs,
    spectral_n_iter = spectral_n_iter,
    seed = seed,
    ...
  )
  attr(layout, "landmarks") <- graph$landmarks
  attr(layout, "landmark_graph") <- graph
  layout
}

#' Build a point-landmark KNN graph from a point-point KNN graph
#'
#' @inheritParams landmark_knn_umap
#' @return A list with landmark KNN matrices and selected landmark indices.
#' @export
landmark_knn_graph <- function(indices,
                               distances,
                               n_landmarks = NULL,
                               landmark_ratio = 0.1,
                               landmark_k = 10L,
                               local_k = 5L,
                               selection = c("hub", "random"),
                               seed = 4L) {
  selection <- match.arg(selection)
  indices <- as.matrix(indices)
  distances <- as.matrix(distances)
  if (!is.integer(indices)) storage.mode(indices) <- "integer"
  storage.mode(distances) <- "double"

  if (!identical(dim(indices), dim(distances))) {
    stop("`indices` and `distances` must have the same dimensions.", call. = FALSE)
  }
  n <- nrow(indices)
  k <- ncol(indices)
  if (n < 2L || k < 1L) {
    stop("`indices` must have at least two rows and one neighbor column.", call. = FALSE)
  }
  if (any(!is.finite(distances)) || any(distances < 0)) {
    stop("`distances` must contain finite non-negative values.", call. = FALSE)
  }

  min_idx <- min(indices)
  max_idx <- max(indices)
  offset <- if (min_idx >= 1L && max_idx <= n) 0L else 1L
  idx1 <- indices + offset
  if (min(idx1) < 1L || max(idx1) > n) {
    stop("KNN indices must refer to rows in the dataset.", call. = FALSE)
  }

  if (is.null(n_landmarks)) {
    n_landmarks <- ceiling(n * landmark_ratio)
  }
  n_landmarks <- max(2L, min(as.integer(n_landmarks), n))
  landmark_k <- max(1L, min(as.integer(landmark_k), k))
  local_k <- max(0L, min(as.integer(local_k), k))

  landmarks <- select_knn_landmarks(idx1, n_landmarks, selection, seed)
  is_landmark <- rep(FALSE, n)
  is_landmark[landmarks] <- TRUE

  candidates <- vector("list", n)
  candidate_distances <- vector("list", n)
  add_candidate <- function(i, landmark, distance) {
    candidates[[i]] <<- c(candidates[[i]], landmark)
    candidate_distances[[i]] <<- c(candidate_distances[[i]], distance)
  }

  for (i in seq_len(n)) {
    hits <- is_landmark[idx1[i, ]]
    if (any(hits)) {
      for (pos in which(hits)) {
        add_candidate(i, idx1[i, pos], distances[i, pos])
      }
    }
  }

  for (landmark in landmarks) {
    for (pos in seq_len(k)) {
      i <- idx1[landmark, pos]
      if (i != landmark) {
        add_candidate(i, landmark, distances[landmark, pos])
      }
    }
  }

  nearest_landmark <- rep(NA_integer_, n)
  nearest_landmark_distance <- rep(Inf, n)
  refresh_nearest <- function() {
    for (i in seq_len(n)) {
      if (length(candidates[[i]]) == 0L) next
      best <- which.min(candidate_distances[[i]])
      nearest_landmark[i] <<- candidates[[i]][best]
      nearest_landmark_distance[i] <<- candidate_distances[[i]][best]
    }
  }
  refresh_nearest()

  for (depth in seq_len(3L)) {
    empty <- which(!is.finite(nearest_landmark_distance))
    if (length(empty) == 0L) break
    for (i in empty) {
      for (pos in seq_len(k)) {
        nb <- idx1[i, pos]
        if (is.finite(nearest_landmark_distance[nb])) {
          add_candidate(i, nearest_landmark[nb], distances[i, pos] + nearest_landmark_distance[nb])
        }
      }
    }
    refresh_nearest()
  }

  fallback_distance <- stats::median(distances, na.rm = TRUE)
  if (!is.finite(fallback_distance) || fallback_distance <= 0) {
    fallback_distance <- max(distances, na.rm = TRUE)
  }
  if (!is.finite(fallback_distance) || fallback_distance <= 0) {
    fallback_distance <- 1
  }
  empty <- which(!is.finite(nearest_landmark_distance))
  if (length(empty) > 0L) {
    for (i in empty) {
      add_candidate(i, landmarks[((i - 1L) %% length(landmarks)) + 1L], fallback_distance)
    }
  }

  out_k <- landmark_k + local_k
  out_idx <- matrix(landmarks[1L], n, out_k)
  out_dist <- matrix(fallback_distance, n, out_k)
  for (i in seq_len(n)) {
    cand <- candidates[[i]]
    dst <- candidate_distances[[i]]
    keep <- is.finite(dst) & cand >= 1L & cand <= n & cand != i
    cand <- cand[keep]
    dst <- dst[keep]
    if (length(cand) == 0L) {
      cand <- landmarks[landmarks != i]
      if (length(cand) == 0L) cand <- landmarks
      cand <- cand[1L]
      dst <- fallback_distance
    }
    best_by_landmark <- tapply(seq_along(dst), cand, function(ii) ii[which.min(dst[ii])])
    keep_idx <- as.integer(best_by_landmark)
    cand <- cand[keep_idx]
    dst <- dst[keep_idx]
    ord <- order(dst, cand)
    cand <- cand[ord]
    dst <- dst[ord]
    take <- min(landmark_k, length(cand))
    out_idx[i, seq_len(take)] <- cand[seq_len(take)]
    out_dist[i, seq_len(take)] <- dst[seq_len(take)]
    if (take < landmark_k) {
      out_idx[i, (take + 1L):landmark_k] <- cand[take]
      out_dist[i, (take + 1L):landmark_k] <- dst[take]
    }
    if (local_k > 0L) {
      cols <- landmark_k + seq_len(local_k)
      out_idx[i, cols] <- idx1[i, seq_len(local_k)]
      out_dist[i, cols] <- distances[i, seq_len(local_k)]
    }
  }

  list(
    indices = out_idx - offset,
    distances = out_dist,
    landmarks = landmarks - offset,
    selection = selection,
    landmark_k = landmark_k,
    local_k = local_k
  )
}

select_knn_landmarks <- function(indices, n_landmarks, selection, seed) {
  n <- nrow(indices)
  if (selection == "random") {
    set.seed(seed)
    return(sort(sample.int(n, n_landmarks)))
  }

  frequency <- tabulate(as.vector(indices), nbins = n)
  order_idx <- order(frequency, decreasing = TRUE)
  selected <- integer(0)
  selected_flag <- rep(FALSE, n)
  covered <- rep(FALSE, n)

  for (candidate in order_idx) {
    if (length(selected) >= n_landmarks) break
    if (covered[candidate]) next
    selected <- c(selected, candidate)
    selected_flag[candidate] <- TRUE
    covered[candidate] <- TRUE
    covered[indices[candidate, ]] <- TRUE
  }

  if (length(selected) < n_landmarks) {
    fill <- order_idx[!selected_flag[order_idx]]
    selected <- c(selected, fill[seq_len(n_landmarks - length(selected))])
  }

  sort(selected)
}
