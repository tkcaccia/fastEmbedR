Sys.setenv(LC_ALL="C", LANG="C")
library(fastknnumap)

score_layout <- function(layout, labels, keep) {
  fastknnumap:::silhouette_score(labels[keep], layout[keep, , drop = FALSE])
}

preserve_score <- function(layout, indices, keep, preserve_k = min(29, ncol(indices))) {
  n <- nrow(layout)
  out <- numeric(length(keep))
  for (ii in seq_along(keep)) {
    i <- keep[ii]
    d <- rowSums((layout - matrix(layout[i, ], n, ncol(layout), byrow = TRUE))^2)
    d[i] <- Inf
    out[ii] <- sum(order(d)[seq_len(preserve_k)] %in% indices[i, seq_len(preserve_k)]) / preserve_k
  }
  mean(out)
}

trunc_nn <- function(nn, k) list(indices = nn$indices[, seq_len(k), drop=FALSE], distances = nn$distances[, seq_len(k), drop=FALSE])

run_one <- function(dataset, x, labels, nn, method, config_id, k, n_epochs, min_dist, neg, lr, keep, n_threads = 8) {
  nnk <- trunc_nn(nn, k)
  idx <- nnk$indices[, -1, drop=FALSE]
  dst <- nnk$distances[, -1, drop=FALSE]
  t <- system.time({
    layout <- switch(method,
      native = fast_knn_umap(idx, dst, mode="hybrid", n_epochs=n_epochs, min_dist=min_dist, negative_sample_rate=neg, learning_rate=lr, spectral_n_iter=25, seed=4),
      native_sgd = fast_knn_umap(idx, dst, mode="sgd", n_epochs=n_epochs, min_dist=min_dist, negative_sample_rate=neg, learning_rate=lr, spectral_n_iter=25, seed=4),
      native_spectral = fast_knn_umap(idx, dst, mode="spectral", min_dist=min_dist, negative_sample_rate=neg, learning_rate=lr, spectral_n_iter=25, seed=4),
      native_random = fast_knn_umap(idx, dst, mode="hybrid", init="random", n_epochs=n_epochs, min_dist=min_dist, negative_sample_rate=neg, learning_rate=lr, spectral_n_iter=25, seed=4),
      native_landmark = landmark_knn_umap(idx, dst, landmark_ratio=0.1, landmark_k=10, local_k=10, mode="hybrid", n_epochs=n_epochs, min_dist=min_dist, negative_sample_rate=neg, learning_rate=lr, spectral_n_iter=25, seed=4),
      knn_tsne = knn_tsne(idx, dst, n_epochs=n_epochs, negative_sample_rate=neg, learning_rate=lr, n_threads=n_threads, seed=4),
      knn_pacmap = knn_pacmap(idx, dst, n_epochs=n_epochs, negative_sample_rate=neg, learning_rate=lr, n_threads=n_threads, seed=4),
      knn_trimap = knn_trimap(idx, dst, n_epochs=n_epochs, negative_sample_rate=neg, learning_rate=lr, n_threads=n_threads, seed=4),
      knn_localmap = knn_localmap(idx, dst, n_epochs=n_epochs, negative_sample_rate=neg, learning_rate=lr, n_threads=n_threads, seed=4),
      stop("unknown method")
    )
  })
  data.frame(dataset=dataset, config_id=config_id, method=method, k=k, n_epochs=n_epochs, min_dist=min_dist, neg=neg, lr=lr, elapsed=t[["elapsed"]], silhouette=score_layout(layout, labels, keep), knn_preservation=preserve_score(layout, idx, keep), error="", stringsAsFactors=FALSE)
}

run_dataset <- function(dataset, x, labels, max_k=100, sample_n=3000, out_file) {
  set.seed(4)
  keep <- sort(sample.int(nrow(x), min(sample_n, nrow(x))))
  message("KNN ", dataset)
  tknn <- system.time(nn <- fastknnumap::nn(x, x, max_k, parallel = TRUE))
  rows <- list(data.frame(dataset=dataset, config_id=0L, method="KNN", k=max_k, n_epochs=NA, min_dist=NA, neg=NA, lr=NA, elapsed=tknn[["elapsed"]], silhouette=NA, knn_preservation=NA, error=""))
  saveRDS(list(results=do.call(rbind, rows), nn=nn), out_file)

  # Every method is run for every parameter row. Do not compare methods across
  # different k, epoch, min_dist, negative-sampling, or learning-rate settings.
  methods <- c("native", "native_sgd", "native_random", "native_landmark", "knn_tsne", "knn_pacmap", "knn_trimap", "knn_localmap")
  params <- expand.grid(
    k = c(30, 50, 100),
    n_epochs = c(200, 500),
    min_dist = c(0.01, 0.1),
    neg = c(5, 10),
    lr = c(1.0, 1.5),
    stringsAsFactors = FALSE
  )
  params$config_id <- seq_len(nrow(params))
  for (i in seq_len(nrow(params))) {
    p <- params[i, ]
    for (method in methods) {
      r <- tryCatch(
        run_one(dataset, x, labels, nn, method, p$config_id, p$k, p$n_epochs, p$min_dist, p$neg, p$lr, keep),
        error=function(e) data.frame(dataset=dataset, config_id=p$config_id, method=method, k=p$k, n_epochs=p$n_epochs, min_dist=p$min_dist, neg=p$neg, lr=p$lr, elapsed=NA, silhouette=NA, knn_preservation=NA, error=conditionMessage(e))
      )
      print(r)
      rows[[length(rows)+1]] <- r
      saveRDS(list(results=rbind.fill(lapply(rows, function(z) { if (!"error" %in% names(z)) z$error <- ""; z })), nn=nn), out_file)
    }
  }
  invisible(do.call(rbind.fill, rows))
}

# local rbind.fill without extra packages
rbind.fill <- function(xs) {
  xs <- lapply(xs, function(x) {
    if (!is.data.frame(x)) x <- as.data.frame(x, stringsAsFactors = FALSE)
    x
  })
  cols <- unique(unlist(lapply(xs, names)))
  xs <- lapply(xs, function(x) {
    for (cc in setdiff(cols, names(x))) x[[cc]] <- NA
    x[, cols, drop = FALSE]
  })
  do.call(rbind, xs)
}

# Fashion-MNIST 5k main
fm <- load_fashion_mnist(n_train=5000, seed=4)
pc <- stats::prcomp(fm$x, center=TRUE, scale.=FALSE, rank.=50)
run_dataset("fashion_mnist_5k_pca50", pc$x, fm$y, out_file="benchmark/fashion_mnist_5k_targeted_sweep.rds")

# Metref check
load("/Users/stefano/Documents/GPUPLS/Data/metref_remote_task.RData")
run_dataset("metref", scale(out$Xtrain), out$Ytrain, sample_n=nrow(out$Xtrain), out_file="benchmark/metref_targeted_sweep.rds")

# Single-cell 5k stratified
load("/Users/stefano/Documents/GPUPLS/Data/singlecell.RData")
set.seed(4)
f <- as.factor(labels)
per <- ceiling(5000 / nlevels(f))
keep <- unlist(lapply(split(seq_along(f), f), function(ii) sample(ii, min(length(ii), per))))
if (length(keep) > 5000) keep <- sample(keep, 5000)
keep <- sort(keep)
run_dataset("singlecell_5k_first20", as.matrix(data[keep,1:20]), labels[keep], out_file="benchmark/singlecell_5k_targeted_sweep.rds")
