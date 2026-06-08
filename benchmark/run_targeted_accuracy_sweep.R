Sys.setenv(LC_ALL="C", LANG="C")
library(fastknnumap)

score_layout <- function(layout, labels, keep) {
  labels <- as.integer(as.factor(labels))
  mean(cluster::silhouette(labels[keep], stats::dist(layout[keep, , drop = FALSE]))[, "sil_width"])
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

run_one <- function(dataset, x, labels, nn, method, k, n_epochs, min_dist, neg, lr, keep, n_threads = 8) {
  nnk <- trunc_nn(nn, k)
  idx <- nnk$indices[, -1, drop=FALSE]
  dst <- nnk$distances[, -1, drop=FALSE]
  t <- system.time({
    layout <- switch(method,
      uwot_fast_sgd = uwot::umap(X=NULL, n_neighbors=k, nn_method=list(idx=nnk$indices, dist=nnk$distances), n_epochs=n_epochs, min_dist=min_dist, negative_sample_rate=neg, init="spectral", fast_sgd=TRUE, seed=4, verbose=FALSE),
      native = fast_knn_umap(idx, dst, mode="hybrid", n_epochs=n_epochs, min_dist=min_dist, negative_sample_rate=neg, learning_rate=lr, spectral_n_iter=25, seed=4),
      native_random = fast_knn_umap(idx, dst, mode="hybrid", init="random", n_epochs=n_epochs, min_dist=min_dist, negative_sample_rate=neg, learning_rate=lr, spectral_n_iter=25, seed=4),
      native_landmark = landmark_knn_umap(idx, dst, landmark_ratio=0.1, landmark_k=10, local_k=10, mode="hybrid", n_epochs=n_epochs, min_dist=min_dist, negative_sample_rate=neg, learning_rate=lr, spectral_n_iter=25, seed=4),
      knn_tsne = knn_tsne(idx, dst, n_epochs=n_epochs, negative_sample_rate=neg, n_threads=n_threads, seed=4),
      stop("unknown method")
    )
  })
  data.frame(dataset=dataset, method=method, k=k, n_epochs=n_epochs, min_dist=min_dist, neg=neg, lr=lr, elapsed=t[["elapsed"]], silhouette=score_layout(layout, labels, keep), knn_preservation=preserve_score(layout, idx, keep), stringsAsFactors=FALSE)
}

run_dataset <- function(dataset, x, labels, max_k=100, sample_n=3000, out_file) {
  set.seed(4)
  keep <- sort(sample.int(nrow(x), min(sample_n, nrow(x))))
  message("KNN ", dataset)
  tknn <- system.time(nn <- Rnanoflann::nn(x, x, max_k))
  rows <- list(data.frame(dataset=dataset, method="KNN", k=max_k, n_epochs=NA, min_dist=NA, neg=NA, lr=NA, elapsed=tknn[["elapsed"]], silhouette=NA, knn_preservation=NA))
  saveRDS(list(results=do.call(rbind, rows), nn=nn), out_file)

  # Accuracy target: uwot at useful k values.
  params <- rbind(
    expand.grid(method="uwot_fast_sgd", k=c(15,30,50,100), n_epochs=200, min_dist=c(0.01,0.1), neg=5, lr=NA, stringsAsFactors=FALSE),
    expand.grid(method="native", k=c(30,50,100), n_epochs=c(200,500,800), min_dist=c(0.001,0.01,0.1), neg=c(5,10), lr=c(1.0,1.5), stringsAsFactors=FALSE),
    expand.grid(method="native_landmark", k=c(50,100), n_epochs=c(200,500), min_dist=c(0.01,0.1), neg=5, lr=1.0, stringsAsFactors=FALSE),
    expand.grid(method="knn_tsne", k=c(30,100), n_epochs=c(100,300), min_dist=NA, neg=5, lr=NA, stringsAsFactors=FALSE)
  )
  for (i in seq_len(nrow(params))) {
    p <- params[i,]
    r <- tryCatch(run_one(dataset, x, labels, nn, p$method, p$k, p$n_epochs, ifelse(is.na(p$min_dist), 0.01, p$min_dist), p$neg, ifelse(is.na(p$lr), 1.0, p$lr), keep), error=function(e) data.frame(dataset=dataset, method=p$method, k=p$k, n_epochs=p$n_epochs, min_dist=p$min_dist, neg=p$neg, lr=p$lr, elapsed=NA, silhouette=NA, knn_preservation=NA, error=conditionMessage(e)))
    print(r)
    rows[[length(rows)+1]] <- r
    saveRDS(list(results=rbind.fill(lapply(rows, function(z) { if (!"error" %in% names(z)) z <- ""; z })), nn=nn), out_file)
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
