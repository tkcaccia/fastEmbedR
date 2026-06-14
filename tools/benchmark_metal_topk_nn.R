library(fastEmbedR)

set.seed(41)
options(fastEmbedR.cpu_nndescent_prefer_faiss = FALSE)
options(fastEmbedR.gpu_approx_recall = FALSE)
options(fastEmbedR.metal_exact_auto_threshold = 0L)

out_dir <- file.path(
  getwd(),
  "results",
  paste0("metal_topk_nn_", format(Sys.time(), "%Y%m%d_%H%M%S"))
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

methods <- c(
  "metal_nndescent",
  "metal",
  "faiss_flat",
  "faiss_hnsw",
  "cpu_nndescent"
)
ps <- c(50L, 100L, 300L, 784L)
ks <- c(15L, 50L, 100L)
n <- as.integer(Sys.getenv("FASTEMBEDR_BENCH_N", "5000"))

make_x <- function(n, p) {
  cls <- rep(seq_len(10L), length.out = n)
  centers <- matrix(rnorm(10L * p, sd = 2.5), 10L, p)
  x <- centers[cls, , drop = FALSE] + matrix(rnorm(n * p), n, p)
  storage.mode(x) <- "double"
  x
}

time_one <- function(x, k, method) {
  gc()
  ok <- TRUE
  err <- NA_character_
  attrs <- list()
  elapsed <- NA_real_
  tryCatch({
    tm <- system.time({
      ans <- nn(x, k = k, backend = method, n_threads = 4L)
    })
    elapsed <- unname(tm[["elapsed"]])
    attrs <- attributes(ans)
  }, error = function(e) {
    ok <<- FALSE
    err <<- conditionMessage(e)
  })

  data.frame(
    method = method,
    n = nrow(x),
    p = ncol(x),
    k = k,
    sec = elapsed,
    status = if (ok) "success" else "failed",
    error = ifelse(is.na(err), "", err),
    backend_attr = if (!is.null(attrs$backend)) as.character(attrs$backend) else "",
    metal_kernel = if (!is.null(attrs$metal_kernel)) as.character(attrs$metal_kernel) else "",
    exact = if (!is.null(attrs$exact)) as.character(attrs$exact) else "",
    stringsAsFactors = FALSE
  )
}

rows <- list()
i <- 0L
for (p in ps) {
  x <- make_x(n, p)
  for (k in ks) {
    for (method in methods) {
      i <- i + 1L
      cat(sprintf("[%03d] method=%s n=%d p=%d k=%d\n", i, method, n, p, k))
      rows[[i]] <- time_one(x, k, method)
      print(rows[[i]][, c("method", "p", "k", "sec", "status", "backend_attr", "metal_kernel")])
      write.csv(do.call(rbind, rows), file.path(out_dir, "nn_speed_grid_partial.csv"), row.names = FALSE)
    }
  }
}

res <- do.call(rbind, rows)
write.csv(res, file.path(out_dir, "nn_speed_grid.csv"), row.names = FALSE)
cat("RESULT_DIR=", out_dir, "\n", sep = "")
print(res)
