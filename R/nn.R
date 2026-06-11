nn_compute <- function(data,
                       points,
                       k,
                       backend,
                       points_missing,
                       exclude_self = FALSE,
                       n_threads = NULL) {
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  if (isTRUE(points_missing)) {
    points <- data
  } else {
    points <- as.matrix(points)
    storage.mode(points) <- "double"
  }

  if (!identical(ncol(data), ncol(points))) {
    stop("`data` and `points` must have the same number of columns.", call. = FALSE)
  }
  if (nrow(data) < 1L || nrow(points) < 1L) {
    stop("`data` and `points` must have at least one row.", call. = FALSE)
  }
  self_query <- isTRUE(points_missing) || (
    nrow(data) == nrow(points) &&
      ncol(data) == ncol(points) &&
      identical(data, points)
  )
  if (isTRUE(exclude_self) && !isTRUE(self_query)) {
    stop("Self-neighbor exclusion is only valid when `points` is `data`.", call. = FALSE)
  }
  if (is.null(k)) {
    k <- if (nrow(data) == 1L) {
      1L
    } else {
      min(
        nrow(data),
        auto_k(nrow(data), include_self = isTRUE(self_query) && !isTRUE(exclude_self))
      )
    }
  }
  k <- as.integer(k)
  if (length(k) != 1L || is.na(k) || !is.finite(k) || k < 1L) {
    stop("`k` must be NULL or a positive integer.", call. = FALSE)
  }
  max_k <- if (isTRUE(exclude_self)) nrow(data) - 1L else nrow(data)
  if (k > max_k) {
    stop("`k` cannot be larger than the available neighbor count.", call. = FALSE)
  }
  finite_input <- if (isTRUE(points_missing)) {
    all(is.finite(data))
  } else {
    all(is.finite(data)) && all(is.finite(points))
  }
  if (!isTRUE(finite_input)) {
    stop("`data` and `points` must contain only finite values.", call. = FALSE)
  }
  n_threads <- normalize_nn_threads(n_threads)

  if (backend %in% c("cuda_nndescent", "cuda_approx")) {
    backend <- "cuda_cuvs_nndescent"
  }
  if (backend %in% c("gpu_nndescent", "gpu_approx") &&
      identical(available_native_gpu_backend(need_knn = TRUE), "cuda")) {
    backend <- "cuda_cuvs_nndescent"
  }

  work_size <- as.double(nrow(data)) * as.double(nrow(points)) * as.double(ncol(data))

  auto_gpu <- resolve_auto_knn_gpu_backend(
    backend = backend,
    self_query = self_query,
    k = k,
    work_size = work_size
  )
  if (!is.na(auto_gpu)) {
    backend <- auto_gpu
  } else if (identical(backend, "auto") &&
             should_use_auto_cpu_approx_self_knn(
               self_query = self_query,
               n = nrow(data),
               p = ncol(data),
               k = k,
               work_size = work_size
             )) {
    backend <- select_cpu_approx_backend(nrow(data), ncol(data), k)
  } else if (identical(backend, "cpu_approx")) {
    if (!isTRUE(self_query)) {
      stop("`backend = \"cpu_approx\"` is only available for self-KNN searches.", call. = FALSE)
    }
    backend <- select_cpu_approx_backend(nrow(data), ncol(data), k)
  }

  gpu_nndescent <- resolve_gpu_nndescent_backend(
    backend = backend,
    self_query = self_query,
    n = nrow(data),
    p = ncol(data),
    k = k,
    exclude_self = isTRUE(exclude_self),
    work_size = work_size
  )
  if (!is.na(gpu_nndescent)) {
    selected_gpu <- sub("_nndescent$", "", gpu_nndescent)
    nonself_k <- if (isTRUE(exclude_self)) k else max(0L, k - 1L)
    if (nonself_k < 1L) {
      out <- list(
        indices = matrix(seq_len(nrow(data)), nrow(data), 1L),
        distances = matrix(0, nrow(data), 1L)
      )
    } else {
      out <- gpu_nndescent_self_knn(
        data,
        k = nonself_k,
        backend = selected_gpu,
        seed = fast_knn_approx_seed()
      )
      if (!isTRUE(exclude_self)) {
        out$indices <- cbind(seq_len(nrow(data)), out$indices)
        out$distances <- cbind(rep(0, nrow(data)), out$distances)
      }
    }
    result <- finish_nn_result(out, gpu_nndescent, k, self_query, exact = FALSE)
    attr(result, "approximation") <- attr(out, "approximation")
    attr(result, "metal_kernel") <- attr(out, "metal_kernel")
    attr(result, "cuda_kernel") <- attr(out, "cuda_kernel")
    if (isTRUE(getOption("fastEmbedR.gpu_approx_recall", FALSE))) {
      result <- attach_knn_recall_subset(
        result,
        data = data,
        k = k,
        exclude_self = isTRUE(exclude_self),
        seed = fast_knn_approx_seed()
      )
    }
    return(result)
  }

  gpu_ivf <- resolve_gpu_ivf_backend(
    backend = backend,
    self_query = self_query,
    n = nrow(data),
    p = ncol(data),
    k = k,
    exclude_self = isTRUE(exclude_self)
  )
  if (!is.na(gpu_ivf$backend)) {
    return(gpu_ivf_self_knn(
      data,
      k = k,
      backend = gpu_ivf$backend,
      label = gpu_ivf$label,
      strategy = gpu_ivf$strategy,
      exclude_self = isTRUE(exclude_self),
      seed = fast_knn_approx_seed()
    ))
  }

  if (identical(backend, "metal_grid")) {
    if (!isTRUE(self_query)) {
      stop("`backend = \"metal_grid\"` is only available for self-KNN searches.", call. = FALSE)
    }
    if (!isTRUE(metal_available())) {
      stop("No Metal GPU backend is available on this machine.", call. = FALSE)
    }
    nonself_k <- if (isTRUE(exclude_self)) k else max(0L, k - 1L)
    if (nonself_k < 1L) {
      out <- list(
        indices = matrix(seq_len(nrow(data)), nrow(data), 1L),
        distances = matrix(0, nrow(data), 1L)
      )
    } else {
      params <- metal_grid_params(nrow(data), ncol(data), nonself_k)
      out <- grid_knn_metal_cpp(
        data,
        as.integer(nonself_k),
        as.integer(params$grid_dims),
        as.integer(params$bins_per_dim),
        as.integer(params$radius)
      )
      if (!isTRUE(exclude_self)) {
        out$indices <- cbind(seq_len(nrow(data)), out$indices)
        out$distances <- cbind(rep(0, nrow(data)), out$distances)
      }
    }
    result <- finish_nn_result(out, "metal_grid", k, self_query, exact = FALSE)
    attr(result, "approximation") <- list(
      strategy = "fastgraph_style_grid_candidate_native",
      backend = "metal",
      grid_dims = as.integer(attr(out, "grid_dims")),
      bins_per_dim = as.integer(attr(out, "grid_bins")),
      radius = as.integer(attr(out, "grid_radius")),
      total_bins = as.numeric(attr(out, "grid_total_bins")),
      max_cells = as.numeric(attr(out, "grid_max_cells"))
    )
    if (isTRUE(getOption("fastEmbedR.gpu_approx_recall", FALSE))) {
      result <- attach_knn_recall_subset(
        result,
        data = data,
        k = k,
        exclude_self = isTRUE(exclude_self),
        seed = fast_knn_approx_seed()
      )
    }
    return(result)
  }

  approx_gpu <- resolve_gpu_approx_backend(
    backend = backend,
    self_query = self_query,
    n = nrow(data),
    p = ncol(data),
    k = k,
    exclude_self = isTRUE(exclude_self),
    work_size = work_size
  )
  if (!is.na(approx_gpu)) {
    return(gpu_approx_self_knn(
      data,
      k = k,
      backend = approx_gpu,
      exclude_self = isTRUE(exclude_self),
      seed = fast_knn_approx_seed()
    ))
  }

  if (backend %in% c("faiss", "cpu_faiss", "cpu_faiss_flat")) {
    if (!isTRUE(faiss_available())) {
      stop(
        "The real FAISS C++ backend is not available in this build. ",
        "Reinstall with `FASTEMBEDR_USE_FAISS=1` and `FAISS_HOME` pointing ",
        "to a FAISS installation.",
        call. = FALSE
      )
    }
    out <- nn_faiss_flat_cpp(
      data,
      points,
      as.integer(k),
      isTRUE(exclude_self),
      as.integer(n_threads)
    )
    result <- finish_nn_result(out, "faiss", k, self_query, exact = TRUE)
    attr(result, "faiss") <- list(
      index_type = as.character(out$index_type),
      library = "faiss",
      backend = "cpu"
    )
    return(result)
  }

  if (backend %in% c("faiss_ivf", "cpu_faiss_index_ivf")) {
    if (!isTRUE(faiss_available())) {
      stop(
        "The real FAISS C++ IVF backend is not available in this build. ",
        "Reinstall with `FASTEMBEDR_USE_FAISS=1` and `FAISS_HOME` pointing ",
        "to a FAISS installation.",
        call. = FALSE
      )
    }
    params <- faiss_ivf_params(nrow(data), k)
    out <- nn_faiss_ivf_cpp(
      data,
      points,
      as.integer(k),
      as.integer(params$nlist),
      as.integer(params$nprobe),
      isTRUE(exclude_self),
      as.integer(n_threads)
    )
    result <- finish_nn_result(out, "faiss_ivf", k, self_query, exact = FALSE)
    attr(result, "approximation") <- list(
      strategy = "faiss_IndexIVFFlat",
      backend = "faiss_ivf",
      library = "faiss",
      nlist = as.integer(out$nlist),
      nprobe = as.integer(out$nprobe)
    )
    return(result)
  }

  if (backend %in% c("cuvs", "gpu_cuvs", "cuda_cuvs")) {
    require_cuvs_backend("cuVS")
    if (cuvs_should_use_nndescent(self_query, nrow(data))) {
      backend <- "cuda_cuvs_nndescent"
    } else {
      backend <- "cuda_cuvs_bruteforce"
    }
  }

  if (identical(backend, "cuda_cuvs_cagra")) {
    require_cuvs_backend("cuVS CAGRA")
    params <- cuvs_cagra_params(nrow(data), k)
    out <- nn_cuvs_cagra_cpp(
      data,
      points,
      as.integer(k),
      isTRUE(exclude_self),
      as.integer(params$graph_degree),
      as.integer(params$intermediate_graph_degree),
      as.integer(params$search_width),
      as.integer(params$itopk_size)
    )
    result <- finish_nn_result(out, "cuda_cuvs_cagra", k, self_query, exact = FALSE)
    attr(result, "approximation") <- list(
      strategy = "rapids_cuvs_cagra",
      backend = "cuda_cuvs_cagra",
      library = "cuvs",
      graph_degree = as.integer(out$graph_degree),
      intermediate_graph_degree = as.integer(out$intermediate_graph_degree),
      search_width = as.integer(out$search_width),
      itopk_size = as.integer(out$itopk_size)
    )
    return(result)
  }

  if (backend %in% c("cuvs_bruteforce", "cuda_cuvs_bruteforce", "cuda_cuvs_exact")) {
    require_cuvs_backend("cuVS brute-force")
    out <- nn_cuvs_bruteforce_cpp(
      data,
      points,
      as.integer(k),
      isTRUE(exclude_self)
    )
    result <- finish_nn_result(out, "cuda_cuvs_bruteforce", k, self_query, exact = TRUE)
    attr(result, "cuvs") <- list(
      index_type = as.character(out$index_type),
      library = "cuvs",
      backend = "cuda"
    )
    return(result)
  }

  if (backend %in% c("cuvs_nndescent", "cuda_cuvs_nndescent")) {
    require_cuvs_backend("cuVS NN-descent")
    if (!isTRUE(self_query)) {
      stop("`backend = \"cuda_cuvs_nndescent\"` is only available for self-KNN searches.", call. = FALSE)
    }
    nonself_k <- if (isTRUE(exclude_self)) k else max(0L, k - 1L)
    if (nonself_k < 1L) {
      out <- list(
        indices = matrix(seq_len(nrow(data)), nrow(data), 1L),
        distances = matrix(0, nrow(data), 1L)
      )
    } else {
      params <- cuvs_nndescent_params(nrow(data), nonself_k)
      out <- nn_cuvs_nndescent_self_cpp(
        data,
        as.integer(nonself_k),
        as.integer(params$graph_degree),
        as.integer(params$intermediate_graph_degree),
        as.integer(params$max_iterations)
      )
      if (!isTRUE(exclude_self)) {
        out$indices <- cbind(seq_len(nrow(data)), out$indices)
        out$distances <- cbind(rep(0, nrow(data)), out$distances)
      }
    }
    result <- finish_nn_result(out, "cuda_cuvs_nndescent", k, self_query, exact = FALSE)
    attr(result, "approximation") <- list(
      strategy = "rapids_cuvs_nndescent",
      backend = "cuda_cuvs_nndescent",
      library = "cuvs",
      graph_degree = as.integer(out$graph_degree),
      intermediate_graph_degree = as.integer(out$intermediate_graph_degree),
      max_iterations = as.integer(out$max_iterations)
    )
    return(result)
  }

  selected_gpu <- NA_character_
  if (backend == "cuda") {
    selected_gpu <- "cuda"
  } else if (backend == "metal") {
    selected_gpu <- "metal"
  } else if (backend == "gpu") {
    selected_gpu <- resolve_native_gpu_backend(need_knn = TRUE)
  }

  if (!is.na(selected_gpu)) {
    gpu_k <- if (isTRUE(exclude_self)) k + 1L else k
    if (gpu_k > 256L) {
      stop("Native GPU backends currently support `k <= 256`.", call. = FALSE)
    }
    if (selected_gpu == "cuda") {
      if (!isTRUE(cuda_available())) {
        stop("No CUDA GPU backend is available on this machine.", call. = FALSE)
      }
      out <- nn_cuda_cpp(data, points, as.integer(gpu_k), FALSE)
      if (isTRUE(exclude_self)) {
        out <- drop_self_knn_result(out, k)
      }
      return(finish_nn_result(out, "cuda", k, self_query))
    }
    if (selected_gpu == "metal") {
      if (!isTRUE(metal_available())) {
        stop("No Metal GPU backend is available on this machine.", call. = FALSE)
      }
      out <- nn_metal_cpp(data, points, as.integer(gpu_k), FALSE)
      if (isTRUE(exclude_self)) {
        out <- drop_self_knn_result(out, k)
      }
      return(finish_nn_result(out, "metal", k, self_query))
    }
    stop("No native GPU backend is available on this machine.", call. = FALSE)
  }

  if (should_use_nndescent_self_knn(
    backend = backend,
    self_query = self_query,
    n = nrow(data),
    p = ncol(data),
    k = k,
    exclude_self = isTRUE(exclude_self),
    work_size = work_size
  )) {
    nonself_k <- if (isTRUE(exclude_self)) k else max(0L, k - 1L)
    if (nonself_k < 1L) {
      out <- list(
        indices = matrix(seq_len(nrow(data)), nrow(data), 1L),
        distances = matrix(0, nrow(data), 1L)
      )
    } else {
      out <- nndescent_self_knn(data, nonself_k, n_threads = n_threads)
      if (!isTRUE(exclude_self)) {
        out$indices <- cbind(seq_len(nrow(data)), out$indices)
        out$distances <- cbind(rep(0, nrow(data)), out$distances)
      }
    }
    result <- finish_nn_result(out, "cpu_nndescent", k, self_query, exact = FALSE)
    attr(result, "approximation") <- attr(out, "approximation")
    return(result)
  }

  if (should_use_clustered_self_knn(
    backend = backend,
    self_query = self_query,
    n = nrow(data),
    p = ncol(data),
    k = k,
    work_size = work_size
  )) {
    out <- clustered_self_knn(data, k, exclude_self = isTRUE(exclude_self), n_threads = n_threads)
    return(finish_nn_result(out, "cpu_clustered", k, self_query, exact = FALSE))
  }

  if (backend %in% c("cpu_ivf", "cpu_faiss_ivf")) {
    if (!isTRUE(self_query)) {
      stop("`backend = \"", backend, "\"` is only available for self-KNN searches.", call. = FALSE)
    }
    nonself_k <- if (isTRUE(exclude_self)) k else max(0L, k - 1L)
    if (nonself_k < 1L) {
      out <- list(
        indices = matrix(seq_len(nrow(data)), nrow(data), 1L),
        distances = matrix(0, nrow(data), 1L)
      )
    } else {
      out <- ivf_self_knn(data, nonself_k, backend = backend, n_threads = n_threads)
      if (!isTRUE(exclude_self)) {
        out$indices <- cbind(seq_len(nrow(data)), out$indices)
        out$distances <- cbind(rep(0, nrow(data)), out$distances)
      }
    }
    result <- finish_nn_result(out, backend, k, self_query, exact = FALSE)
    attr(result, "approximation") <- attr(out, "approximation")
    return(result)
  }

  if (identical(backend, "cpu_annoy")) {
    if (!isTRUE(self_query)) {
      stop("`backend = \"cpu_annoy\"` is only available for self-KNN searches.", call. = FALSE)
    }
    nonself_k <- if (isTRUE(exclude_self)) k else max(0L, k - 1L)
    if (nonself_k < 1L) {
      out <- list(
        indices = matrix(seq_len(nrow(data)), nrow(data), 1L),
        distances = matrix(0, nrow(data), 1L)
      )
    } else {
      out <- annoy_self_knn(data, nonself_k, n_threads = n_threads)
      if (!isTRUE(exclude_self)) {
        out$indices <- cbind(seq_len(nrow(data)), out$indices)
        out$distances <- cbind(rep(0, nrow(data)), out$distances)
      }
    }
    result <- finish_nn_result(out, "cpu_annoy", k, self_query, exact = FALSE)
    attr(result, "approximation") <- attr(out, "approximation")
    return(result)
  }

  if (identical(backend, "cpu_vptree")) {
    if (!isTRUE(self_query)) {
      stop("`backend = \"cpu_vptree\"` is only available for self-KNN searches.", call. = FALSE)
    }
    nonself_k <- if (isTRUE(exclude_self)) k else max(0L, k - 1L)
    if (nonself_k < 1L) {
      out <- list(
        indices = matrix(seq_len(nrow(data)), nrow(data), 1L),
        distances = matrix(0, nrow(data), 1L)
      )
    } else {
      out <- vptree_self_knn(data, nonself_k, n_threads = n_threads)
      if (!isTRUE(exclude_self)) {
        out$indices <- cbind(seq_len(nrow(data)), out$indices)
        out$distances <- cbind(rep(0, nrow(data)), out$distances)
      }
    }
    return(finish_nn_result(out, "cpu_vptree", k, self_query, exact = TRUE))
  }

  if (identical(backend, "cpu_nndescent")) {
    if (!isTRUE(self_query)) {
      stop("`backend = \"cpu_nndescent\"` is only available for self-KNN searches.", call. = FALSE)
    }
    nonself_k <- if (isTRUE(exclude_self)) k else max(0L, k - 1L)
    if (nonself_k < 1L) {
      out <- list(
        indices = matrix(seq_len(nrow(data)), nrow(data), 1L),
        distances = matrix(0, nrow(data), 1L)
      )
    } else {
      out <- nndescent_self_knn(data, nonself_k, n_threads = n_threads)
      if (!isTRUE(exclude_self)) {
        out$indices <- cbind(seq_len(nrow(data)), out$indices)
        out$distances <- cbind(rep(0, nrow(data)), out$distances)
      }
    }
    result <- finish_nn_result(out, "cpu_nndescent", k, self_query, exact = FALSE)
    attr(result, "approximation") <- attr(out, "approximation")
    return(result)
  }

  if (identical(backend, "cpu_clustered")) {
    if (!isTRUE(self_query)) {
      stop("`backend = \"cpu_clustered\"` is only available for self-KNN searches.", call. = FALSE)
    }
    out <- clustered_self_knn(data, k, exclude_self = isTRUE(exclude_self), n_threads = n_threads)
    return(finish_nn_result(out, "cpu_clustered", k, self_query, exact = FALSE))
  }

  out <- nn_cpp(
    data,
    points,
    as.integer(k),
    "euclidean",
    FALSE,
    FALSE,
    0,
    TRUE,
    as.integer(n_threads),
    isTRUE(exclude_self)
  )
  finish_nn_result(out, "cpu", k, self_query)
}

resolve_auto_knn_gpu_backend <- function(backend,
                                         self_query,
                                         k,
                                         work_size) {
  if (!identical(backend, "auto")) return(NA_character_)
  if (!isTRUE(self_query)) return(NA_character_)
  if (k > 256L) return(NA_character_)
  if (work_size < 5e8) return(NA_character_)
  selected <- available_native_gpu_backend(need_knn = TRUE)
  if (is.na(selected)) return(NA_character_)
  if (identical(selected, "cuda")) {
    if (isTRUE(cuvs_available())) return("cuda_cuvs_nndescent")
    return(NA_character_)
  }
  paste0(selected, "_nndescent")
}

should_use_auto_cpu_approx_self_knn <- function(self_query,
                                                n,
                                                p,
                                                k,
                                                work_size) {
  if (!isTRUE(self_query)) return(FALSE)
  if (n < 5000L || k < 10L || p < 2L) return(FALSE)
  if (work_size < 5e8) return(FALSE)
  TRUE
}

select_cpu_approx_backend <- function(n, p, k) {
  n <- as.integer(n)
  p <- as.integer(p)
  k <- as.integer(k)
  if (n >= 10000L && p >= 2L && k >= 10L) {
    return("cpu_nndescent")
  }
  "cpu_ivf"
}

normalize_nn_threads <- function(n_threads) {
  if (is.null(n_threads)) {
    n_threads <- suppressWarnings(parallel::detectCores(logical = FALSE))
  }
  n_threads <- suppressWarnings(as.integer(n_threads))
  if (length(n_threads) != 1L || is.na(n_threads) || !is.finite(n_threads) || n_threads < 1L) {
    n_threads <- 1L
  }
  as.integer(max(1L, min(64L, n_threads)))
}

should_use_clustered_self_knn <- function(backend,
                                          self_query,
                                          n,
                                          p,
                                          k,
                                          work_size) {
  FALSE
}

metal_grid_params <- function(n, p, k) {
  grid_dims <- getOption("fastEmbedR.grid_dims", NULL)
  if (is.null(grid_dims)) {
    grid_dims <- min(5L, as.integer(p))
  } else {
    grid_dims <- suppressWarnings(as.integer(grid_dims))
    if (length(grid_dims) != 1L || is.na(grid_dims) || !is.finite(grid_dims)) {
      grid_dims <- min(5L, as.integer(p))
    }
  }
  grid_dims <- as.integer(max(1L, min(5L, as.integer(p), grid_dims)))

  bins <- getOption("fastEmbedR.grid_bins", NULL)
  if (is.null(bins)) {
    bins <- 0L
  } else {
    bins <- suppressWarnings(as.integer(bins))
    if (length(bins) != 1L || is.na(bins) || !is.finite(bins)) bins <- 0L
  }

  radius <- getOption("fastEmbedR.grid_radius", NULL)
  if (is.null(radius)) {
    radius <- 1L
  } else {
    radius <- suppressWarnings(as.integer(radius))
    if (length(radius) != 1L || is.na(radius) || !is.finite(radius)) radius <- 1L
  }

  list(
    grid_dims = as.integer(grid_dims),
    bins_per_dim = as.integer(bins),
    radius = as.integer(max(1L, min(4L, radius)))
  )
}

resolve_gpu_approx_backend <- function(backend,
                                       self_query,
                                       n,
                                       p,
                                       k,
                                       exclude_self,
                                       work_size) {
  explicit <- backend %in% c("gpu_approx", "metal_approx")
  if (!explicit) {
    return(NA_character_)
  }
  if (!isTRUE(self_query)) {
    if (explicit) {
      stop("Approximate GPU KNN currently supports self-KNN only.", call. = FALSE)
    }
    return(NA_character_)
  }
  if (isTRUE(exclude_self) && n < 2L) return(NA_character_)
  nonself_k <- if (isTRUE(exclude_self)) k else k - 1L
  if (nonself_k < 1L) return(NA_character_)
  if (k > 256L) {
    if (explicit) {
      stop("Approximate native GPU KNN currently supports `k <= 256`.", call. = FALSE)
    }
    return(NA_character_)
  }

  if (identical(backend, "metal_approx")) {
    if (!isTRUE(metal_available())) {
      stop("No Metal GPU backend is available on this machine.", call. = FALSE)
    }
    return("metal")
  }
  if (identical(backend, "gpu_approx")) {
    selected <- resolve_native_gpu_backend(need_knn = TRUE)
    if (identical(selected, "cuda")) {
      stop(
        "Native CUDA approximate KNN has been removed. ",
        "Use `backend = \"cuda_cuvs_nndescent\"` for RAPIDS cuVS NN-descent.",
        call. = FALSE
      )
    }
    return(selected)
  }
  available_native_gpu_backend(need_knn = TRUE)
}

resolve_gpu_nndescent_backend <- function(backend,
                                          self_query,
                                          n,
                                          p,
                                          k,
                                          exclude_self,
                                          work_size) {
  explicit <- backend %in% c(
    "gpu_nndescent", "metal_nndescent",
    "gpu_approx", "metal_approx"
  )
  if (!explicit) return(NA_character_)
  if (!isTRUE(self_query)) {
    stop("Native GPU NN-descent currently supports self-KNN only.", call. = FALSE)
  }
  if (isTRUE(exclude_self) && n < 2L) return(NA_character_)
  nonself_k <- if (isTRUE(exclude_self)) k else k - 1L
  if (nonself_k < 1L) return(NA_character_)
  if (p < 2L) return(NA_character_)
  if (k > 256L) {
    stop("Native GPU NN-descent currently supports `k <= 256`.", call. = FALSE)
  }

  selected <- if (backend %in% c("metal_nndescent", "metal_approx")) {
    if (!isTRUE(metal_available())) {
      stop("No Metal GPU backend is available on this machine.", call. = FALSE)
    }
    "metal"
  } else {
    resolve_native_gpu_backend(need_knn = TRUE)
  }
  if (identical(selected, "cuda")) {
    stop(
      "Native CUDA NN-descent has been removed. ",
      "Use `backend = \"cuda_cuvs_nndescent\"` for RAPIDS cuVS NN-descent.",
      call. = FALSE
    )
  }
  paste0(selected, "_nndescent")
}

resolve_gpu_ivf_backend <- function(backend,
                                    self_query,
                                    n,
                                    p,
                                    k,
                                    exclude_self) {
  out <- list(backend = NA_character_, label = NA_character_, strategy = NA_character_)
  explicit <- backend %in% c(
    "gpu_ivf", "cuda_ivf", "metal_ivf",
    "gpu_faiss", "cuda_faiss", "metal_faiss"
  )
  if (!explicit) return(out)
  if (!isTRUE(self_query)) {
    stop("GPU IVF/FAISS-style KNN currently supports self-KNN only.", call. = FALSE)
  }
  if (isTRUE(exclude_self) && n < 2L) return(out)
  nonself_k <- if (isTRUE(exclude_self)) k else k - 1L
  if (nonself_k < 1L) return(out)
  if (k > 256L) {
    stop("Native GPU IVF/FAISS-style KNN currently supports `k <= 256`.", call. = FALSE)
  }

  selected <- if (backend %in% c("cuda_ivf", "cuda_faiss")) {
    if (!isTRUE(cuda_available())) {
      stop("No CUDA GPU backend is available on this machine.", call. = FALSE)
    }
    "cuda"
  } else if (backend %in% c("metal_ivf", "metal_faiss")) {
    if (!isTRUE(metal_available())) {
      stop("No Metal GPU backend is available on this machine.", call. = FALSE)
    }
    "metal"
  } else {
    resolve_native_gpu_backend(need_knn = TRUE)
  }
  out$backend <- selected
  out$label <- switch(
    backend,
    gpu_ivf = paste0(selected, "_ivf"),
    gpu_faiss = paste0(selected, "_faiss_ivf"),
    cuda_faiss = "cuda_faiss_ivf",
    metal_faiss = "metal_faiss_ivf",
    backend
  )
  out$strategy <- if (grepl("faiss", backend, fixed = TRUE)) {
    "faiss_style_ivf_flat_native"
  } else {
    "ivf_flat_native"
  }
  out
}

should_use_gpu_approx_self_knn <- function(backend,
                                           self_query,
                                           n,
                                           p,
                                           k,
                                           exclude_self,
                                           work_size) {
  if (!backend %in% c("gpu_approx", "cuda_approx", "metal_approx")) return(FALSE)
  if (!isTRUE(self_query)) return(FALSE)
  if (p < 2L || k < 10L || k > 256L) return(FALSE)
  nonself_k <- if (isTRUE(exclude_self)) k else k - 1L
  if (nonself_k < 1L) return(FALSE)
  TRUE
}

fast_knn_approx_seed <- function() {
  value <- getOption("fastEmbedR.approx_knn_seed", 4L)
  value <- suppressWarnings(as.integer(value))
  if (length(value) != 1L || is.na(value) || !is.finite(value)) 4L else value
}

gpu_approx_params <- function(n, k) {
  n <- as.integer(n)
  k <- as.integer(k)
  anchors <- getOption("fastEmbedR.gpu_approx_anchors", NULL)
  if (is.null(anchors)) {
    anchors <- max(128L, ceiling(4 * sqrt(n)), ceiling(n / 500))
    anchors <- min(n - 1L, 4096L, as.integer(anchors))
  } else {
    anchors <- suppressWarnings(as.integer(anchors))
    if (length(anchors) != 1L || is.na(anchors) || !is.finite(anchors)) {
      anchors <- max(128L, ceiling(4 * sqrt(n)))
    }
    anchors <- max(2L, min(n - 1L, anchors))
  }
  projection_k <- getOption("fastEmbedR.gpu_approx_projection_k", NULL)
  if (is.null(projection_k)) {
    projection_k <- max(12L, ceiling(k / 2))
    projection_k <- min(anchors, as.integer(projection_k))
  } else {
    projection_k <- suppressWarnings(as.integer(projection_k))
    if (length(projection_k) != 1L || is.na(projection_k) || !is.finite(projection_k)) {
      projection_k <- max(12L, ceiling(k / 2))
    }
    projection_k <- max(1L, min(anchors, projection_k))
  }
  cols <- clustered_knn_graph_columns(projection_k, k)
  list(
    anchors = as.integer(anchors),
    projection_k = as.integer(projection_k),
    bucket_cols = as.integer(cols$bucket_cols),
    query_cols = as.integer(cols$query_cols)
  )
}

gpu_approx_self_knn <- function(data,
                                k,
                                backend,
                                exclude_self = FALSE,
                                seed = 4L,
                                label = NULL,
                                strategy = "anchor_projection_candidate_knn") {
  n <- nrow(data)
  label <- if (is.null(label)) paste0(backend, "_approx") else as.character(label)
  nonself_k <- if (isTRUE(exclude_self)) k else k - 1L
  if (nonself_k < 1L) {
    out <- list(
      indices = matrix(seq_len(n), n, 1L),
      distances = matrix(0, n, 1L)
    )
    return(finish_nn_result(out, label, k, TRUE, exact = FALSE))
  }

  params <- gpu_approx_params(n, nonself_k)
  anchors <- select_landmark_rows(data, params$anchors, seed)
  projection <- nn_compute(
    data[anchors, , drop = FALSE],
    data,
    k = params$projection_k,
    backend = backend,
    points_missing = FALSE,
    exclude_self = FALSE
  )
  out <- if (identical(backend, "cuda")) {
    landmark_candidate_knn_cuda_cpp(
      data,
      projection$indices,
      as.integer(nonself_k),
      as.integer(params$bucket_cols),
      as.integer(params$query_cols)
    )
  } else if (identical(backend, "metal")) {
    landmark_candidate_knn_metal_cpp(
      data,
      projection$indices,
      as.integer(nonself_k),
      as.integer(params$bucket_cols),
      as.integer(params$query_cols)
    )
  } else {
    stop("Unsupported approximate GPU backend.", call. = FALSE)
  }

  if (!isTRUE(exclude_self)) {
    out$indices <- cbind(seq_len(n), out$indices)
    out$distances <- cbind(rep(0, n), out$distances)
  }
  result <- finish_nn_result(out, label, k, TRUE, exact = FALSE)
  attr(result, "approximation") <- list(
    strategy = strategy,
    backend = backend,
    anchors = as.integer(params$anchors),
    projection_k = as.integer(params$projection_k),
    bucket_cols = as.integer(params$bucket_cols),
    query_cols = as.integer(params$query_cols),
    seed = as.integer(seed)
  )
  if (isTRUE(getOption("fastEmbedR.gpu_approx_recall", FALSE))) {
    result <- attach_knn_recall_subset(
      result,
      data = data,
      k = k,
      exclude_self = isTRUE(exclude_self),
      seed = seed
    )
  }
  result
}

gpu_ivf_self_knn <- function(data,
                             k,
                             backend,
                             label,
                             strategy,
                             exclude_self = FALSE,
                             seed = 4L) {
  gpu_approx_self_knn(
    data,
    k = k,
    backend = backend,
    exclude_self = exclude_self,
    seed = seed,
    label = label,
    strategy = strategy
  )
}

gpu_nndescent_option <- function(backend, name, default = NULL) {
  backend <- as.character(backend)
  keys <- c(
    sprintf("fastEmbedR.%s_nndescent_%s", backend, name),
    sprintf("fastEmbedR.gpu_nndescent_%s", name)
  )
  for (key in keys) {
    value <- getOption(key, NULL)
    if (!is.null(value)) return(value)
  }
  default
}

gpu_nndescent_graph_degree <- function(n, k, backend = "metal") {
  graph_degree <- gpu_nndescent_option(backend, "graph_degree", NULL)
  if (is.null(graph_degree)) {
    graph_degree <- if (!is.null(n) && n >= 50000L && k <= 30L) {
      max(k, 64L)
    } else {
      k
    }
  }
  graph_degree <- suppressWarnings(as.integer(graph_degree))
  if (length(graph_degree) != 1L || is.na(graph_degree) || !is.finite(graph_degree)) {
    graph_degree <- k
  }
  if (!is.null(n)) graph_degree <- min(graph_degree, as.integer(n) - 1L)
  as.integer(max(k, min(256L, graph_degree)))
}

gpu_nndescent_params <- function(k, backend = "metal", n = NULL) {
  graph_degree <- gpu_nndescent_graph_degree(n, k, backend = backend)
  default_iters <- if (backend %in% c("metal", "cuda") && !is.null(n) && n >= 50000L) {
    3L
  } else {
    1L
  }
  n_iters <- gpu_nndescent_option(backend, "iters", default_iters)
  sources <- gpu_nndescent_option(backend, "sources", NULL)
  neighbors <- gpu_nndescent_option(backend, "neighbors", NULL)
  delta <- gpu_nndescent_option(backend, "delta", 0.015)

  n_iters <- suppressWarnings(as.integer(n_iters))
  if (length(n_iters) != 1L || is.na(n_iters) || !is.finite(n_iters)) n_iters <- 1L
  n_iters <- as.integer(max(1L, min(5L, n_iters)))

  if (is.null(sources)) {
    sources <- max(3L, min(graph_degree, 10L))
  } else {
    sources <- suppressWarnings(as.integer(sources))
    if (length(sources) != 1L || is.na(sources) || !is.finite(sources)) {
      sources <- max(3L, min(graph_degree, 10L))
    }
  }
  sources <- as.integer(max(1L, min(graph_degree, sources)))

  if (is.null(neighbors)) {
    neighbors <- max(5L, min(graph_degree, ceiling(graph_degree / 2)))
  } else {
    neighbors <- suppressWarnings(as.integer(neighbors))
    if (length(neighbors) != 1L || is.na(neighbors) || !is.finite(neighbors)) {
      neighbors <- max(5L, min(graph_degree, ceiling(graph_degree / 2)))
    }
  }
  neighbors <- as.integer(max(1L, min(graph_degree, neighbors)))

  delta <- suppressWarnings(as.numeric(delta))
  if (length(delta) != 1L || is.na(delta) || !is.finite(delta) || delta < 0) {
    delta <- 0.015
  }

  list(
    graph_degree = graph_degree,
    n_iters = n_iters,
    sources = sources,
    neighbors = neighbors,
    delta = delta
  )
}

metal_nndescent_params <- function(k) {
  gpu_nndescent_params(k, backend = "metal")
}

gpu_nndescent_self_knn <- function(data,
                                   k,
                                   backend,
                                   seed = 4L) {
  n <- nrow(data)
  k <- as.integer(k)
  backend <- match.arg(as.character(backend), c("cuda", "metal"))
  if (length(k) != 1L || is.na(k) || !is.finite(k) || k < 1L || k >= n) {
    stop("`k` must be in [1, nrow(data) - 1].", call. = FALSE)
  }
  if (k > 256L) {
    stop("Native GPU NN-descent currently supports `k <= 256`.", call. = FALSE)
  }
  if (identical(backend, "cuda") && !isTRUE(cuda_available())) {
    stop("No CUDA GPU backend is available on this machine.", call. = FALSE)
  }
  if (identical(backend, "metal") && !isTRUE(metal_available())) {
    stop("No Metal GPU backend is available on this machine.", call. = FALSE)
  }
  output_k <- k
  params <- gpu_nndescent_params(output_k, backend = backend, n = n)
  work_k <- params$graph_degree

  base <- gpu_ivf_self_knn(
    data,
    k = work_k,
    backend = backend,
    label = paste0(backend, "_ivf_seed"),
    strategy = "ivf_flat_native",
    exclude_self = TRUE,
    seed = seed
  )
  indices <- base$indices
  distances <- base$distances
  changes <- numeric(params$n_iters)
  used_iters <- 0L
  candidate_stats <- list()
  flags <- matrix(TRUE, nrow = nrow(indices), ncol = ncol(indices))
  update_frac <- 1
  candidate_attr <- function(x, name, default) {
    value <- attr(x, name, exact = TRUE)
    if (is.null(value)) default else value
  }

  for (iter in seq_len(params$n_iters)) {
    if (backend %in% c("metal", "cuda")) {
      # Native port of the mlx-vis NN-descent schedule: expand aggressively
      # while the graph is moving, then use only NEW-neighbour sources and
      # skip reverse candidates near convergence.
      adaptive_scale <- min(1, sqrt(max(update_frac, 0)) * 3)
      min_sources <- min(work_k, 3L)
      iter_sources <- min(
        work_k,
        max(min_sources, as.integer(ceiling(params$sources * adaptive_scale)))
      )
      min_neighbors <- min(work_k, max(1L, as.integer(ceiling(work_k / 2))))
      iter_neighbors <- min(
        work_k,
        max(min_neighbors, as.integer(ceiling(params$neighbors * adaptive_scale)))
      )
      active_only <- isTRUE(update_frac < 0.5)
      use_reverse <- isTRUE(update_frac >= 0.10)
      candidate_indices <- nndescent_candidate_matrix_mlx_cpp(
        indices,
        flags,
        as.integer(iter_sources),
        as.integer(iter_neighbors),
        use_reverse = use_reverse,
        active_only = active_only
      )
    } else {
      candidate_indices <- nndescent_candidate_matrix_cpp(
        indices,
        as.integer(params$sources),
        as.integer(params$neighbors)
      )
    }
    candidate_stats[[iter]] <- list(
      columns = as.integer(ncol(candidate_indices)),
      mean_unique = as.numeric(attr(candidate_indices, "mean_unique_candidates")),
      max_unique = as.integer(attr(candidate_indices, "max_unique_candidates")),
      raw_columns = as.integer(attr(candidate_indices, "raw_candidate_columns")),
      sources = as.integer(candidate_attr(candidate_indices, "sources", params$sources)),
      neighbors = as.integer(candidate_attr(candidate_indices, "neighbors", params$neighbors)),
      active_rows = as.integer(candidate_attr(candidate_indices, "active_rows", n)),
      use_reverse = isTRUE(candidate_attr(candidate_indices, "use_reverse", TRUE)),
      active_only = isTRUE(candidate_attr(candidate_indices, "active_only", FALSE))
    )

    refined <- if (identical(backend, "cuda")) {
      row_candidate_knn_cuda_cpp(
        data,
        candidate_indices,
        as.integer(work_k)
      )
    } else {
      row_candidate_knn_metal_cpp(
        data,
        candidate_indices,
        as.integer(work_k)
      )
    }
    new_indices <- refined$indices
    new_flags <- new_indices != indices
    changes[iter] <- mean(new_flags)
    if (identical(backend, "metal")) {
      flags <- new_flags
      update_frac <- changes[iter]
    }
    indices <- new_indices
    distances <- refined$distances
    used_iters <- iter
    if (changes[iter] < params$delta) break
  }

  if (work_k > output_k) {
    keep <- seq_len(output_k)
    indices <- indices[, keep, drop = FALSE]
    distances <- distances[, keep, drop = FALSE]
  }

  out <- list(indices = indices, distances = distances)
  if (identical(backend, "cuda")) {
    attr(out, "cuda_kernel") <- "row_candidate_knn"
  } else {
    attr(out, "metal_kernel") <- "row_candidate_knn"
  }
  attr(out, "approximation") <- list(
    strategy = paste0("mlx_vis_adaptive_seeded_nndescent_native_", backend),
    backend = backend,
    seed_backend = paste0(backend, "_ivf"),
    output_graph_degree = as.integer(output_k),
    graph_degree = as.integer(work_k),
    n_iters = as.integer(used_iters),
    changes = changes[seq_len(used_iters)],
    sources = as.integer(params$sources),
    neighbors = as.integer(params$neighbors),
    candidate_columns = vapply(candidate_stats[seq_len(used_iters)], `[[`, integer(1L), "columns"),
    mean_unique_candidates = vapply(candidate_stats[seq_len(used_iters)], `[[`, numeric(1L), "mean_unique"),
    raw_candidate_columns = vapply(candidate_stats[seq_len(used_iters)], `[[`, integer(1L), "raw_columns"),
    iteration_sources = vapply(candidate_stats[seq_len(used_iters)], `[[`, integer(1L), "sources"),
    iteration_neighbors = vapply(candidate_stats[seq_len(used_iters)], `[[`, integer(1L), "neighbors"),
    active_rows = vapply(candidate_stats[seq_len(used_iters)], `[[`, integer(1L), "active_rows"),
    use_reverse = vapply(candidate_stats[seq_len(used_iters)], `[[`, logical(1L), "use_reverse"),
    active_only = vapply(candidate_stats[seq_len(used_iters)], `[[`, logical(1L), "active_only"),
    delta = as.numeric(params$delta),
    seed = as.integer(seed)
  )
  out
}

metal_nndescent_self_knn <- function(data,
                                     k,
                                     seed = 4L) {
  gpu_nndescent_self_knn(data, k = k, backend = "metal", seed = seed)
}

cuda_nndescent_self_knn <- function(data,
                                    k,
                                    seed = 4L) {
  gpu_nndescent_self_knn(data, k = k, backend = "cuda", seed = seed)
}

knn_recall_subset_size <- function(n) {
  value <- getOption("fastEmbedR.gpu_approx_recall_sample", 512L)
  value <- suppressWarnings(as.integer(value))
  if (length(value) != 1L || is.na(value) || !is.finite(value) || value < 1L) {
    value <- 512L
  }
  as.integer(min(n, value))
}

attach_knn_recall_subset <- function(result,
                                     data,
                                     k,
                                     exclude_self,
                                     seed) {
  n <- nrow(data)
  compare_k <- if (isTRUE(exclude_self)) k else k - 1L
  if (compare_k < 1L || n < 2L) {
    attr(result, "recall") <- data.frame(
      k = compare_k,
      recall_at_k = NA_real_,
      median_recall_at_k = NA_real_,
      min_recall_at_k = NA_real_,
      sample_size = 0L,
      stringsAsFactors = FALSE
    )
    return(result)
  }
  sample_size <- knn_recall_subset_size(n)
  set.seed(as.integer(seed) + 1009L)
  rows <- sort(sample.int(n, sample_size))
  exact_raw <- nn_compute(
    data,
    data[rows, , drop = FALSE],
    k = min(n, compare_k + 1L),
    backend = "cpu",
    points_missing = FALSE,
    exclude_self = FALSE
  )
  exact_idx <- matrix(0L, nrow = sample_size, ncol = compare_k)
  for (i in seq_along(rows)) {
    keep <- exact_raw$indices[i, ] != rows[i]
    row_idx <- exact_raw$indices[i, keep]
    if (length(row_idx) < compare_k) {
      row_idx <- exact_raw$indices[i, seq_len(ncol(exact_raw$indices))]
    }
    exact_idx[i, ] <- row_idx[seq_len(compare_k)]
  }
  approx_idx <- if (isTRUE(exclude_self)) {
    result$indices[rows, seq_len(compare_k), drop = FALSE]
  } else {
    result$indices[rows, 1L + seq_len(compare_k), drop = FALSE]
  }
  recall <- knn_recall(
    list(indices = approx_idx),
    list(indices = exact_idx),
    k = compare_k
  )
  recall$sample_size <- as.integer(sample_size)
  attr(result, "recall") <- recall
  result
}

should_use_nndescent_self_knn <- function(backend,
                                          self_query,
                                          n,
                                          p,
                                          k,
                                          exclude_self,
                                          work_size) {
  if (!identical(backend, "cpu_nndescent")) return(FALSE)
  if (!isTRUE(self_query)) return(FALSE)
  if (n < 10000L || k < 10L || p < 2L) return(FALSE)
  if (work_size < 5e8) return(FALSE)
  nonself_k <- if (isTRUE(exclude_self)) k else k - 1L
  nonself_k >= 1L
}

nndescent_pool_size <- function(n, k) {
  as.integer(min(n - 1L, max(k + 15L, min(160L, ceiling(2.5 * k)))))
}

nndescent_iterations <- function(n, k) {
  out <- if (n >= 50000L) 3L else 4L
  if (k < 30L) out <- out + 1L
  as.integer(out)
}

nndescent_self_knn <- function(data,
                               k,
                               seed = 4L,
                               n_threads = NULL) {
  n <- nrow(data)
  k <- as.integer(k)
  if (length(k) != 1L || is.na(k) || !is.finite(k) || k < 1L || k >= n) {
    stop("`k` must be in [1, nrow(data) - 1].", call. = FALSE)
  }
  if (is.null(n_threads)) {
    n_threads <- suppressWarnings(parallel::detectCores(logical = FALSE))
    if (length(n_threads) != 1L || is.na(n_threads) || !is.finite(n_threads)) {
      n_threads <- 1L
    }
  }
  pool_size <- nndescent_pool_size(n, k)
  n_iters <- nndescent_iterations(n, k)
  max_candidates <- as.integer(min(n - 1L, max(pool_size * 4L, k * 12L)))
  n_random_projections <- if (n >= 50000L) 8L else 6L
  out <- nndescent_self_knn_cpp(
    data,
    as.integer(k),
    as.integer(pool_size),
    as.integer(n_iters),
    as.integer(max_candidates),
    as.integer(n_random_projections),
    as.integer(seed),
    TRUE,
    as.integer(max(1L, min(8L, n_threads)))
  )
  params <- list(
    strategy = "native_cpu_nndescent",
    backend = "cpu",
    pool_size = pool_size,
    n_iters = n_iters,
    max_candidates = max_candidates,
    n_random_projections = n_random_projections,
    reverse_candidates = "rank_ordered"
  )
  attr(out, "nndescent") <- params
  attr(out, "approximation") <- params
  out
}

clustered_knn_center_count <- function(n, k) {
  n <- as.integer(n)
  k <- as.integer(k)
  max_centers <- max(2L, n - 1L)
  count <- max(16L, ceiling(sqrt(n) * 2))
  count <- max(count, ceiling(n / max(250L, 10L * k)))
  as.integer(min(max_centers, count))
}

clustered_knn_graph_columns <- function(projection_k, k) {
  projection_k <- as.integer(max(1L, projection_k))
  k <- as.integer(max(1L, k))
  bucket_cols <- min(projection_k, max(2L, min(4L, ceiling(k / 10))))
  query_cols <- min(projection_k, max(bucket_cols, min(8L, ceiling(k / 5))))
  list(bucket_cols = bucket_cols, query_cols = query_cols)
}

clustered_self_knn <- function(data,
                               k,
                               exclude_self = TRUE,
                               seed = 4L,
                               n_threads = NULL) {
  n <- nrow(data)
  if (n < 2L) {
    stop("`data` must have at least two rows.", call. = FALSE)
  }
  k <- as.integer(k)
  if (length(k) != 1L || is.na(k) || !is.finite(k) || k < 1L) {
    stop("`k` must be a positive integer.", call. = FALSE)
  }

  nonself_k <- if (isTRUE(exclude_self)) k else k - 1L
  if (nonself_k < 1L) {
    return(list(
      indices = matrix(seq_len(n), n, 1L),
      distances = matrix(0, n, 1L)
    ))
  }
  if (nonself_k >= n) {
    stop("`k` cannot be larger than the available neighbor count.", call. = FALSE)
  }

  n_centers <- clustered_knn_center_count(n, nonself_k)
  centers <- select_landmark_rows(data, n_centers, seed)
  x_centers <- data[centers, , drop = FALSE]
  projection_k <- min(n_centers, max(2L, min(12L, ceiling(nonself_k / 2))))
  projection <- nn_compute(
    x_centers,
    data,
    k = projection_k,
    backend = "cpu",
    points_missing = FALSE,
    exclude_self = FALSE,
    n_threads = n_threads
  )
  cols <- clustered_knn_graph_columns(ncol(projection$indices), nonself_k)
  n_threads <- normalize_nn_threads(n_threads)
  out <- landmark_candidate_knn_cpp(
    data,
    projection$indices,
    as.integer(nonself_k),
    as.integer(cols$bucket_cols),
    as.integer(cols$query_cols),
    TRUE,
    as.integer(max(1L, min(8L, n_threads)))
  )
  storage.mode(out$indices) <- "integer"
  if (!identical(typeof(out$distances), "double")) storage.mode(out$distances) <- "double"

  if (!isTRUE(exclude_self)) {
    out$indices <- cbind(seq_len(n), out$indices)
    out$distances <- cbind(rep(0, n), out$distances)
  }
  out
}

annoy_tree_count <- function(n, k) {
  value <- getOption("fastEmbedR.annoy_n_trees", NULL)
  if (is.null(value)) {
    value <- if (n >= 10000L) 50L else 24L
  }
  value <- suppressWarnings(as.integer(value))
  if (length(value) != 1L || is.na(value) || !is.finite(value)) value <- 24L
  as.integer(max(1L, min(128L, value)))
}

annoy_leaf_size <- function(k) {
  value <- getOption("fastEmbedR.annoy_leaf_size", NULL)
  if (is.null(value)) {
    value <- max(64L, min(256L, ceiling(2L * k)))
  }
  value <- suppressWarnings(as.integer(value))
  if (length(value) != 1L || is.na(value) || !is.finite(value)) value <- max(64L, 2L * k)
  as.integer(max(k + 1L, min(1024L, value)))
}

annoy_search_k <- function(k, n_trees, leaf_size) {
  value <- getOption("fastEmbedR.annoy_search_k", NULL)
  if (is.null(value)) {
    value <- max(n_trees * leaf_size, 50L * k)
  }
  value <- suppressWarnings(as.integer(value))
  if (length(value) != 1L || is.na(value) || !is.finite(value)) value <- n_trees * leaf_size
  as.integer(max(k, value))
}

annoy_self_knn <- function(data,
                           k,
                           seed = 4L,
                           n_threads = NULL) {
  n <- nrow(data)
  k <- as.integer(k)
  if (length(k) != 1L || is.na(k) || !is.finite(k) || k < 1L || k >= n) {
    stop("`k` must be in [1, nrow(data) - 1].", call. = FALSE)
  }
  n_threads <- normalize_nn_threads(n_threads)
  n_trees <- annoy_tree_count(n, k)
  leaf_size <- annoy_leaf_size(k)
  search_k <- annoy_search_k(k, n_trees, leaf_size)
  out <- annoy_self_knn_cpp(
    data,
    as.integer(k),
    as.integer(n_trees),
    as.integer(leaf_size),
    as.integer(search_k),
    as.integer(seed),
    TRUE,
    as.integer(max(1L, min(8L, n_threads)))
  )
  attr(out, "approximation") <- list(
    strategy = "annoy_style_random_projection_forest_native",
    backend = "cpu_annoy",
    n_trees = as.integer(out$n_trees),
    leaf_size = as.integer(out$leaf_size),
    search_k = as.integer(out$search_k),
    n_leaves = as.integer(out$n_leaves),
    seed = as.integer(seed)
  )
  out
}

ivf_list_count <- function(n, k) {
  n <- as.integer(n)
  k <- as.integer(k)
  count <- max(16L, ceiling(sqrt(n)))
  count <- min(count, ceiling(n / max(50L, 20L * k)))
  as.integer(max(4L, min(n, count, 1024L)))
}

ivf_probe_count <- function(nlist, k) {
  nlist <- as.integer(nlist)
  k <- as.integer(k)
  as.integer(max(1L, min(nlist, max(4L, ceiling(sqrt(nlist)), ceiling(k / 5)))))
}

faiss_ivf_params <- function(n, k) {
  n <- as.integer(n)
  k <- as.integer(k)
  nlist <- getOption("fastEmbedR.faiss_nlist", NULL)
  if (is.null(nlist)) nlist <- getOption("fastEmbedR.ivf_nlist", NULL)
  nprobe <- getOption("fastEmbedR.faiss_nprobe", NULL)
  if (is.null(nprobe)) nprobe <- getOption("fastEmbedR.ivf_nprobe", NULL)

  nlist <- if (is.null(nlist)) ivf_list_count(n, k) else suppressWarnings(as.integer(nlist))
  if (length(nlist) != 1L || is.na(nlist) || !is.finite(nlist)) {
    nlist <- ivf_list_count(n, k)
  }
  nlist <- max(1L, min(n, nlist))

  nprobe <- if (is.null(nprobe)) ivf_probe_count(nlist, k) else suppressWarnings(as.integer(nprobe))
  if (length(nprobe) != 1L || is.na(nprobe) || !is.finite(nprobe)) {
    nprobe <- ivf_probe_count(nlist, k)
  }
  nprobe <- max(1L, min(nlist, nprobe))
  list(nlist = as.integer(nlist), nprobe = as.integer(nprobe))
}

cuvs_option_int <- function(name, default, min_value = 1L, max_value = .Machine$integer.max) {
  value <- getOption(paste0("fastEmbedR.cuvs_", name), NULL)
  value <- if (is.null(value)) default else suppressWarnings(as.integer(value))
  if (length(value) != 1L || is.na(value) || !is.finite(value)) value <- default
  as.integer(max(min_value, min(max_value, value)))
}

cuvs_cagra_params <- function(n, k) {
  n <- as.integer(n)
  k <- as.integer(k)
  graph_degree <- cuvs_option_int(
    "graph_degree",
    default = max(64L, k + 1L),
    min_value = k + 1L,
    max_value = max(1L, n - 1L)
  )
  intermediate_graph_degree <- cuvs_option_int(
    "intermediate_graph_degree",
    default = max(128L, graph_degree * 2L),
    min_value = graph_degree,
    max_value = max(1L, n - 1L)
  )
  search_width <- cuvs_option_int(
    "search_width",
    default = 0L,
    min_value = 0L,
    max_value = 1024L
  )
  itopk_size <- cuvs_option_int(
    "itopk_size",
    default = max(64L, graph_degree),
    min_value = k,
    max_value = 4096L
  )
  list(
    graph_degree = as.integer(graph_degree),
    intermediate_graph_degree = as.integer(intermediate_graph_degree),
    search_width = as.integer(search_width),
    itopk_size = as.integer(itopk_size)
  )
}

cuvs_nndescent_params <- function(n, k) {
  n <- as.integer(n)
  k <- as.integer(k)
  graph_degree <- cuvs_option_int(
    "nndescent_graph_degree",
    default = k,
    min_value = k,
    max_value = max(1L, n - 1L)
  )
  intermediate_graph_degree <- cuvs_option_int(
    "nndescent_intermediate_graph_degree",
    default = max(graph_degree * 2L, graph_degree),
    min_value = graph_degree,
    max_value = max(1L, n - 1L)
  )
  max_iterations <- cuvs_option_int(
    "nndescent_max_iterations",
    default = 20L,
    min_value = 1L,
    max_value = 200L
  )
  list(
    graph_degree = as.integer(graph_degree),
    intermediate_graph_degree = as.integer(intermediate_graph_degree),
    max_iterations = as.integer(max_iterations)
  )
}

cuvs_nndescent_threshold <- function() {
  value <- getOption("fastEmbedR.cuvs_nndescent_threshold", 50000L)
  value <- suppressWarnings(as.integer(value))
  if (length(value) != 1L || is.na(value) || !is.finite(value)) value <- 50000L
  as.integer(max(2L, value))
}

cuvs_should_use_nndescent <- function(self_query, n) {
  isTRUE(self_query) && as.integer(n) >= cuvs_nndescent_threshold()
}

require_cuvs_backend <- function(label = "cuVS") {
  if (isTRUE(cuvs_available())) return(invisible(TRUE))
  info <- tryCatch(cuvs_info_json_cpp(), error = function(e) NA_character_)
  reason <- json_get_string(info, "reason")
  suffix <- if (!is.na(reason) && nzchar(reason)) paste0(" Reason: ", reason, ".") else ""
  stop(
    label,
    " backend is not available in this build or no CUDA device is visible.",
    suffix,
    " Reinstall with `FASTEMBEDR_USE_CUDA=1 FASTEMBEDR_USE_CUVS=1` and ",
    "`CUVS_HOME` pointing to a RAPIDS cuVS installation.",
    call. = FALSE
  )
}

ivf_self_knn <- function(data,
                         k,
                         backend = "cpu_ivf",
                         seed = 4L,
                         n_threads = NULL) {
  n <- nrow(data)
  k <- as.integer(k)
  if (length(k) != 1L || is.na(k) || !is.finite(k) || k < 1L || k >= n) {
    stop("`k` must be in [1, nrow(data) - 1].", call. = FALSE)
  }
  n_threads <- normalize_nn_threads(n_threads)
  nlist <- getOption("fastEmbedR.ivf_nlist", NULL)
  nprobe <- getOption("fastEmbedR.ivf_nprobe", NULL)
  nlist <- if (is.null(nlist)) ivf_list_count(n, k) else as.integer(nlist)
  if (length(nlist) != 1L || is.na(nlist) || !is.finite(nlist)) {
    nlist <- ivf_list_count(n, k)
  }
  nlist <- max(1L, min(as.integer(n), nlist))
  nprobe <- if (is.null(nprobe)) ivf_probe_count(nlist, k) else as.integer(nprobe)
  if (length(nprobe) != 1L || is.na(nprobe) || !is.finite(nprobe)) {
    nprobe <- ivf_probe_count(nlist, k)
  }
  nprobe <- max(1L, min(nlist, nprobe))
  out <- ivf_self_knn_cpp(
    data,
    as.integer(k),
    as.integer(nlist),
    as.integer(nprobe),
    as.integer(seed),
    TRUE,
    as.integer(max(1L, min(8L, n_threads)))
  )
  attr(out, "approximation") <- list(
    strategy = if (identical(backend, "cpu_faiss_ivf")) {
      "faiss_style_ivf_flat_native"
    } else {
      "ivf_flat_native"
    },
    backend = backend,
    nlist = as.integer(out$nlist),
    nprobe = as.integer(out$nprobe),
    seed = as.integer(seed)
  )
  out
}

vptree_self_knn <- function(data,
                            k,
                            n_threads = NULL) {
  n <- nrow(data)
  k <- as.integer(k)
  if (length(k) != 1L || is.na(k) || !is.finite(k) || k < 1L || k >= n) {
    stop("`k` must be in [1, nrow(data) - 1].", call. = FALSE)
  }
  n_threads <- normalize_nn_threads(n_threads)
  vptree_self_knn_cpp(
    data,
    as.integer(k),
    TRUE,
    as.integer(max(1L, min(8L, n_threads)))
  )
}

#' Nearest neighbors from row-wise matrices
#'
#' `nn()` provides a package-native nearest-neighbor entry point compatible with
#' the common `nn(data, points, k)` use case. Explicit `backend = "cpu"`
#' performs exact brute-force search in C++ and is mainly a reference path for
#' small data or recall checks. `backend = "auto"` chooses a recorded fast path
#' for large self-KNN searches: native CUDA/Metal NN-descent when available,
#' then a native CPU approximate path. Use `backend = "cpu_approx"` to force the CPU
#' approximate selector, or `backend = "cpu_ivf"`, `"cpu_nndescent"`,
#' `"cpu_clustered"`, or `"cpu_vptree"` to choose a specific native multi-CPU
#' strategy. Exact Euclidean search can also use native CUDA or Metal GPU
#' backends when requested explicitly.
#'
#' @param data Numeric matrix of reference observations in rows.
#' @param points Numeric matrix of query observations in rows. Defaults to
#'   `data`.
#' @param k Number of neighbors to return. `NULL` chooses the package's
#'   automatic neighborhood size and includes the self-neighbor when `points`
#'   is `data`.
#' @param backend Execution backend. `"auto"` records the selected fast backend
#'   in `attr(result, "backend")`. `"cpu"` always uses the exact C++ CPU path.
#'   `"cpu_approx"` chooses the current native CPU approximation. `"cpu_ivf"` and
#'   `"cpu_faiss_ivf"` use a package-native IVF-flat search; the latter records
#'   the FAISS-style IVF-flat strategy but does not link to external FAISS.
#'   `"faiss"` uses the real FAISS C++ `IndexFlatL2` backend when fastEmbedR was
#'   built against FAISS, and `"faiss_ivf"` uses real FAISS `IndexIVFFlat`.
#'   `"cuda_cuvs"` uses a RAPIDS-inspired cuVS policy: exact cuVS brute force
#'   for small searches and cuVS NN-descent for large self-KNN. Use
#'   `"cuda_cuvs_cagra"` to force cuVS CAGRA, `"cuda_cuvs_bruteforce"` for
#'   exact cuVS brute-force search, and `"cuda_cuvs_nndescent"` for cuVS
#'   NN-descent self-KNN.
#'   `"cpu_annoy"` uses a package-native Annoy-style random-projection forest.
#'   `"cpu_nndescent"`, `"cpu_clustered"`, and `"cpu_vptree"` force the native
#'   NN-descent, legacy clustered-IVF, and VP-tree strategies for self-KNN searches.
#'   `"gpu"` requests CUDA when available and otherwise Metal. `"cuda"` and
#'   `"metal"` request exact GPU backends explicitly. `"metal_nndescent"`
#'   requests seeded native Metal NN-descent refinement. `"cuda_nndescent"`
#'   and `"cuda_approx"` are compatibility aliases for
#'   `"cuda_cuvs_nndescent"`; CUDA approximate KNN is routed through RAPIDS
#'   cuVS NN-descent rather than the removed native CUDA NN-descent branch.
#'   `"gpu_nndescent"`, `"gpu_approx"`, and `"metal_approx"` are convenience
#'   aliases for the preferred native GPU NN-descent path. `"gpu_ivf"`,
#'   `"cuda_ivf"`, `"metal_ivf"`, `"gpu_faiss"`, `"cuda_faiss"`, and
#'   `"metal_faiss"` request native GPU IVF/FAISS-style IVF-flat searches and
#'   fail clearly if the requested GPU backend is unavailable. `"metal_grid"`
#'   requests the experimental FastGraph-style Metal grid-candidate search for
#'   low-dimensional or PCA-reduced self-KNN.
#' @param n_threads Number of CPU worker threads for CPU backends. GPU backends
#'   ignore this argument.
#' @return A list with integer matrix `indices` and numeric matrix `distances`.
#'   Indices are 1-based.
#' @export
nn <- function(data,
               points = data,
               k = NULL,
               backend = c(
                 "auto", "cpu", "cpu_approx", "cpu_ivf", "cpu_faiss_ivf",
                 "faiss", "cpu_faiss", "cpu_faiss_flat", "faiss_ivf", "cpu_faiss_index_ivf",
                 "cuvs", "gpu_cuvs", "cuda_cuvs", "cuda_cuvs_cagra",
                 "cuvs_bruteforce", "cuda_cuvs_bruteforce", "cuda_cuvs_exact",
                 "cuvs_nndescent", "cuda_cuvs_nndescent",
	                 "cpu_annoy", "cpu_nndescent", "cpu_clustered", "cpu_vptree",
	                 "gpu", "cuda", "metal", "gpu_approx", "cuda_approx", "metal_approx",
	                 "gpu_nndescent", "cuda_nndescent", "metal_nndescent",
	                 "gpu_ivf", "cuda_ivf", "metal_ivf", "gpu_faiss", "cuda_faiss", "metal_faiss",
	                 "metal_grid"
	               ),
               n_threads = NULL) {
  backend <- match.arg(backend)
  nn_compute(data, points, k, backend, missing(points), exclude_self = FALSE, n_threads = n_threads)
}

nn_without_self <- function(data,
                            k,
                            backend = c(
                              "auto", "cpu", "cpu_approx", "cpu_ivf", "cpu_faiss_ivf",
                              "faiss", "cpu_faiss", "cpu_faiss_flat", "faiss_ivf", "cpu_faiss_index_ivf",
                              "cuvs", "gpu_cuvs", "cuda_cuvs", "cuda_cuvs_cagra",
                              "cuvs_bruteforce", "cuda_cuvs_bruteforce", "cuda_cuvs_exact",
                              "cuvs_nndescent", "cuda_cuvs_nndescent",
	                              "cpu_annoy", "cpu_nndescent", "cpu_clustered", "cpu_vptree",
	                              "gpu", "cuda", "metal", "gpu_approx", "cuda_approx", "metal_approx",
	                              "gpu_nndescent", "cuda_nndescent", "metal_nndescent",
	                              "gpu_ivf", "cuda_ivf", "metal_ivf", "gpu_faiss", "cuda_faiss", "metal_faiss",
	                              "metal_grid"
	                            ),
                            n_threads = NULL) {
  backend <- match.arg(backend)
  nn_compute(data, data, k, backend, TRUE, exclude_self = TRUE, n_threads = n_threads)
}

knn_recall <- function(approx, exact, k = NULL) {
  approx_idx <- if (is.list(approx)) approx$indices else approx
  exact_idx <- if (is.list(exact)) exact$indices else exact
  approx_idx <- as.matrix(approx_idx)
  exact_idx <- as.matrix(exact_idx)
  if (nrow(approx_idx) != nrow(exact_idx)) {
    stop("Approximate and exact KNN must have the same number of rows.", call. = FALSE)
  }
  k <- if (is.null(k)) min(ncol(approx_idx), ncol(exact_idx)) else as.integer(k)
  if (length(k) != 1L || is.na(k) || !is.finite(k) || k < 1L) {
    stop("`k` must be a positive integer.", call. = FALSE)
  }
  k <- min(k, ncol(approx_idx), ncol(exact_idx))
  values <- knn_recall_cpp(approx_idx, exact_idx, as.integer(k))
  data.frame(
    k = k,
    recall_at_k = unname(values["recall_at_k"]),
    median_recall_at_k = unname(values["median_recall_at_k"]),
    min_recall_at_k = unname(values["min_recall_at_k"]),
    stringsAsFactors = FALSE
  )
}

drop_self_knn_result <- function(raw, k) {
  indices <- raw$indices
  distances <- raw$distances
  if (!is.matrix(indices)) indices <- as.matrix(indices)
  if (!is.matrix(distances)) distances <- as.matrix(distances)
  if (!is.integer(indices)) storage.mode(indices) <- "integer"
  if (!identical(typeof(distances), "double")) storage.mode(distances) <- "double"
  keep <- matrix(1L, nrow(indices), k)
  keep_dist <- matrix(0, nrow(indices), k)
  for (i in seq_len(nrow(indices))) {
    row_keep <- which(indices[i, ] != i | distances[i, ] > sqrt(.Machine$double.eps))
    if (length(row_keep) < k) {
      row_keep <- seq_len(ncol(indices))
    }
    row_keep <- row_keep[seq_len(k)]
    keep[i, ] <- indices[i, row_keep]
    keep_dist[i, ] <- distances[i, row_keep]
  }
  list(indices = keep, distances = keep_dist)
}

#' @export
print.fastEmbedR_nn <- function(x, ...) {
  cat("fastEmbedR KNN\n")
  cat("  queries: ", nrow(x$indices), "\n", sep = "")
  cat("  neighbors: ", ncol(x$indices), "\n", sep = "")
  cat("  backend: ", attr(x, "backend"), "\n", sep = "")
  if (!isTRUE(attr(x, "exact"))) {
    cat("  exact: false\n")
    recall <- attr(x, "recall")
    if (is.data.frame(recall) && nrow(recall) >= 1L && is.finite(recall$recall_at_k[1L])) {
      cat(
        "  recall@",
        recall$k[1L],
        " on exact subset: ",
        formatC(recall$recall_at_k[1L], digits = 3, format = "f"),
        " (n=",
        recall$sample_size[1L],
        ")\n",
        sep = ""
      )
    }
  }
  if (isTRUE(attr(x, "self_query"))) {
    cat("  first column: self-neighbor\n")
  }
  invisible(x)
}

#' Check whether the native Metal backend is available
#'
#' @return `TRUE` when a Metal device is available to the package.
#' @export
metal_available <- function() {
  isTRUE(metal_available_cpp())
}

#' Check whether the native CUDA backend is available
#'
#' @return `TRUE` when the package was built with CUDA support and the CUDA
#'   runtime reports at least one available device.
#' @export
cuda_available <- function() {
  isTRUE(cuda_available_cpp())
}

#' Check whether the real FAISS C++ backend is available
#'
#' @return `TRUE` when fastEmbedR was compiled and linked against FAISS.
#' @export
faiss_available <- function() {
  isTRUE(faiss_available_cpp())
}

#' Check whether the RAPIDS cuVS backend is available
#'
#' @return `TRUE` when fastEmbedR was compiled and linked against RAPIDS cuVS
#'   and the CUDA runtime reports at least one available device.
#' @export
cuvs_available <- function() {
  isTRUE(cuvs_available_cpp())
}
