exact_knn_reference <- function(x, k) {
    squared_norm <- rowSums(x * x)
    d2 <- outer(squared_norm, squared_norm, "+") - 2 * tcrossprod(x)
    diag(d2) <- Inf
    indices <- t(vapply(seq_len(nrow(x)), function(i) {
        order(d2[i, ], method = "radix")[seq_len(k)]
    }, integer(k)))
    list(indices = indices, distances = sqrt(pmax(0, matrix(d2[cbind(
        rep(seq_len(nrow(x)), each = k), as.vector(t(indices))
    )], nrow(x), k, byrow = TRUE))))
}

knn_recall_test <- function(observed, expected) {
    k <- ncol(expected$indices)
    mean(vapply(seq_len(nrow(expected$indices)), function(i) {
        length(intersect(observed$indices[i, ], expected$indices[i, ])) / k
    }, numeric(1)))
}

test_that("native CPU HNSW reaches its recall tier", {
    set.seed(4)
    x <- matrix(rnorm(600 * 12), nrow = 600)
    truth <- exact_knn_reference(x, 10L)
    observed <- native_hnsw_knn_cpp(x, 10L, 2L, "euclidean", 0.99)

    expect_identical(dim(observed$indices), c(600L, 10L))
    expect_gte(knn_recall_test(observed, truth), 0.99)
    expect_identical(attr(observed, "backend"), "cpu")
    expect_identical(observed$method, "native_hnsw")
    expect_false(observed$recall_audited)
    expect_true(is.na(observed$target_met))
    expect_identical(
        observed$recall_status,
        "not_audited_fixed_heuristic"
    )
})

test_that("native CPU HNSW construction is invariant to thread count", {
    set.seed(44)
    x <- matrix(rnorm(800 * 12), nrow = 800)
    serial <- native_hnsw_knn_cpp(x, 12L, 1L, "euclidean", 0.99)
    parallel <- native_hnsw_knn_cpp(x, 12L, 4L, "euclidean", 0.99)

    expect_identical(parallel$indices, serial$indices)
    expect_identical(parallel$distances, serial$distances)
})

test_that(paste(
    "precompute_knn exposes the native CPU policy without",
    "self neighbors"
), {
    set.seed(24)
    x <- matrix(rnorm(240 * 10), nrow = 240)
    truth <- exact_knn_reference(x, 12L)
    observed <- precompute_knn(
        x,
        k = 12L, metric = "euclidean", backend = "cpu", n.cores = 2L
    )

    expect_s3_class(observed, "fastEmbedR_knn")
    expect_identical(dim(observed$indices), c(240L, 12L))
    expect_identical(dim(observed$distances), c(240L, 12L))
    expect_false(any(observed$indices == row(observed$indices)))
    expect_gte(knn_recall_test(observed, truth), 0.99)
    expect_identical(observed$backend_requested, "cpu")
    expect_identical(observed$execution_backend, "cpu")
    expect_identical(observed$engine, "native_cpu_hnsw")
    expect_identical(observed$result_residency, "host")
    expect_null(observed$target_recall)
    expect_false(observed$recall_audited)
    expect_identical(
        observed$recall_status,
        "not_audited_fixed_heuristic"
    )
    expect_true(is.finite(observed$elapsed_sec))
    expect_identical(attr(observed, "exclude_self"), TRUE)

    layout <- umap_knn(observed, backend = "cpu", seed = 1L)
    expect_identical(dim(layout), c(240L, 2L))
    expect_true(all(is.finite(layout)))

    tsne_layout <- tsne_knn(
        observed,
        perplexity = 4,
        init_data = x,
        backend = "cpu",
        n.cores = 2L,
        early_exaggeration_iter = 2L,
        n_iter = 3L
    )
    expect_identical(dim(tsne_layout), c(240L, 2L))
    expect_true(all(is.finite(tsne_layout)))
})

test_that("precompute_knn preserves the float32 host path", {
    skip_if_not_installed("float")
    set.seed(25)
    x <- float::fl(matrix(rnorm(180 * 7), nrow = 180))
    observed <- precompute_knn(x, k = 9L, backend = "cpu", n.cores = 2L)

    expect_s3_class(observed, "fastEmbedR_knn")
    expect_true(inherits(observed$distances, "float32"))
    expect_identical(dim(observed$distances), c(180L, 9L))
})

test_that("precompute_knn exposes no search-algorithm selector", {
    expect_identical(
        names(formals(precompute_knn)),
        c("data", "k", "metric", "backend", "n.cores")
    )
    expect_error(
        precompute_knn(matrix(rnorm(40), nrow = 10), k = 10L),
        "between 1 and nrow\\(data\\) - 1"
    )
})

test_that("precompute_knn Metal output is native or fails explicitly", {
    set.seed(27)
    x <- matrix(rnorm(96 * 6), nrow = 96)
    if (!isTRUE(native_metal_knn_available_cpp())) {
        expect_error(
            precompute_knn(x, k = 7L, backend = "metal"),
            "Native Metal KNN"
        )
    } else {
        observed <- precompute_knn(x, k = 7L, backend = "metal")
        expect_s3_class(observed, "fastEmbedR_knn")
        expect_identical(observed$execution_backend, "metal")
        expect_identical(observed$result_residency, "host")
        expect_match(observed$method, "native_metal_(exact|ivf)")
    }
})

test_that("native CPU HNSW supports query-to-reference search", {
    set.seed(14)
    reference <- matrix(rnorm(240 * 10), nrow = 240)
    query <- matrix(rnorm(40 * 10), nrow = 40)
    truth <- test_exact_knn(reference, query, k = 12L)
    observed <- native_hnsw_query_cpp(
        reference, query, 12L, 2L, "euclidean", 0.99
    )

    expect_identical(dim(observed$indices), c(40L, 12L))
    expect_gte(knn_recall_test(observed, truth), 0.99)
    expect_identical(attr(observed, "backend"), "cpu")
    expect_identical(observed$method, "native_hnsw_query")
})

test_that("fastEmbedR has no faissR package dependency or runtime bridge", {
    desc <- utils::packageDescription("fastEmbedR")
    dependency_text <- paste(
        unlist(
            desc[c("Depends", "Imports", "Suggests", "Enhances")],
            use.names = FALSE
        ),
        collapse = " "
    )
    expect_false(grepl("faissR", dependency_text, fixed = TRUE))

    runtime_functions <- c(
        "fastembedr_nn_without_self",
        "fastembedr_native_query_knn",
        "fastembedr_gpu_knn_to_host"
    )
    runtime_text <- vapply(runtime_functions, function(name) {
        paste(
            deparse(body(get(name, envir = asNamespace("fastEmbedR")))),
            collapse = "\n"
        )
    }, character(1))
    expect_false(any(grepl("faissR", runtime_text, fixed = TRUE)))
})

test_that("native KNN consumes float32 input without a double input copy", {
    skip_if_not_installed("float")
    set.seed(4)
    x <- float::fl(matrix(rnorm(500 * 8), nrow = 500))
    observed <- native_hnsw_knn_cpp(x, 8L, 2L, "euclidean", 0.99)

    expect_identical(dim(observed$indices), c(500L, 8L))
    expect_true(all(is.finite(observed$distances)))
})

test_that("native Metal exact and IVF searches are recall checked", {
    skip_if_not(
        isTRUE(native_metal_knn_available_cpp()),
        "Metal is unavailable"
    )
    set.seed(4)
    exact_data <- matrix(rnorm(400 * 10), nrow = 400)
    truth <- exact_knn_reference(exact_data, 10L)
    exact <- native_metal_knn_cpp(exact_data, 10L, "exact", "euclidean", 0.99)
    expect_equal(knn_recall_test(exact, truth), 1, tolerance = 1e-8)

    ivf_data <- matrix(rnorm(2000 * 32), nrow = 2000)
    ivf_truth <- exact_knn_reference(ivf_data, 10L)
    ivf <- native_metal_knn_cpp(ivf_data, 10L, "ivf", "euclidean", 0.99)
    expect_true(isTRUE(ivf$target_met))
    expect_gte(knn_recall_test(ivf, ivf_truth), 0.99)
    expect_identical(attr(ivf, "backend"), "metal")
})

test_that("native Metal exact search supports high-dimensional rows", {
    skip_if_not(
        isTRUE(native_metal_knn_available_cpp()),
        "Metal is unavailable"
    )
    set.seed(41)
    high_dimensional <- matrix(rnorm(40 * 1536), nrow = 40)
    truth <- exact_knn_reference(high_dimensional, 6L)
    observed <- native_metal_knn_cpp(
        high_dimensional, 6L, "exact", "euclidean", 1
    )
    expect_equal(knn_recall_test(observed, truth), 1, tolerance = 1e-8)

    query <- high_dimensional[1:5, , drop = FALSE]
    query_result <- native_metal_query_knn_cpp(
        high_dimensional, query, 6L, "exact", "euclidean", 1
    )
    expect_identical(dim(query_result$indices), c(5L, 6L))
    expect_true(all(is.finite(query_result$distances)))
})

test_that("native Metal top-k supports standard t-SNE affinity width", {
    skip_if_not(
        isTRUE(native_metal_knn_available_cpp()),
        "Metal is unavailable"
    )
    set.seed(42)
    reference <- matrix(rnorm(140 * 12), nrow = 140)
    query <- matrix(rnorm(20 * 12), nrow = 20)
    observed <- native_metal_query_knn_cpp(
        reference, query, 90L, "exact", "euclidean", 1
    )
    truth <- t(vapply(seq_len(nrow(query)), function(i) {
        difference <- sweep(reference, 2L, query[i, ], "-")
        distance <- sqrt(rowSums(difference^2))
        order(distance)[seq_len(90L)]
    }, integer(90L)))
    recall <- mean(vapply(seq_len(nrow(query)), function(i) {
        length(intersect(observed$indices[i, ], truth[i, ])) / 90
    }, numeric(1)))

    expect_identical(dim(observed$indices), c(20L, 90L))
    expect_equal(recall, 1, tolerance = 1e-8)
})

test_that("native Metal query IVF is recall-gated against exact query KNN", {
    skip_if_not(
        isTRUE(native_metal_knn_available_cpp()),
        "Metal is unavailable"
    )
    set.seed(940)
    reference <- matrix(rnorm(5000 * 24), nrow = 5000)
    query <- matrix(rnorm(400 * 24), nrow = 400)
    exact <- native_metal_query_knn_cpp(
        reference, query, 15L, "exact", "euclidean", 1
    )
    approximate <- native_metal_query_knn_cpp(
        reference, query, 15L, "ivf", "euclidean", 0.99
    )
    recall <- mean(vapply(seq_len(nrow(query)), function(row) {
        length(intersect(
            exact$indices[row, ],
            approximate$indices[row, ]
        )) / 15
    }, numeric(1)))

    expect_identical(approximate$method, "native_metal_ivf_query")
    expect_true(isTRUE(approximate$target_met))
    expect_gte(recall, 0.99)
    expect_true(all(
        approximate$indices >= 1L &
            approximate$indices <= nrow(reference)
    ))
})

test_that("Metal query routing accounts for the full distance workload", {
    small <- fastembedr_query_nn_policy(
        "metal",
        n_reference = 12000L, n_query = 4000L, p = 64L
    )
    large <- fastembedr_query_nn_policy(
        "metal",
        n_reference = 35000L, n_query = 35000L, p = 784L
    )
    expect_identical(small$method, "exact")
    expect_identical(large$method, "ivf")
    expect_equal(large$target_recall, 0.99)
})

test_that("CUDA query routing accounts for query work", {
    small_query <- fastembedr_query_nn_policy(
        "cuda",
        n_reference = 200000L, n_query = 16L, p = 784L
    )
    low_work <- fastembedr_query_nn_policy(
        "cuda",
        n_reference = 200000L, n_query = 100L, p = 16L
    )
    large_query <- fastembedr_query_nn_policy(
        "cuda",
        n_reference = 200000L, n_query = 10000L, p = 128L
    )
    expect_identical(small_query$method, "exact")
    expect_identical(low_work$method, "exact")
    expect_identical(large_query$method, "ivf")
    expect_equal(large_query$target_recall, 0.99)
})

test_that("native KNN rejects fractional integer controls", {
    x <- matrix(rnorm(80L), nrow = 20L)
    expect_error(
        precompute_knn(x, k = 3.8, backend = "cpu"),
        "one integer"
    )
    expect_error(
        precompute_knn(x, k = 3L, backend = "cpu", n.cores = 1.5),
        "positive integer"
    )
    expect_error(
        fastembedr_native_query_knn(
            x, x[1:2, , drop = FALSE], k = 2.5,
            backend = "cpu"
        ),
        "positive integer"
    )
})

test_that("one-call routing uses native CPU and Metal KNN", {
    expect_identical(
        fastembedr_embedding_nn_policy("cpu", 70000L)$method, "hnsw"
    )
    expect_identical(
        fastembedr_embedding_nn_policy("metal", 70000L)$method, "ivf"
    )
    expect_identical(
        fastembedr_embedding_nn_policy("metal", 1000L)$method, "exact"
    )
    expect_identical(
        fastembedr_embedding_nn_policy("cuda", 70000L)$method, "exact"
    )
    expect_identical(
        fastembedr_embedding_nn_policy("cuda", 100000L)$method, "ivf"
    )
    expected_exact_engine <- if (
        isTRUE(fastembedr_build_config_cpp()$faiss_gpu_compiled)
    ) {
        "native_faiss_gpu_exact"
    } else {
        "native_cuvs_gpu_exact"
    }
    expect_identical(
        fastembedr_nn_policy_engine(
            fastembedr_embedding_nn_policy("cuda", 70000L),
            keep_gpu = TRUE
        ),
        expected_exact_engine
    )
})

test_that("native CUDA KNN never silently falls back", {
    expect_type(native_cuda_knn_available_cpp(), "logical")
    if (!isTRUE(native_cuda_knn_available_cpp())) {
        expect_error(
            fastembedr_nn_without_self(
                matrix(rnorm(64), nrow = 16),
                k = 3L,
                backend = "cuda",
                method = "auto",
                metric = "euclidean",
                keep_gpu = TRUE
            ),
            "Native CUDA KNN"
        )
    }
})

test_that("precompute_knn CUDA output stays resident or fails explicitly", {
    set.seed(26)
    x <- matrix(rnorm(128 * 8), nrow = 128)
    if (!isTRUE(native_cuda_knn_available_cpp())) {
        expect_error(
            precompute_knn(x, k = 8L, backend = "cuda"),
            "Native CUDA KNN"
        )
    } else {
        observed <- precompute_knn(x, k = 8L, backend = "cuda")
        expect_s3_class(observed, "fastEmbedR_gpu_knn")
        expect_s3_class(observed, "fastEmbedR_knn")
        expect_identical(observed$result_residency, "cuda")
        expect_identical(observed$device_to_host_result_copies, 0L)
        expect_false(any(c("indices", "distances") %in% names(observed)))
    }
})

test_that(paste(
    "native CUDA exact KNN remains resident and matches the",
    "exact oracle"
), {
    skip_if_not(
        isTRUE(native_cuda_knn_available_cpp()),
        "native CUDA KNN is unavailable"
    )
    set.seed(4)
    x <- matrix(rnorm(256 * 12), nrow = 256)
    truth <- exact_knn_reference(x, 10L)
    device_knn <- native_cuda_knn_cpp(
        x, 10L, "exact", "euclidean", 0.99,
        keep_gpu = TRUE
    )

    expect_s3_class(device_knn, "fastEmbedR_gpu_knn")
    expect_identical(device_knn$result_residency, "cuda")
    expect_identical(device_knn$layout, "column_major_query_by_k")
    expect_identical(device_knn$distance_type, "float32")
    expect_identical(device_knn$device_to_host_result_copies, 0L)
    expect_false(isTRUE(device_knn$cpu_fallback))
    expect_equal(device_knn$resident_result_bytes, 256 * 10 * 8)
    expect_lt(
        device_knn$peak_temporary_search_bytes,
        256 * 11 * 16
    )

    observed <- native_cuda_knn_to_host_cpp(device_knn)
    expect_identical(dim(observed$indices), c(256L, 10L))
    expect_equal(knn_recall_test(observed, truth), 1, tolerance = 1e-8)
    expect_true(all(is.finite(observed$distances)))
})
