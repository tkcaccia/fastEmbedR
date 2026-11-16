exact_knn_reference <- function(x, k) {
    squared_norm <- rowSums(x * x)
    d2 <- outer(squared_norm, squared_norm, "+") - 2 * tcrossprod(x)
    diag(d2) <- Inf
    indices <- t(vapply(seq_len(nrow(x)), function(i) {
        order(d2[i, ], method = "radix")[seq_len(k)]
    }, integer(k)))
    list(indices = indices, distances = sqrt(pmax(matrix(d2[cbind(
        rep(seq_len(nrow(x)), each = k), as.vector(t(indices))
    )], nrow(x), k, byrow = TRUE), 0)))
}

faissr_exact_knn_test <- function(data, k, n_threads = 1L,
                                    metric = "euclidean",
                                    target_recall = 0.99) {
    run_faissr_cpu_exact(data, NULL, k, metric, "double",
        target_recall, n_threads)
}

faissr_exact_query_test <- function(data, query, k, n_threads = 1L,
                                    metric = "euclidean",
                                    target_recall = 0.99) {
    run_faissr_cpu_exact(data, query, k, metric, "double",
        target_recall, n_threads)
}

knn_recall_test <- function(observed, expected) {
    k <- ncol(expected$indices)
    mean(vapply(seq_len(nrow(expected$indices)), function(i) {
        length(intersect(observed$indices[i, ], expected$indices[i, ])) / k
    }, numeric(1)))
}

test_that("faissR persistent CUDA host output excludes self natively", {
    skip_if_not(fastembedr_cuda_knn_available())
    set.seed(20261115)
    x <- matrix(rnorm(128L * 8L), nrow = 128L)
    index <- faissr_gpu_index_build_cpp(x, 6L, "flat", 0L,
        1L, 1L, FALSE)
    found <- faissr_gpu_index_search_cpp(index$pointer, x, 5L,
        "double_self_exclude")
    truth <- exact_knn_reference(x, 5L)
    expect_identical(found$indices, truth$indices)
    expect_equal(found$distances, truth$distances, tolerance = 1e-5)
    expect_true(found$exclude_self)
    expect_identical(found$backend_used, "faiss_gpu_persistent_query")
    expect_identical(found$devices, 0L)
})

metric_knn_reference <- function(x, k, metric) {
    if (identical(metric, "correlation")) {
        x <- x - rowMeans(x)
    }
    if (!identical(metric, "euclidean")) {
        norm <- sqrt(rowSums(x * x))
        norm[norm == 0] <- 1
        x <- x / norm
    }
    exact_knn_reference(x, k)
}

test_that("faissR HNSW preserves metric tuning and recall", {
    set.seed(45)
    x <- matrix(rnorm(450 * 14), nrow = 450)
    expected <- list(
        euclidean = c(12L, 60L, 45L),
        cosine = c(24L, 160L, 120L),
        correlation = c(48L, 320L, 400L))
    for (metric in names(expected)) {
        truth <- metric_knn_reference(x, 10L, metric)
        observed <- run_faissr_hnsw(
            x, NULL, 10L, metric, "double", 0.99, 2L)
        expect_gte(knn_recall_test(observed, truth), 0.99)
        expect_identical(observed$backend_used, "faiss_hnsw")
        expect_identical(observed$method, "faiss_hnsw")
        expect_identical(c(observed$M, observed$efConstruction,
            observed$efSearch), expected[[metric]])
    }
})

test_that("native CPU exact search matches exhaustive references", {
    set.seed(46)
    x <- matrix(rnorm(180 * 11), nrow = 180)
    for (metric in c("euclidean", "cosine", "correlation")) {
        truth <- metric_knn_reference(x, 9L, metric)
        serial <- faissr_exact_knn_test(x, 9L, 1L, metric, 0.99)
        parallel <- faissr_exact_knn_test(x, 9L, 3L, metric, 0.99)
        expect_equal(knn_recall_test(serial, truth), 1)
        expect_identical(parallel$indices, serial$indices)
        expect_identical(parallel$distances, serial$distances)
        expect_true(serial$exact)
        expect_true(serial$exact_recall_by_construction)
        expect_true(serial$target_met)
        expect_match(serial$backend_used, "^faiss_flat_")
        expect_identical(serial$recall_status, "exact_by_construction")
    }
})

test_that("native CPU exact search handles zero normalized rows", {
    x <- rbind(c(0, 0), c(0, 0), c(1, 0), c(-1, 0))
    observed <- faissr_exact_knn_test(x, 2L, 2L, "cosine", 0.99)

    expect_identical(observed$indices[1L, 1L], 2L)
    expect_equal(observed$distances[1L, ], c(0, 1))
    expect_identical(observed$indices[2L, 1L], 1L)
    expect_equal(observed$distances[2L, ], c(0, 1))
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
    expect_equal(knn_recall_test(observed, truth), 1)
    expect_identical(observed$backend_requested, "cpu")
    expect_identical(observed$execution_backend, "cpu")
    expect_identical(observed$knn_provider, "faissR")
    expect_identical(observed$engine, "faissr_cpu_flat")
    expect_identical(observed$result_residency, "host")
    expect_equal(observed$target_recall, 0.99)
    expect_true(observed$recall_audited)
    expect_true(observed$target_met)
    expect_true(observed$exact)
    expect_identical(
        observed$recall_status,
        "exact_by_construction"
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

test_that("KNN recall tier accepts an explicit argument and session option", {
    x <- matrix(seq_len(80), nrow = 20L)
    withr::local_options(fastEmbedR.target_recall = 0.90)
    for (backend in c("cpu", "cuda", "metal")) {
        expect_equal(fastembedr_embedding_nn_policy(backend, 200000L,
            16L)$target_recall, 0.90)
        expect_equal(fastembedr_query_nn_policy(backend, 200000L,
            1000L, 16L)$target_recall, 0.90)
    }
    option_result <- precompute_knn(x, k = 3L, backend = "cpu")
    expect_equal(option_result$target_recall, 0.90)
    expect_true(option_result$exact)
    explicit <- precompute_knn(x, k = 3L, backend = "cpu",
        target_recall = 0.95)
    expect_equal(explicit$target_recall, 0.95)
    queried <- precompute_query_knn(x[1:15, , drop = FALSE],
        x[16:20, , drop = FALSE], k = 3L, backend = "cpu",
        target_recall = 0.99)
    expect_equal(queried$target_recall, 0.99)
    expect_error(precompute_knn(x, k = 3L, backend = "cpu",
        target_recall = 0.97), "target_recall")
    expect_error(precompute_query_knn(x, x, k = 3L,
        target_recall = NA_real_), "target_recall")
})

test_that("Euclidean IVF audit counts ties and checks distances", {
    x <- rbind(c(0, 0), c(2, 0), c(0, 2), c(4, 0))
    found <- matrix(3L, nrow = 1L)
    truth <- matrix(c(1L, 2L), nrow = 1L)
    audited <- fastembedr_euclidean_tie_audit(
        x, 1L, found, matrix(2, nrow = 1L), truth, 1L)
    expect_equal(audited$mean, 1)
    expect_equal(audited$max_relative_distance_error, 0)
    expect_error(fastembedr_euclidean_tie_audit(
        x, 1L, found, matrix(5, nrow = 1L), truth, 1L),
        "distances disagree")
})

test_that("one-call embedding recall reaches metadata and rejects bypasses", {
    x <- scale(as.matrix(iris[1:40, 1:4]))
    withr::local_options(fastEmbedR.target_recall = 0.90)
    fit_tsne <- tsne(x, perplexity = 5, backend = "cpu", n.cores = 1L,
        early_exaggeration_iter = 0L, n_iter = 1L,
        auto_config = FALSE, target_recall = 0.95)
    expect_equal(fit_tsne$parameters$target_recall, 0.95)
    fit_umap <- umap(x, n_neighbors = 10L, backend = "cpu",
        n.cores = 1L, target_recall = 0.99)
    expect_equal(fit_umap$parameters$target_recall, 0.99)
    option_fit <- umap(x, n_neighbors = 10L, backend = "cpu",
        n.cores = 1L)
    expect_equal(option_fit$parameters$target_recall, 0.90)
    nn <- precompute_knn(x, k = 10L, backend = "cpu")
    supplied <- umap(x, nn = nn, n_neighbors = 10L, n.cores = 1L)
    expect_true(is.na(supplied$parameters$target_recall))
    expect_error(umap(x, nn = nn, n_neighbors = 10L,
        target_recall = 0.95), "Precomputed KNN")
    expect_error(tsne(x, nn = nn, perplexity = 5,
        target_recall = 0.95), "Precomputed KNN")
    landmark <- umap(x, landmarks = 0.5, n_neighbors = 5L,
        target_recall = 0.95)
    expect_identical(landmark$parameters$target_recall, 0.95)
    expect_error(tsne(x, perplexity = 5,
        target_recall = 0.97), "target_recall")
})

test_that("native exact KNN agrees across numeric input storage", {
    skip_if_not_installed("float")
    set.seed(47)
    integers <- matrix(sample.int(255L, 40L * 12L, replace = TRUE), 40L)
    reference <- faissr_exact_knn_test(
        integers, 7L, 1L, "euclidean", 0.99
    )
    inputs <- list(
        matrix(as.numeric(integers), nrow(integers)),
        float::fl(integers)
    )
    for (input in inputs) {
        observed <- faissr_exact_knn_test(
            input, 7L, 1L, "euclidean", 0.99
        )
        expect_identical(observed$indices, reference$indices)
        expect_equal(observed$distances, reference$distances)
    }
})

test_that("precompute_knn exposes no search-algorithm selector", {
    expect_identical(
        names(formals(precompute_knn)),
        c("data", "k", "metric", "backend", "n.cores", "storage",
            "output", "memory_limit", "devices", "checkpoint",
            "resume", "checkpoint_every", "target_recall")
    )
    expect_error(
        precompute_knn(matrix(rnorm(40), nrow = 10), k = 10L),
        "between 1 and nrow\\(data\\) - 1"
    )
})

test_that("precompute_knn Metal output is native or fails explicitly", {
    set.seed(27)
    x <- matrix(rnorm(96 * 6), nrow = 96)
    if (!isTRUE(fastembedr_metal_knn_available())) {
        expect_error(
            precompute_knn(x, k = 7L, backend = "metal"),
            "Native Metal KNN"
        )
    } else {
        observed <- precompute_knn(x, k = 7L, backend = "metal")
        expect_s3_class(observed, "fastEmbedR_knn")
        expect_identical(observed$execution_backend, "metal")
        expect_identical(observed$knn_provider, "faissR")
        expect_identical(observed$result_residency, "host")
        expect_match(observed$method, "native_metal_(exact|ivf)")
    }
})

test_that("native CPU exact query search matches an exhaustive reference", {
    set.seed(47)
    reference <- matrix(rnorm(120 * 8), nrow = 120)
    query <- matrix(rnorm(30 * 8), nrow = 30)
    truth <- test_exact_knn(reference, query, k = 7L)
    observed <- faissr_exact_query_test(
        reference, query, 7L, 3L, "euclidean", 0.99
    )

    expect_identical(observed$indices, truth$indices)
    expect_equal(observed$distances, truth$distances, tolerance = 1e-5)
    expect_identical(observed$method, "faissr_flat_query")
    expect_true(observed$exact_recall_by_construction)
})

test_that("large CPU matrix search uses faissR HNSW without fallback", {
    set.seed(42)
    x <- matrix(rnorm(5000L * 4L), nrow = 5000L)
    observed <- precompute_knn(x, k = 8L, backend = "cpu", n.cores = 2L)
    expect_identical(observed$engine, "faissr_cpu_hnsw")
    expect_identical(observed$backend_used, "faiss_hnsw")
    expect_identical(observed$method, "faiss_hnsw")
    expect_false(any(observed$indices == row(observed$indices)))
    expect_true(observed$recall_audited)
    expect_true(observed$target_met)
    expect_gte(observed$pilot_recall, observed$pilot_guard_recall)
    expect_identical(
        observed$recall_status,
        "pilot_exact_reference_passed"
    )
    query <- precompute_query_knn(
        x, x[1:4, , drop = FALSE], k = 8L,
        backend = "cpu", n.cores = 2L
    )
    expect_identical(query$engine, "faissr_cpu_hnsw")
    expect_identical(query$backend_used, "faiss_hnsw")
    expect_identical(query$method, "faiss_hnsw_query")
    expect_identical(dim(query$indices), c(4L, 8L))
    expect_true(query$recall_audited)
    cosine <- precompute_knn(
        x, k = 8L, metric = "cosine", backend = "cpu", n.cores = 2L
    )
    expect_identical(cosine$engine, "faissr_cpu_hnsw")
    expect_identical(cosine$method, "faiss_hnsw")
    if (requireNamespace("float", quietly = TRUE)) {
        float_knn <- precompute_knn(
            float::fl(x), k = 8L, backend = "cpu", n.cores = 2L
        )
        expect_identical(float_knn$engine, "faissr_cpu_hnsw")
        expect_true(inherits(float_knn$distances, "float32"))
    }
})

test_that("a failed FAISS HNSW call never switches to native HNSW", {
    x <- matrix(rnorm(40), nrow = 10L)
    local_mocked_bindings(
        run_faissr_hnsw = function(...) stop("FAISS HNSW unavailable"),
        .package = "fastEmbedR"
    )
    expect_error(
        fastEmbedR:::fastembedr_nn_without_self(
            x, k = 2L, backend = "cpu", method = "hnsw"
        ),
        "FAISS HNSW unavailable"
    )
})

test_that("native Metal exact and IVF searches are recall checked", {
    skip_if_not(
        isTRUE(fastembedr_metal_knn_available()),
        "Metal is unavailable"
    )
    set.seed(4)
    exact_data <- matrix(rnorm(400 * 10), nrow = 400)
    truth <- exact_knn_reference(exact_data, 10L)
    exact <- faissr_metal_knn_cpp(exact_data, 10L, "exact", "euclidean", 0.99)
    expect_equal(knn_recall_test(exact, truth), 1, tolerance = 1e-8)

    ivf_data <- matrix(rnorm(2000 * 32), nrow = 2000)
    ivf_truth <- exact_knn_reference(ivf_data, 10L)
    ivf <- faissr_metal_knn_cpp(ivf_data, 10L, "ivf", "euclidean", 0.99)
    expect_true(isTRUE(ivf$target_met))
    expect_gte(knn_recall_test(ivf, ivf_truth), 0.99)
    pilot_starts <- round((0:3) * (nrow(ivf_data) - 64L) / 3L)
    pilot_rows <- unlist(lapply(pilot_starts, function(start) {
        seq.int(start + 1L, start + 64L)
    }))
    row_recall <- vapply(pilot_rows, function(i) {
        mean(ivf$indices[i, ] %in% ivf_truth$indices[i, ])
    }, numeric(1L))
    distance_ratio <- vapply(pilot_rows, function(i) {
        max(ivf$distances[i, ]) / max(ivf_truth$distances[i, ])
    }, numeric(1L))
    expect_equal(ivf$pilot_min_recall, min(row_recall))
    expect_equal(
        ivf$pilot_max_k_distance_ratio, max(1, distance_ratio),
        tolerance = 1e-4
    )
    expect_identical(ivf$accelerator, "metal")
    expect_identical(ivf$backend_used, "faissr_native_metal_ivf")
})

test_that("Metal recall tiers distinguish pilots from exact search", {
    skip_if_not(fastembedr_metal_knn_available(),
        "Metal is unavailable")
    set.seed(164)
    x <- matrix(rnorm(50001L * 8L), ncol = 8L)
    for (tier in c(0.90, 0.95, 0.99)) {
        found <- precompute_knn(x, k = 12L, backend = "metal",
            target_recall = tier)
        expect_identical(found$target_recall, tier)
        expect_identical(found$backend_used,
            "faissr_native_metal_ivf")
        expect_true(found$recall_audited)
        expect_false(found$exact_recall_by_construction)
        expect_identical(found$recall_status,
            "runtime_audited_target_met")
        expect_gte(found$pilot_recall, tier)
    }
    exact <- precompute_knn(x[1:64, , drop = FALSE], k = 12L,
        backend = "metal", target_recall = 0.90)
    expect_identical(exact$recall_status, "exact_by_construction")
    expect_true(exact$exact_recall_by_construction)
    expect_false(exact$recall_audited)
    expect_true(is.na(exact$pilot_recall))
})

test_that("Metal recall validation rejects misses and changed targets", {
    out <- list(method = "native_metal_ivf",
        backend_used = "faissr_native_metal_ivf",
        accelerator = "metal", target_recall = 0.95,
        pilot_recall = 0.90, target_met = TRUE)
    expect_error(fastEmbedR:::validate_faissr_metal_knn(out, 0.95),
        "did not meet")
    out$pilot_recall <- 0.97
    out$target_met <- FALSE
    expect_error(fastEmbedR:::validate_faissr_metal_knn(out, 0.95),
        "did not meet")
    out$target_met <- TRUE
    expect_error(fastEmbedR:::validate_faissr_metal_knn(out, 0.99),
        "changed the backend or recall target")
})

test_that("native Metal exact search supports high-dimensional rows", {
    skip_if_not(
        isTRUE(fastembedr_metal_knn_available()),
        "Metal is unavailable"
    )
    set.seed(41)
    high_dimensional <- matrix(rnorm(40 * 1536), nrow = 40)
    truth <- exact_knn_reference(high_dimensional, 6L)
    observed <- faissr_metal_knn_cpp(
        high_dimensional, 6L, "exact", "euclidean", 1
    )
    expect_equal(knn_recall_test(observed, truth), 1, tolerance = 1e-8)

    query <- high_dimensional[1:5, , drop = FALSE]
    query_result <- faissr_metal_query_knn_cpp(
        high_dimensional, query, 6L, "exact", "euclidean", 1
    )
    expect_identical(dim(query_result$indices), c(5L, 6L))
    expect_true(all(is.finite(query_result$distances)))
})

test_that("Metal query bridge rejects changed method and backend", {
    valid <- list(
        indices = matrix(1L, 2L, 1L),
        distances = matrix(0, 2L, 1L),
        backend = "metal", accelerator = "metal",
        backend_used = "faissr_native_metal_exact_query",
        method = "native_metal_exact_query", index_base = 1L,
        metric = "euclidean", target_recall = 0.99,
        n_reference = 3L, n_query = 2L,
        exclude_self = FALSE, exact = TRUE, target_met = TRUE
    )
    check <- fastEmbedR:::fastembedr_check_metal_query_knn
    expect_invisible(check(valid, 3L, 2L, 1L, "exact",
        "euclidean", 0.99))
    changed <- valid
    changed$backend_used <- "faiss_cpu_flat_query"
    expect_error(check(changed, 3L, 2L, 1L, "exact",
        "euclidean", 0.99), "requested Metal")
    changed <- valid
    changed$method <- "native_metal_ivf_query"
    changed$backend_used <- "faissr_native_metal_ivf_query"
    changed$exact <- FALSE
    expect_error(check(changed, 3L, 2L, 1L, "exact",
        "euclidean", 0.99), "requested Metal")
    changed <- valid
    changed$target_met <- FALSE
    expect_error(check(changed, 3L, 2L, 1L, "exact",
        "euclidean", 0.99), "recall target")
})

test_that("native Metal top-k supports standard t-SNE affinity width", {
    skip_if_not(
        isTRUE(fastembedr_metal_knn_available()),
        "Metal is unavailable"
    )
    set.seed(42)
    reference <- matrix(rnorm(140 * 12), nrow = 140)
    query <- matrix(rnorm(20 * 12), nrow = 20)
    observed <- faissr_metal_query_knn_cpp(
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
        isTRUE(fastembedr_metal_knn_available()),
        "Metal is unavailable"
    )
    set.seed(940)
    reference <- matrix(rnorm(5000 * 24), nrow = 5000)
    query <- matrix(rnorm(400 * 24), nrow = 400)
    exact <- faissr_metal_query_knn_cpp(
        reference, query, 15L, "exact", "euclidean", 1
    )
    approximate <- faissr_metal_query_knn_cpp(
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
    expect_false(
        "faiss_gpu_compiled" %in% names(fastembedr_build_config_cpp())
    )
    expect_identical(
        fastembedr_embedding_nn_policy("cpu", 70000L)$method, "hnsw"
    )
    expect_identical(
        fastembedr_embedding_nn_policy("cpu", 4999L)$method, "exact"
    )
    expect_identical(
        fastembedr_embedding_nn_policy("cpu", 5000L)$method, "hnsw"
    )
    expect_identical(
        fastembedr_query_nn_policy("cpu", 4999L, 10L, 4L)$method,
        "exact"
    )
    expect_identical(
        fastembedr_query_nn_policy("cpu", 5000L, 10L, 4L)$method,
        "hnsw"
    )
    expect_identical(
        fastembedr_embedding_nn_policy("metal", 70000L)$method, "ivf"
    )
    expect_identical(
        fastembedr_embedding_nn_policy("metal", 1000L)$method, "exact"
    )
    expect_identical(
        fastembedr_embedding_nn_policy("metal", 44808L, 50L)$method,
        "exact"
    )
    expect_identical(
        fastembedr_embedding_nn_policy("metal", 5000L, 784L)$method,
        "exact"
    )
    expect_identical(
        fastembedr_embedding_nn_policy("metal", 15000L, 784L)$method,
        "ivf"
    )
    expect_identical(
        fastembedr_embedding_nn_policy("metal", 70000L, 784L)$method,
        "ivf"
    )
    expect_identical(
        fastembedr_embedding_nn_policy("cuda", 70000L)$method, "exact"
    )
    expect_identical(
        fastembedr_embedding_nn_policy("cuda", 100000L)$method, "ivf"
    )
    expect_identical(
        fastembedr_nn_policy_engine(
            fastembedr_embedding_nn_policy("cuda", 70000L),
            keep_gpu = TRUE
        ),
        "faissr_cuda_gpu_exact"
    )
})

test_that("CUDA resident KNN never silently falls back", {
    expect_type(fastembedr_cuda_knn_available(), "logical")
    if (!isTRUE(fastembedr_cuda_knn_available()) &&
        !isTRUE(faissR::faiss_gpu_available())) {
        expect_error(
            fastembedr_nn_without_self(
                matrix(rnorm(64), nrow = 16),
                k = 3L,
                backend = "cuda",
                method = "auto",
                metric = "euclidean",
                keep_gpu = TRUE
            ),
            "faissR CUDA KNN is unavailable"
        )
    }
})

test_that("CUDA embedding KNN uses the faissR GPU provider", {
    skip_if_not(isTRUE(faissR::cuda_available()) &&
        isTRUE(faissR::faiss_gpu_available()))
    set.seed(817)
    x <- matrix(rnorm(200L * 16L), nrow = 200L)
    truth <- exact_knn_reference(x, 8L)
    for (method in c("exact", "ivf")) {
        out <- fastEmbedR:::run_faissr_cuda_embedding_knn(
            x, 8L, method, "euclidean", "float", 0.99
        )
        expected_backend <- if (method == "ivf") {
            "faiss_gpu_persistent_query"
        } else "faiss_gpu_single_self"
        expect_identical(out$backend_used, expected_backend)
        expect_identical(out$multi_gpu_mode, "single")
        expect_identical(out$result_residency, "host")
        expect_identical(dim(out$indices), c(200L, 8L))
        expect_false(any(out$indices == row(out$indices)))
        expect_false(isTRUE(out$cpu_fallback))
        if (method == "exact") {
            expect_equal(knn_recall_test(out, truth), 1)
        } else {
            expect_true(out$recall_audited)
            expect_gte(out$pilot_recall, 0.99)
        }
    }
    expect_error(
        fastEmbedR:::run_faissr_cuda_embedding_knn(
            x, 8L, "ivf", "euclidean", "float", 0.99,
            devices = c(0L, 1L)
        ),
        "device"
    )
})

test_that("precompute_knn CUDA output stays resident or fails explicitly", {
    set.seed(26)
    x <- matrix(rnorm(128 * 8), nrow = 128)
    if (!isTRUE(faissR::cuda_available())) {
        expect_error(
            precompute_knn(x, k = 8L, backend = "cuda"),
            "CUDA|device"
        )
    } else {
        observed <- precompute_knn(x, k = 8L, backend = "cuda")
        expect_s3_class(observed, "fastEmbedR_gpu_knn")
        expect_s3_class(observed, "fastEmbedR_knn")
        expect_identical(observed$result_residency, "cuda")
        expect_identical(observed$knn_provider, "faissR")
        expect_identical(observed$device_to_host_result_copies, 0L)
        expect_false(any(c("indices", "distances") %in% names(observed)))
    }
})

test_that("resident precompute KNN forwards explicit CUDA devices", {
    x <- matrix(seq_len(20), nrow = 5L)
    expect_error(precompute_knn(x, 2L, backend = "cpu", devices = 0L),
        "requires CUDA")
    with_mocked_bindings(
        massive_validate_devices = function(devices, backend, nrow) {
            expect_identical(backend, "cuda")
            expect_identical(devices, c(0L, 1L))
            expect_identical(nrow, 5L)
            devices
        },
        run_faissr_cuda_multi_knn = function(data, k, method, metric,
                output, target_recall, devices) {
            expect_identical(devices, c(0L, 1L))
            expect_identical(target_recall, 0.95)
            list(indices = matrix(2L, 5L, k),
                distances = matrix(1, 5L, k),
                backend_used = "faiss_gpu_multi_replicate",
                devices = devices, gpu_count = 2L,
                multi_gpu_mode = "replicate", cpu_fallback = FALSE)
        },
        {
            observed <- precompute_knn(x, 2L, backend = "cuda",
                devices = c(0L, 1L), target_recall = 0.95)
            expect_s3_class(observed, "fastEmbedR_knn")
            expect_false(inherits(observed, "fastEmbedR_gpu_knn"))
            expect_identical(observed$result_residency, "host")
            expect_identical(observed$knn_provider, "faissR")
            expect_identical(observed$devices, c(0L, 1L))
            expect_identical(observed$target_recall, 0.95)
        }, .package = "fastEmbedR")
})

test_that("host IVF output gets exact-reference recall and distance audit", {
    set.seed(213L)
    x <- matrix(rnorm(40L * 6L), nrow = 40L)
    found <- faissR::nn(x, k = 5L, exclude_self = TRUE,
        backend = "cpu", method = "flat", metric = "euclidean")
    found$result_residency <- "host"
    audit <- fastEmbedR:::fastembedr_audit_resident_ivf(
        x, found, 5L, "euclidean", 0.99)
    expect_equal(audit$mean, 1)
    expect_equal(audit$tie$mean, 1)
    expect_lte(audit$tie$max_relative_distance_error, 1e-5)
})

test_that("host CUDA IVF recall metadata rejects an audited miss", {
    x <- matrix(1:20, nrow = 5L)
    local_mocked_bindings(
        fastembedr_audit_resident_ivf = function(...) list(
            rows = 1:2, mean = 0.97, minimum = 0.9,
            reference_backend = "faiss_flat_l2",
            tie = list(mean = 0.98)),
        .package = "fastEmbedR")
    exact <- fastEmbedR:::fastembedr_cuda_multi_recall_result(
        list(), x, 2L, "flat", "euclidean", 0.95)
    expect_identical(exact$recall_status, "exact_by_construction")
    expect_false(exact$recall_audited)
    expect_true(exact$target_met)
    passing <- fastEmbedR:::fastembedr_cuda_multi_recall_result(
        list(), x, 2L, "ivf", "euclidean", 0.95)
    expect_equal(passing$pilot_recall, 0.97)
    expect_true(passing$recall_audited)
    expect_identical(passing$recall_status, "runtime_audited_target_met")
    expect_error(fastEmbedR:::fastembedr_cuda_multi_recall_result(
        list(), x, 2L, "ivf", "euclidean", 0.99),
        "below target")
})

test_that("CUDA query IVF requires a measured passing faissR pilot", {
    out <- list(accelerator = "cuda", backend_used = "faiss_gpu_ivf_flat",
        indices = matrix(1L, 2L, 1L), index_base = 1L,
        cpu_fallback = FALSE, target_met = TRUE,
        pilot_recall = 0.96)
    checked <- fastEmbedR:::fastembedr_check_cuda_query_knn(
        out, 2L, 1L, "ivf", 0.95)
    expect_identical(attr(checked, "backend"), "cuda")
    expect_identical(attr(checked, "exclude_self"), FALSE)
    expect_error(fastEmbedR:::fastembedr_check_cuda_query_knn(
        out, 2L, 1L, "ivf", 0.99), "missed")
    out$pilot_recall <- NULL
    expect_error(fastEmbedR:::fastembedr_check_cuda_query_knn(
        out, 2L, 1L, "ivf", 0.95), "missed")
})

test_that("CUDA query KNN has no fastEmbedR native implementation", {
    expect_false(exists(
        "native_cuda_query_knn_cpp", asNamespace("fastEmbedR"),
        inherits = FALSE
    ))
    expect_true(exists(
        "run_faissr_cuda_query_knn", asNamespace("fastEmbedR"),
        inherits = FALSE
    ))
    expect_false(exists(
        "native_cuda_index_build_cpp", asNamespace("fastEmbedR"),
        inherits = FALSE
    ))
    expect_false(exists(
        "native_cuda_index_search_cpp", asNamespace("fastEmbedR"),
        inherits = FALSE
    ))
    for (name in c(
        "native_cuda_knn_cpp",
        "native_cuda_knn_to_host_cpp",
        "native_cuda_knn_available_cpp"
    )) {
        expect_false(exists(name, asNamespace("fastEmbedR"),
            inherits = FALSE))
    }
})
