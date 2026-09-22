test_that("auto_k chooses bounded size-aware neighborhoods", {
    expect_equal(fastEmbedR:::auto_k(100L), 15L)
    expect_equal(fastEmbedR:::auto_k(100L, include_self = TRUE), 16L)
    expect_equal(fastEmbedR:::auto_k(1000L), 30L)
    expect_equal(fastEmbedR:::auto_k(20000L), 50L)
    expect_equal(fastEmbedR:::auto_k(matrix(0, 8L, 3L)), 7L)
    expect_error(fastEmbedR:::auto_k(1L), "at least two")
})

test_that("automatic embedding K is openTSNE focused", {
    expect_equal(fastEmbedR:::auto_embedding_k(1000L, "tsne"), 30L)
    expect_equal(fastEmbedR:::auto_embedding_k(1000L, include_self = TRUE), 31L)
})

test_that("embedding metrics expose the supported native choices", {
    expect_equal(
        fastEmbedR:::resolve_embedding_metric("euclidean"),
        "euclidean"
    )
    expect_equal(fastEmbedR:::resolve_embedding_metric("cosine"), "cosine")
    expect_equal(
        fastEmbedR:::resolve_embedding_metric("correlation"),
        "correlation"
    )
    expect_equal(
        fastEmbedR:::resolve_embedding_metric("inner_product"),
        "inner_product"
    )
    expect_error(fastEmbedR:::resolve_embedding_metric("manhattan"), "arg")
})

test_that("preprocessing PCA uses package-native RSVD", {
    set.seed(39)
    x <- matrix(rnorm(80L * 40L), 80L, 40L)
    pre <- fastEmbedR:::prepare_embedding_data(
        x,
        standardize = TRUE,
        pca_dims = 5L,
        seed = 39L,
        backend = "cpu"
    )

    expect_equal(dim(pre$data), c(80L, 5L))
    expect_true(all(is.finite(pre$data)))
    expect_equal(pre$preprocess$pca_method, "rsvd")
    expect_equal(pre$preprocess$pca_backend, "cpu_rsvd")
})

test_that("explicit CUDA PCA never falls back to CPU", {
    if (isTRUE(fastEmbedR:::embedding_cuda_available_cpp())) {
        skip("CUDA is available; the hardware test covers the native path.")
    }
    x <- matrix(rnorm(120L), 30L, 4L)
    expect_error(
        pca(x, ncomp = 2L, backend = "cuda"),
        "CUDA|RAFT"
    )
    expect_error(
        fastEmbedR:::prepare_embedding_data(
            x,
            standardize = FALSE,
            pca_dims = 2L,
            seed = 4L,
            backend = "cuda"
        ),
        "CUDA|RAFT"
    )
})

test_that(paste(
    "float32 matrix input is accepted and preserved without",
    "preprocessing"
), {
    skip_if_not_installed("float")
    set.seed(41)
    x <- float::fl(matrix(rnorm(60L), 20L, 3L))
    pre <- fastEmbedR:::prepare_embedding_data(
        x,
        standardize = FALSE,
        pca_dims = NULL,
        seed = 41L,
        backend = "cpu"
    )

    expect_s4_class(pre$data, "float32")
    expect_equal(dim(pre$data), c(20L, 3L))
    expect_equal(pre$preprocess$preprocess_backend, "none")
    expect_match(
        pre$preprocess$preprocess_backend_reason,
        "float32_input_preserved"
    )

    x_bad <- x
    x_bad[1, 1] <- Inf
    expect_error(
        fastEmbedR:::prepare_embedding_data(
            x_bad,
            standardize = FALSE,
            pca_dims = NULL,
            seed = 41L,
            backend = "cpu"
        ),
        "finite"
    )
})

test_that("float32 standardization stays float32 and matches scale", {
    skip_if_not_installed("float")
    set.seed(43)
    x <- matrix(rnorm(400L), 100L, 4L)
    x_float <- float::fl(x)

    pre <- fastEmbedR:::prepare_embedding_data(
        x_float,
        standardize = TRUE,
        pca_dims = NULL,
        seed = 43L,
        backend = "cpu"
    )

    expect_s4_class(pre$data, "float32")
    expect_equal(dim(pre$data), dim(x))
    expect_equal(
        as.vector(float::dbl(pre$data)),
        as.vector(scale(x)),
        tolerance = 1e-6
    )
    expect_equal(pre$preprocess$preprocess_backend, "cpu_float32")
    expect_equal(
        pre$preprocess$preprocess_backend_reason,
        "native_float32_column_standardization"
    )
})

test_that("float32 landmark partition preserves row order and values", {
    skip_if_not_installed("float")
    x <- matrix(seq_len(60), nrow = 10, ncol = 6)
    x_float <- float::fl(x)
    landmark_rows <- c(8L, 2L, 5L, 1L)
    query_rows <- setdiff(seq_len(nrow(x)), landmark_rows)

    split <- split_float32_rows_cpp(
        x_float,
        landmark_rows,
        query_rows,
        n_threads = 2L
    )

    expect_s4_class(split$landmarks, "float32")
    expect_s4_class(split$query, "float32")
    expect_equal(float::dbl(split$landmarks), x[landmark_rows, , drop = FALSE])
    expect_equal(float::dbl(split$query), x[query_rows, , drop = FALSE])
})

test_that("float32 landmark projections avoid full double materialization", {
    skip_if_not_installed("float")
    set.seed(45)
    x <- matrix(rnorm(1600), nrow = 100, ncol = 16)
    x_float <- float::fl(x)
    directions <- matrix(rnorm(16 * 4), nrow = 16, ncol = 4)
    directions <- sweep(
        directions,
        2L,
        sqrt(colSums(directions * directions)),
        "/"
    )

    projected <- landmark_projection_float32_cpp(
        x_float,
        directions,
        n_direct = 4L,
        n_threads = 2L
    )
    reference <- cbind(x[, 1:4, drop = FALSE], x %*% directions)

    expect_type(projected, "double")
    expect_equal(projected, reference, tolerance = 1e-6)
})

test_that("tsne and umap accept float32 matrix input", {
    skip_if_not_installed("float")
    set.seed(42)
    x <- float::fl(matrix(rnorm(120L), 30L, 4L))

    fit_tsne <- tsne(
        x,
        perplexity = 1,
        early_exaggeration_iter = 2L,
        n_iter = 3L,
        seed = 42L,
        n.cores = 2L
    )
    expect_s3_class(fit_tsne, "fastEmbedR_embedding")
    expect_equal(dim(fit_tsne$layout), c(30L, 2L))
    expect_s4_class(fit_tsne$layout, "float32")
    expect_equal(fit_tsne$preprocess$preprocess_backend, "none")

    fit_umap <- umap(
        x,
        n_neighbors = 5L,
        seed = 42L,
        n.cores = 2L
    )
    expect_s3_class(fit_umap, "fastEmbedR_embedding")
    expect_equal(dim(fit_umap$layout), c(30L, 2L))
    expect_s4_class(fit_umap$layout, "float32")
    expect_equal(fit_umap$parameters$preprocess$preprocess_backend, "none")
})

test_that(paste(
    "double matrix input keeps a double layout despite float",
    "internal KNN"
), {
    set.seed(46)
    x <- matrix(rnorm(120L), 30L, 4L)

    fit_tsne <- tsne(
        x,
        perplexity = 1,
        early_exaggeration_iter = 2L,
        n_iter = 3L,
        seed = 46L,
        n.cores = 2L
    )
    expect_type(fit_tsne$layout, "double")
    expect_false(inherits(fit_tsne$layout, "float32"))

    fit_umap <- umap(
        x,
        n_neighbors = 5L,
        seed = 46L,
        n.cores = 2L
    )
    expect_type(fit_umap$layout, "double")
    expect_false(inherits(fit_umap$layout, "float32"))
})

test_that(paste(
    "layout finalizer preserves requested type and float32",
    "memory footprint"
), {
    skip_if_not_installed("float")
    set.seed(48)
    raw_layout <- matrix(rnorm(20000L), 10000L, 2L)

    float_layout <- fastEmbedR:::finalize_embedding_layout(
        raw_layout,
        "TSNE",
        return_float32 = TRUE
    )
    double_layout <- fastEmbedR:::finalize_embedding_layout(
        float_layout,
        "TSNE",
        return_float32 = FALSE
    )

    expect_s4_class(float_layout, "float32")
    expect_type(double_layout, "double")
    expect_equal(dim(float_layout), dim(double_layout))
    expect_equal(colnames(float_layout), c("TSNE1", "TSNE2"))
    expect_equal(colnames(double_layout), c("TSNE1", "TSNE2"))
    expect_equal(attr(float_layout, "precision"), "float32")
    expect_equal(attr(double_layout, "precision"), "double")

    float_bytes <- as.numeric(object.size(float_layout))
    double_bytes <- as.numeric(object.size(double_layout))
    expect_lt(float_bytes, double_bytes)
    expect_lt(float_bytes / double_bytes, 0.65)
})

test_that("float32 layouts are decoded before plotting and scoring", {
    skip_if_not_installed("float")
    expected <- matrix(c(1, 2, 3, 4, 5, 6), 3L, 2L)
    layout <- float::fl(expected)

    decoded <- fastEmbedR:::embedding_dense_double_matrix(layout)

    expect_type(decoded, "double")
    expect_equal(decoded, expected, tolerance = 1e-7)
})

test_that("float32 finiteness validation avoids coercion", {
    skip_if_not_installed("float")
    x <- float::fl(matrix(c(1, 2, 3, 4), nrow = 2L))
    expect_true(float32_all_finite_cpp(x))
    bad <- float::fl(matrix(c(1, 2, Inf, 4), nrow = 2L))
    expect_false(float32_all_finite_cpp(bad))
    expect_s4_class(prepare_embedding_data(x, FALSE, NULL, 1L)$data, "float32")
})

test_that("float32 KNN bridge passes float32 data directly to native HNSW", {
    skip_if_not_installed("float")
    set.seed(44)
    x <- float::fl(matrix(rnorm(40L), 10L, 4L))

    out <- fastEmbedR:::fastembedr_nn_without_self(
        x,
        k = 3L,
        backend = "cpu",
        output = "float",
        n_threads = 2L
    )

    expect_equal(dim(out$indices), c(10L, 3L))
    expect_s4_class(out$distances, "float32")
    expect_identical(attr(out, "backend"), "cpu")
    expect_identical(attr(out, "method"), "native_hnsw")
})

test_that("one-call KNN policy selects native CPU and Metal defaults", {
    expect_equal(
        fastEmbedR:::fastembedr_embedding_nn_policy("cpu", n = 1000L),
        list(
            backend = "cpu", method = "hnsw", tuning = "auto",
            target_recall = NA_real_,
            recall_status = "not_audited_fixed_heuristic"
        )
    )
    expect_equal(
        fastEmbedR:::fastembedr_embedding_nn_policy("metal", n = 1000L),
        list(
            backend = "metal", method = "exact", tuning = "auto",
            target_recall = 0.99
        )
    )
    expect_equal(
        fastEmbedR:::fastembedr_embedding_nn_policy("cpu", n = 200000L),
        list(
            backend = "cpu", method = "hnsw", tuning = "auto",
            target_recall = NA_real_,
            recall_status = "not_audited_fixed_heuristic"
        )
    )
    expect_equal(
        fastEmbedR:::fastembedr_embedding_nn_policy("metal", n = 200000L),
        list(
            backend = "metal", method = "ivf", tuning = "auto",
            target_recall = 0.99
        )
    )
    expect_equal(
        fastEmbedR:::fastembedr_embedding_nn_policy("cuda", n = 1000L),
        list(
            backend = "cuda", method = "exact", tuning = "auto",
            target_recall = 0.99
        )
    )
})

test_that("CPU matrix input uses package-native HNSW only", {
    set.seed(45)
    x <- matrix(rnorm(40L), 10L, 4L)
    out <- fastEmbedR:::fastembedr_nn_without_self(
        x,
        k = 2L,
        backend = "cpu",
        method = "hnsw",
        tuning = "auto",
        target_recall = 0.99
    )

    expect_equal(dim(out$indices), c(10L, 2L))
    expect_identical(attr(out, "backend"), "cpu")
    expect_identical(attr(out, "method"), "native_hnsw")
    expect_error(
        fastEmbedR:::fastembedr_nn_without_self(
            x,
            k = 2L, backend = "cpu", method = "exact"
        ),
        "HNSW"
    )
})

test_that("CUDA KNN bridge consumes native GPU-resident output", {
    set.seed(47)
    x <- matrix(rnorm(40L), 10L, 4L)
    captured <- new.env(parent = emptyenv())
    native_mock <- function(data, k, method, metric, target_recall, keep_gpu) {
        captured$called <- TRUE
        captured$k <- k
        captured$method <- method
        captured$metric <- metric
        captured$target_recall <- target_recall
        captured$keep_gpu <- keep_gpu
        structure(
            list(
                handle = "mock",
                backend_used = "native_cuda_faiss_gpu_bfknn_l2",
                gpu_provider = "fastEmbedR_native_faiss_gpu",
                metric = metric,
                n_query = nrow(data),
                k = k,
                exclude_self = TRUE,
                result_residency = "cuda",
                distance_type = "float32",
                layout = "column_major_query_by_k"
            ),
            class = "fastEmbedR_gpu_knn"
        )
    }
    with_mocked_bindings(
        native_cuda_knn_available_cpp = function() TRUE,
        native_cuda_faiss_gpu_available_cpp = function() TRUE,
        native_cuda_knn_cpp = native_mock,
        {
            out <- fastEmbedR:::fastembedr_nn_without_self(
                x,
                k = 2L,
                backend = "cuda",
                method = "auto",
                metric = "correlation",
                target_recall = 0.99,
                keep_gpu = TRUE
            )

            out_gpu <- fastEmbedR:::fastembedr_nn_without_self(
                x,
                k = 2L,
                backend = "cuda",
                method = "auto",
                metric = "inner_product",
                target_recall = 0.99,
                keep_gpu = TRUE
            )
        },
        .package = "fastEmbedR"
    )

    expect_true(captured$called)
    expect_true(captured$keep_gpu)
    expect_equal(captured$method, "auto")
    expect_equal(captured$target_recall, 0.99)
    expect_s3_class(out, "fastEmbedR_gpu_knn")
    expect_equal(out$backend_used, "native_cuda_faiss_gpu_bfknn_l2")
    expect_equal(out$metric, "correlation")
    expect_s3_class(out_gpu, "fastEmbedR_gpu_knn")
    expect_equal(out_gpu$metric, "inner_product")
})

test_that("external GPU KNN objects require explicit user materialization", {
    gpu_knn <- structure(
        list(
            indices_ptr = new.env(parent = emptyenv()),
            distances_ptr = new.env(parent = emptyenv()),
            backend_used = "external_cuda",
            result_residency = "cuda",
            metric = "inner_product",
            n_query = 4L,
            k = 2L
        ),
        class = "external_gpu_knn"
    )

    expect_error(
        fastEmbedR:::coerce_knn_input(gpu_knn),
        "does not call another package"
    )
})

test_that("CUDA openTSNE keeps native fastEmbedR GPU KNN on device", {
    captured <- new.env(parent = emptyenv())
    gpu_knn <- structure(
        list(
            handle = "mock",
            backend_used = "cuda_auto",
            result_residency = "cuda",
            distance_type = "float32",
            metric = "euclidean",
            layout = "column_major_query_by_k",
            exclude_self = TRUE,
            n_query = 8L,
            k = 9L
        ),
        class = "fastEmbedR_gpu_knn"
    )
    with_mocked_bindings(
        tsne_knn = function(indices, distances = NULL,
                            n_neighbors = NULL, ...) {
            captured$gpu_input <- fastEmbedR:::fastembedr_is_gpu_knn(indices)
            captured$distances_null <- is.null(distances)
            out <- matrix(0, 8L, 2L)
            attr(out, "fastEmbedR_config") <- list(
                n_neighbors = 9L,
                perplexity = 3,
                knn_residency = "cuda_device"
            )
            out
        },
        {
            fit <- tsne(
                gpu_knn, backend = "cuda", perplexity = 3,
                keep_knn = TRUE
            )
        },
        .package = "fastEmbedR"
    )

    expect_true(captured$gpu_input)
    expect_true(captured$distances_null)
    expect_s3_class(fit, "fastEmbedR_embedding")
    expect_s3_class(fit$knn, "fastEmbedR_gpu_knn")
    expect_equal(
        fit$parameters$preprocess,
        "none_precomputed_gpu_knn_cuda_random_init"
    )
})

test_that("CUDA UMAP keeps native fastEmbedR GPU KNN on device", {
    captured <- new.env(parent = emptyenv())
    x <- matrix(stats::rnorm(32L), 8L, 4L)
    gpu_knn <- structure(
        list(
            handle = "mock",
            backend_used = "cuda_auto",
            result_residency = "cuda",
            distance_type = "float32",
            metric = "euclidean",
            layout = "column_major_query_by_k",
            exclude_self = TRUE,
            n_query = 8L,
            k = 3L
        ),
        class = "fastEmbedR_gpu_knn"
    )
    with_mocked_bindings(
        fast_knn_umap = function(indices, distances = NULL, ...) {
            captured$gpu_input <- fastEmbedR:::fastembedr_is_gpu_knn(indices)
            captured$distances_null <- is.null(distances)
            out <- matrix(0, 8L, 2L)
            attr(out, "fastEmbedR_config") <- list(
                knn_n_neighbors = 3L,
                n_neighbors = 3L,
                knn_residency = "cuda_device",
                graph_storage = "native_cuda_device_coo_fused"
            )
            out
        },
        {
            fit <- umap(x, nn = gpu_knn, backend = "cuda", keep_knn = TRUE)
        },
        .package = "fastEmbedR"
    )

    expect_true(captured$gpu_input)
    expect_true(captured$distances_null)
    expect_s3_class(fit, "fastEmbedR_embedding")
    expect_s3_class(fit$knn, "fastEmbedR_gpu_knn")
    expect_equal(fit$parameters$knn_residency, "cuda_device")
})

test_that("one-call CUDA embeddings use the intended KNN residency policy", {
    x <- matrix(stats::rnorm(32L), 8L, 4L)
    y_init <- matrix(0, 8L, 2L)
    gpu_knn <- structure(
        list(
            handle = "mock",
            backend_used = "cuda_auto",
            result_residency = "cuda",
            distance_type = "float32",
            metric = "euclidean",
            layout = "column_major_query_by_k",
            exclude_self = TRUE,
            n_query = 8L,
            k = 6L
        ),
        class = "fastEmbedR_gpu_knn"
    )
    captured <- new.env(parent = emptyenv())
    captured$keep_gpu <- list()
    with_mocked_bindings(
        fastembedr_nn_without_self = function(
                data, k, backend, method, metric, output,
                n_threads, tuning, target_recall, keep_gpu) {
            captured$keep_gpu[[length(captured$keep_gpu) + 1L]] <- keep_gpu
            expect_identical(backend, "cuda")
            if (isTRUE(keep_gpu)) {
                gpu_knn
            } else {
                structure(
                    list(
                        indices = matrix(rep(1:3, each = 8L), 8L, 3L),
                        distances = matrix(1, 8L, 3L),
                        backend_used = "cuda_host_exact"
                    ),
                    class = "list"
                )
            }
        },
        fast_knn_umap = function(indices, distances = NULL, ...) {
            captured$umap_gpu_input <-
                fastEmbedR:::fastembedr_is_gpu_knn(indices)
            captured$umap_distances_null <- is.null(distances)
            out <- matrix(0, 8L, 2L)
            attr(out, "fastEmbedR_config") <- list(
                knn_n_neighbors = 3L,
                n_neighbors = 3L,
                knn_residency = "host_materialized_from_cuda",
                graph_storage = "cpu_binary_csr"
            )
            out
        },
        tsne_knn = function(indices, distances = NULL, ...) {
            captured$tsne_gpu_input <-
                fastEmbedR:::fastembedr_is_gpu_knn(indices)
            captured$tsne_distances_null <- is.null(distances)
            out <- matrix(0, 8L, 2L)
            attr(out, "fastEmbedR_config") <- list(
                n_neighbors = 6L,
                perplexity = 2,
                knn_residency = "cuda_device"
            )
            out
        },
        {
            fit_umap <- umap(
                x, n_neighbors = 3L, backend = "cuda",
                keep_knn = TRUE
            )
            fit_tsne <- tsne(
                x, perplexity = 2, Y_init = y_init,
                backend = "cuda", keep_knn = TRUE
            )
        },
        .package = "fastEmbedR"
    )

    expect_equal(captured$keep_gpu, list(TRUE, TRUE))
    expect_true(captured$umap_gpu_input)
    expect_true(captured$umap_distances_null)
    expect_true(captured$tsne_gpu_input)
    expect_true(captured$tsne_distances_null)
    expect_s3_class(fit_umap$knn, "fastEmbedR_gpu_knn")
    expect_s3_class(fit_tsne$knn, "fastEmbedR_gpu_knn")
})

test_that("float32 KNN distances use less memory than double distances", {
    skip_if_not_installed("float")
    set.seed(49)
    distances <- matrix(stats::runif(10000L), 1000L, 10L)
    knn_double <- list(
        indices = matrix(sample.int(1000L, 10000L, replace = TRUE), 1000L, 10L),
        distances = distances
    )
    knn_float <- knn_double
    knn_float$distances <- float::fl(distances)

    expect_type(knn_double$distances, "double")
    expect_s4_class(knn_float$distances, "float32")

    double_bytes <- as.numeric(object.size(knn_double$distances))
    float_bytes <- as.numeric(object.size(knn_float$distances))
    expect_lt(float_bytes, double_bytes)
    expect_lt(float_bytes / double_bytes, 0.65)
})

test_that(paste(
    "UMAP CSR graph weights stay float32 through prepared",
    "optimizer path"
), {
    skip_if_not_installed("float")
    idx <- cbind(
        c(2L, 1L, 1L, 1L, 2L, 3L, 4L, 5L, 6L, 7L),
        c(3L, 3L, 2L, 2L, 3L, 4L, 5L, 6L, 7L, 8L),
        c(4L, 4L, 5L, 3L, 4L, 5L, 6L, 7L, 8L, 9L)
    )
    dst <- float::fl(matrix(seq_len(length(idx)) / 100, nrow(idx), ncol(idx)))
    graph <- fastEmbedR:::umap_build_csr_graph(
        idx,
        dst,
        col_start = 0L,
        n_cols = ncol(idx),
        edge_budget = ncol(idx),
        n_threads = 2L,
        graph_mode = "fuzzy"
    )

    expect_s4_class(graph$weights, "float32")
    expect_s4_class(graph$epochs_per_sample, "float32")

    cfg <- fastEmbedR:::fast_knn_umap_config(
        nrow(idx), ncol(idx), backend = "cpu"
    )
    init <- fastEmbedR:::umap_init_from_csr_graph(
        graph,
        n_components = 2L,
        cfg = cfg,
        seed = 45L
    )
    layout <- fastEmbedR:::fast_knn_umap_csr_init_cpp(
        graph$offsets,
        graph$neighbors,
        graph$weights,
        init,
        2L,
        cfg$min_dist,
        cfg$negative_sample_rate,
        cfg$learning_rate,
        cfg$repulsion_strength,
        2L,
        45L,
        FALSE
    )
    expect_equal(dim(layout), c(nrow(idx), 2L))
    expect_true(all(is.finite(layout)))
})

test_that("tsne convenience wrapper runs the automatic KNN workflow", {
    set.seed(43)
    x <- rbind(matrix(rnorm(500), 25L, 20L), matrix(rnorm(500, 2), 25L, 20L))
    labels <- rep(1:2, each = 25L)

    fit <- tsne(
        x,
        perplexity = 1,
        early_exaggeration_iter = 2L,
        n_iter = 3L,
        seed = 43L
    )

    expect_s3_class(fit, "fastEmbedR_embedding")
    expect_equal(fit$parameters$method, "tsne")
    expect_equal(dim(fit$layout), c(nrow(x), 2L))
    expect_equal(colnames(fit$layout), c("TSNE1", "TSNE2"))
    expect_equal(fit$parameters$init, "pca_rsvd")
    expect_equal(fit$parameters$init_backend, "cpu_rsvd")
})

test_that("tsne PCA initialization uses native CPU RSVD", {
    set.seed(44)
    x <- matrix(rnorm(160L), 40L, 4L)
    init <- tsne_pca_init(x, n_components = 2L, seed = 44L, backend = "cpu")

    expect_equal(dim(init), c(40L, 2L))
    expect_true(all(is.finite(init)))
    expect_equal(attr(init, "fastEmbedR_init_method"), "pca_rsvd")
    expect_equal(attr(init, "fastEmbedR_init_backend"), "cpu_rsvd")
    expect_equal(
        attr(init, "fastEmbedR_init_package"),
        "fastEmbedR native CPU RSVD"
    )
})

test_that("public PCA API uses randomized SVD", {
    set.seed(45)
    x <- matrix(rnorm(300L), 60L, 5L)
    fit <- pca(x, ncomp = 2L, seed = 45L, backend = "cpu", n.cores = 2L)

    expect_s3_class(fit, "fastEmbedR_pca")
    expect_equal(dim(fit$scores), c(60L, 2L))
    expect_equal(dim(fit$loadings), c(5L, 2L))
    expect_equal(fit$method, "rsvd")
    expect_equal(fit$backend, "cpu_rsvd")
    expect_equal(fit$engine, "native_cpu_float32")
    expect_equal(fit$precision, "float32")
    expect_true(all(is.finite(fit$scores)))
    expect_true(all(is.finite(fit$loadings)))
    expect_true(all(is.finite(fit$singular_values)))
    expect_identical(fit$n.cores_requested, 2L)
    expect_true(is.na(fit$n.cores_effective) || fit$n.cores_effective >= 1L)
    expect_match(fit$core_control, "environment")
    expect_error(pca(x, n.cores = 0L), "positive integer")
    expect_error(pca(x, n.cores = NA_integer_), "positive integer")
})

test_that("native CPU RSVD agrees with exact PCA", {
    set.seed(451)
    x <- matrix(rnorm(2400L), 120L, 20L)
    fit <- pca(x, ncomp = 5L, seed = 451L, backend = "cpu", n.cores = 2L)
    reference <- stats::prcomp(x, center = TRUE, scale. = FALSE, rank. = 5L)
    reference_scores <- scale(reference$x, center = TRUE, scale = FALSE)
    candidate_scores <- scale(
        as.matrix(fit$scores), center = TRUE, scale = FALSE
    )
    correlation <- sum(svd(
        crossprod(reference_scores, candidate_scores),
        nu = 0L,
        nv = 0L
    )$d) / sqrt(
        sum(reference_scores * reference_scores) *
            sum(candidate_scores * candidate_scores)
    )

    expect_gte(correlation, 0.999)
    expect_equal(
        sum(candidate_scores * candidate_scores) /
            sum(reference_scores * reference_scores),
        1,
        tolerance = 1e-5
    )
})

test_that("native CPU RSVD preserves float32 output", {
    skip_if_not_installed("float")
    set.seed(452)
    x <- float::fl(matrix(rnorm(2400L), 120L, 20L))
    fit <- pca(x, ncomp = 5L, seed = 452L, backend = "cpu", n.cores = 2L)

    expect_s4_class(fit$scores, "float32")
    expect_s4_class(fit$loadings, "float32")
    expect_identical(fit$precision, "float32")
    expect_lt(as.numeric(object.size(fit$scores)), 120L * 5L * 8L)
    expect_true(all(is.finite(as.matrix(fit$scores))))
})

test_that("public PCA API can return an openTSNE-ready initialization", {
    set.seed(46)
    x <- matrix(rnorm(600L), 100L, 6L)
    xtest <- matrix(rnorm(120L), 20L, 6L)
    fit <- pca(
        x,
        ncomp = 2L,
        xtest = xtest,
        seed = 46L,
        backend = "cpu",
        tsne_init = TRUE
    )

    expect_s3_class(fit, "fastEmbedR_pca")
    expect_equal(dim(fit$tsne_init), c(100L, 2L))
    expect_equal(dim(fit$scores_test), c(20L, 2L))
    expected_test <- sweep(xtest, 2L, fit$center, "-", check.margin = FALSE)
    expected_test <- sweep(
        expected_test, 2L, fit$scale, "/", check.margin = FALSE
    )
    expected_test <- expected_test %*% fit$loadings
    colnames(expected_test) <- colnames(fit$scores)
    expect_equal(fit$scores_test, expected_test, tolerance = 1e-10)
    expect_equal(unname(colMeans(fit$tsne_init)), c(0, 0), tolerance = 1e-12)
    expect_equal(max(apply(fit$tsne_init, 2L, stats::sd)), 1e-4,
        tolerance = 1e-12
    )
    expect_identical(
        attr(fit$tsne_init, "fastEmbedR_init_backend"),
        fit$backend
    )
    expect_identical(
        attr(fit$tsne_init, "fastEmbedR_init_method"),
        paste0("pca_", fit$method)
    )
    expect_error(pca(x, tsne_init = NA), "must be TRUE or FALSE")
    expect_error(pca(x, xtest = matrix(0, 2L, 5L)), "same number of columns")
})

test_that("high-level embeddings avoid retaining KNN matrices by default", {
    set.seed(47)
    x <- matrix(rnorm(90), 30L, 3L)

    compact <- tsne(
        x,
        perplexity = 1,
        early_exaggeration_iter = 2L,
        n_iter = 3L,
        seed = 47L
    )
    retained <- tsne(
        x,
        perplexity = 1,
        early_exaggeration_iter = 2L,
        n_iter = 3L,
        seed = 47L,
        keep_knn = TRUE
    )

    expect_null(compact$knn)
    expect_equal(dim(retained$knn$indices), c(nrow(x), 1L))
    expect_equal(dim(retained$knn$distances), c(nrow(x), 1L))
})

test_that("CPU UMAP and openTSNE return genuine three-dimensional layouts", {
    x <- scale(as.matrix(iris[, 1:4]))

    umap_fit <- umap(
        x,
        n_neighbors = 15L,
        n_components = 3L,
        backend = "cpu",
        n.cores = 2L,
        seed = 4L
    )
    tsne_fit <- tsne(
        x,
        perplexity = 5,
        n_components = 3L,
        backend = "cpu",
        n.cores = 2L,
        seed = 4L,
        early_exaggeration_iter = 5L,
        n_iter = 10L,
        auto_config = FALSE
    )

    expect_equal(dim(umap_fit$layout), c(nrow(x), 3L))
    expect_equal(dim(tsne_fit$layout), c(nrow(x), 3L))
    expect_identical(umap_fit$parameters$n_components, 3L)
    expect_identical(tsne_fit$parameters$n_components, 3L)
    expect_identical(tsne_fit$parameters$negative_gradient_method, "exact")
    expect_true(all(is.finite(umap_fit$layout)))
    expect_true(all(is.finite(tsne_fit$layout)))
})

test_that("UMAP reports requested and validated effective CPU workers", {
    set.seed(911)
    x <- matrix(rnorm(240 * 5), nrow = 240)
    d <- as.matrix(stats::dist(x))
    diag(d) <- Inf
    k <- 15L
    indices <- t(apply(d, 1L, order))[, seq_len(k), drop = FALSE]
    distances <- matrix(
        d[cbind(rep(seq_len(nrow(d)), each = k), as.vector(t(indices)))],
        nrow = nrow(d), byrow = TRUE
    )
    prepared <- prepare_umap_knn(
        indices, distances,
        backend = "cpu", n.cores = 8,
        graph_mode = "fuzzy"
    )
    expect_identical(prepared$config$n.cores_requested, 8L)
    expect_identical(prepared$config$n.cores_effective, 4L)
    expect_identical(prepared$config$n_threads, 4L)
})
