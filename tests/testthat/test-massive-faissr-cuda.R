test_that("single-device landmark CUDA KNN uses persistent faissR", {
    skip_if_not_installed("float")
    skip_if_not(faissR::faiss_gpu_available())
    set.seed(1147)
    reference <- matrix(rnorm(43L * 6L), ncol = 6L)
    query <- matrix(rnorm(19L * 6L), ncol = 6L)
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    reference_path <- file.path(directory, "reference.f32")
    query_path <- file.path(directory, "query.f32")
    writeBin(as.vector(t(reference)), reference_path,
        size = 4L, endian = "little")
    writeBin(as.vector(t(query)), query_path,
        size = 4L, endian = "little")
    found <- massive_landmark_knn(
        massive_matrix(query_path, 19L, 6L),
        massive_matrix(reference_path, 43L, 6L),
        5L, file.path(directory, "knn"), backend = "cuda",
        method = "exact", chunk_rows = 7L,
        memory_limit = "256MB")
    expected <- test_exact_knn(reference, query, 5L,
        n_threads = 1L, metric = "euclidean")
    observed <- fastEmbedR:::massive_read_knn_rows(found, n = 19L)
    expect_equal(observed$indices, expected$indices)
    expect_equal(observed$distances, expected$distances,
        tolerance = 1e-4)
    expect_identical(found$knn_provider, "faissR")
    expect_identical(found$reference_storage, "R_float32")
    expect_identical(found$query_storage, "R_float32")
    expect_identical(fastEmbedR:::massive_knn_query_storage(
        list(format = "memory"),
        list(method = "exact", backend = "cuda"),
        list(provider = "faissR")), "R_float32")
    expect_true(found$index_reused)
    expect_true(found$recall_audited)
})

test_that("single-device IVF returns distances for its selected IDs", {
    skip_if_not_installed("float")
    skip_if_not(faissR::faiss_gpu_available())
    set.seed(1148)
    reference <- matrix(rnorm(10000L * 8L), ncol = 8L)
    query <- matrix(rnorm(18L * 8L), ncol = 8L)
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    reference_path <- file.path(directory, "reference.f32")
    query_path <- file.path(directory, "query.f32")
    writeBin(as.vector(t(reference)), reference_path,
        size = 4L, endian = "little")
    writeBin(as.vector(t(query)), query_path,
        size = 4L, endian = "little")
    found <- massive_landmark_knn(
        massive_matrix(query_path, 18L, 8L),
        massive_matrix(reference_path, 10000L, 8L),
        5L, file.path(directory, "knn"), backend = "cuda",
        method = "ivf", chunk_rows = 7L,
        memory_limit = "256MB")
    observed <- fastEmbedR:::massive_read_knn_rows(found, n = 18L)
    expect_true(all(observed$indices >= 1L))
    expect_true(all(observed$indices <= nrow(reference)))
    for (row in seq_len(nrow(query))) {
        selected <- reference[observed$indices[row, ], , drop = FALSE]
        distances <- sqrt(rowSums((selected - matrix(query[row, ],
            nrow = 5L, ncol = 8L, byrow = TRUE))^2))
        expect_equal(observed$distances[row, ], distances,
            tolerance = 1e-4)
    }
    expect_identical(found$knn_provider, "faissR")
    expect_identical(found$method, "ivf")
    expect_false(found$recall_audited)
    expect_true(found$nlist > 0L && found$nprobe > 0L)
})

test_that("CUDA landmark embeddings run with faissR and no cuVS", {
    skip_if_not_installed("float")
    skip_if_not(faissR::faiss_gpu_available())
    skip_if_not(fastEmbedR:::embedding_cuda_available_cpp())
    skip_if(fastEmbedR:::native_cuda_knn_available_cpp())
    set.seed(1149)
    data <- matrix(rnorm(57L * 4L), ncol = 4L)
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    input <- file.path(directory, "data.f32")
    writeBin(as.vector(t(data)), input, size = 4L, endian = "little")
    source <- massive_matrix(input, 57L, 4L)
    embedding <- umap(source, massive = "landmark", landmarks = 20L,
        output = file.path(directory, "umap.f32"), backend = "cuda",
        n_neighbors = 5L, transform_k = 5L, chunk_rows = 9L,
        memory_limit = "512MB")
    expect_identical(embedding$graph$knn_provider, "faissR")
    expect_identical(embedding$backend, "cuda")
    expect_true(all(is.finite(as.matrix(embedding$layout))))
    transformed <- tsne(source, massive = "landmark", landmarks = 20L,
        output = file.path(directory, "tsne.f32"), backend = "cuda",
        perplexity = 3, early_exaggeration_iter = 3L, n_iter = 3L,
        transform_perplexity = 3, transform_iter = 3L,
        chunk_rows = 9L, memory_limit = "512MB")
    expect_identical(transformed$graph$knn_provider, "faissR")
    expect_identical(transformed$backend, "cuda")
    expect_true(all(is.finite(as.matrix(transformed$layout))))
})

test_that("resident CUDA host KNN uses faissR without cuVS", {
    skip_if_not(faissR::faiss_gpu_available())
    skip_if(fastEmbedR:::native_cuda_knn_available_cpp())
    set.seed(1817)
    data <- matrix(rnorm(61L * 5L), ncol = 5L)
    found <- fastEmbedR:::fastembedr_nn_without_self(
        data, k = 4L, backend = "cuda", method = "auto",
        metric = "euclidean", keep_gpu = FALSE
    )
    expect_identical(found$backend_used, "faiss_gpu_single_self")
    expect_identical(found$result_residency, "host")
    expect_identical(found$gpu_provider, "faiss_gpu")
    expect_equal(dim(found$indices), c(61L, 4L))
    expect_false(any(found$indices == row(found$indices)))
    distance <- as.matrix(stats::dist(data))
    diag(distance) <- Inf
    expected <- t(vapply(seq_len(nrow(data)), function(index) {
        order(distance[index, ])[seq_len(4L)]
    }, integer(4L)))
    expect_equal(found$indices, expected)
    for (metric in c("cosine", "correlation")) {
        values <- if (metric == "correlation") {
            data - rowMeans(data)
        } else data
        unit <- values / sqrt(rowSums(values^2))
        similarity <- tcrossprod(unit)
        diag(similarity) <- -Inf
        expected <- t(vapply(seq_len(nrow(data)), function(index) {
            order(similarity[index, ], decreasing = TRUE)[seq_len(4L)]
        }, integer(4L)))
        found <- fastEmbedR:::fastembedr_nn_without_self(
            data, k = 4L, backend = "cuda", method = "exact",
            metric = metric, keep_gpu = FALSE
        )
        expect_equal(found$indices, expected)
    }
})

test_that("faissR resident CUDA KNN feeds both optimizers", {
    skip_if_not(faissR::faiss_gpu_available())
    skip_if_not(fastEmbedR:::embedding_cuda_available_cpp())
    skip_if(fastEmbedR:::native_cuda_knn_available_cpp())
    set.seed(2217)
    data <- matrix(rnorm(43L * 5L), ncol = 5L)
    knn <- fastEmbedR:::fastembedr_nn_without_self(
        data, 5L, "cuda", "exact", "euclidean", keep_gpu = TRUE
    )
    expect_s3_class(knn, "faissR_gpu_knn")
    expect_identical(knn$result_residency, "cuda")
    expect_identical(knn$gpu_provider, "faiss_gpu")
    expect_identical(knn$device_to_host_result_copies, 0L)
    host <- fastEmbedR:::fastembedr_gpu_knn_to_host(knn)
    distance <- as.matrix(stats::dist(data))
    diag(distance) <- Inf
    expected <- t(vapply(seq_len(nrow(data)), function(index) {
        order(distance[index, ])[seq_len(5L)]
    }, integer(5L)))
    expect_equal(host$indices, expected)
    expect_identical(attr(host, "gpu_resident_source"), TRUE)
    umap_fit <- fastEmbedR:::fast_knn_umap(
        knn, backend = "cuda", n_components = 2L
    )
    expect_equal(dim(umap_fit), c(43L, 2L))
    expect_true(all(is.finite(umap_fit)))
    tsne_fit <- fastEmbedR:::tsne_knn(
        knn, backend = "cuda", perplexity = 1.5,
        n_neighbors = 5L, early_exaggeration_iter = 3L, n_iter = 3L
    )
    expect_equal(dim(tsne_fit), c(43L, 2L))
    expect_true(all(is.finite(tsne_fit)))
})

test_that("CUDA reference queries use faissR host and device results", {
    skip_if_not(faissR::faiss_gpu_available())
    set.seed(2241)
    data <- matrix(rnorm(37L * 5L), ncol = 5L)
    query <- data[c(3L, 11L, 29L), , drop = FALSE] + 0.01
    distance <- outer(seq_len(nrow(query)), seq_len(nrow(data)),
        Vectorize(function(i, j) sqrt(sum((query[i, ] - data[j, ])^2))))
    expected <- t(vapply(seq_len(nrow(query)), function(i) {
        order(distance[i, ])[seq_len(4L)]
    }, integer(4L)))
    host <- fastEmbedR:::fastembedr_native_query_knn(
        data, query, 4L, backend = "cuda", method = "exact"
    )
    expect_identical(host$indices, expected)
    expect_true(startsWith(host$backend_used, "faiss_gpu"))
    device <- fastEmbedR:::fastembedr_native_query_knn(
        data, query, 4L, backend = "cuda", method = "exact",
        keep_gpu = TRUE
    )
    expect_s3_class(device, "faissR_gpu_knn")
    expect_identical(device$result_residency, "cuda")
    observed <- fastEmbedR:::fastembedr_gpu_knn_to_host(device)
    expect_identical(observed$indices, expected)
})
