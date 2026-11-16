test_that("landmark KNN records the selected recall tier", {
    set.seed(118)
    query <- massive_matrix(matrix(rnorm(48L * 6L), ncol = 6L))
    reference <- massive_matrix(matrix(rnorm(64L * 6L), ncol = 6L))
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    old <- options(fastEmbedR.target_recall = 0.95)
    on.exit(options(old), add = TRUE)
    exact <- massive_landmark_knn(query, reference, 4L,
        file.path(directory, "exact"), method = "exact")
    expect_identical(exact$target_recall, 0.95)
    expect_identical(exact$recall_status, "exact_by_construction")
    for (tier in c(0.9, 0.95, 0.99)) {
        found <- massive_landmark_knn(query, reference, 4L,
            file.path(directory, paste0("hnsw-", tier)), method = "hnsw",
            target_recall = tier)
        expect_identical(found$target_recall, tier)
        expect_identical(found$recall_status, "not_runtime_audited")
        expect_true(is.na(found$pilot_recall))
    }
    invalid <- file.path(directory, "invalid")
    expect_error(massive_landmark_knn(query, reference, 4L, invalid,
        target_recall = 0.8), "target_recall")
    expect_false(any(file.exists(paste0(invalid,
        c(".indices.u32", ".distances.f32")))))
})

test_that("landmark HNSW passes each tier to faissR", {
    reference <- massive_matrix(matrix(seq_len(96L) / 13, ncol = 6L))
    received <- NA_real_
    for (tier in c(0.9, 0.95, 0.99)) {
        with_mocked_bindings(
            faissr_hnsw_index_build_cpp = function(data, n, p, k,
                    n.cores, m, ef, target_recall) {
                received <<- target_recall
                list(pointer = NULL)
            }, {
                fastEmbedR:::massive_faiss_hnsw_reference(
                    reference, 4L, 1L, target_recall = tier)
            }, .package = "fastEmbedR")
        expect_identical(received, tier)
    }
})

test_that("landmark Metal query preserves the requested tier", {
    received <- NA_real_
    state <- list(matrix = matrix(1, 2L, 2L), target_recall = 0.9)
    with_mocked_bindings(
        faissr_metal_query_knn_cpp = function(data, query, k, method,
                metric, target_recall) {
            received <<- target_recall
            list(indices = matrix(1L, 1L, 1L),
                distances = matrix(0, 1L, 1L),
                backend = "metal", accelerator = "metal",
                backend_used = "faissr_native_metal_exact_query",
                method = "native_metal_exact_query", index_base = 1L,
                metric = metric, target_recall = target_recall,
                n_reference = nrow(data), n_query = nrow(query),
                exclude_self = FALSE, exact = TRUE,
                target_met = TRUE)
        }, {
            fastEmbedR:::massive_knn_query(state,
                matrix(1, 1L, 2L), 1L, "exact", 1L, "metal")
        }, .package = "fastEmbedR")
    expect_identical(received, 0.9)
})

test_that("multi-GPU full IVF sends every tier to its worker route", {
    received <- NA_real_
    for (tier in c(0.90, 0.95, 0.99)) {
        with_mocked_bindings(
            massive_validate_devices = function(...) c(0L, 1L),
            massive_full_knn_devices = function(..., target_recall) {
                received <<- target_recall
                list(target_recall = target_recall)
            }, {
                found <- fastEmbedR:::massive_full_knn_device_request(
                    list(nrow = 8L), 3L, "unused", "cuda", 1L,
                    NULL, NULL, "256MB", FALSE, FALSE, 0L, 1L,
                    c(0L, 1L), "ivf_sharded", tier)
                expect_identical(found$target_recall, tier)
            }, .package = "fastEmbedR")
        expect_identical(received, tier)
    }
})

test_that("multi-GPU full KNN checkpoint binds the recall tier", {
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    input <- file.path(directory, "input.f32")
    writeBin(as.numeric(seq_len(24L)), input, size = 4L)
    source <- massive_matrix(input, 8L, 3L)
    checkpoint <- fastEmbedR:::massive_full_knn_devices_checkpoint
    call <- function(tier, resume) checkpoint(source, 3L,
        file.path(directory, "graph"), "ivf_sharded", c(0L, 1L),
        4L, 8L, "256MB", 2L, 1L, resume, tier)
    with_mocked_bindings(
        massive_cuda_device_uuids = function(...) c("GPU-A", "GPU-B"),
        {
            expect_true(file.exists(call(0.90, FALSE)))
            expect_error(call(0.95, TRUE), "does not match")
            expect_true(file.exists(call(0.90, TRUE)))
        }, .package = "fastEmbedR")
})

test_that("landmark checkpoint rejects a changed recall tier", {
    data <- massive_matrix(matrix(seq_len(96L) / 17, ncol = 6L))
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    setup <- fastEmbedR:::massive_knn_setup(data, data, 4L,
        file.path(directory, "knn"), "cpu", 1L, "hnsw", NULL,
        "256MB", FALSE, target_recall = 0.9)
    fastEmbedR:::massive_knn_checkpoint(data, setup, FALSE)
    setup$target_recall <- 0.95
    expect_error(fastEmbedR:::massive_knn_checkpoint(data, setup, TRUE),
        "does not match")
})

test_that("file-backed UMAP and t-SNE carry the tier to landmark KNN", {
    set.seed(119)
    x <- matrix(rnorm(64L * 4L), ncol = 4L)
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    input <- file.path(directory, "input.f32")
    writeBin(as.vector(t(x)), input, size = 4L, endian = "little")
    source <- massive_matrix(input, nrow(x), ncol(x))
    fit_umap <- umap(source, massive = "landmark", landmarks = 20L,
        n_neighbors = 5L, transform_k = 5L, backend = "cpu",
        output = file.path(directory, "umap.f32"), target_recall = 0.9)
    expect_identical(fit_umap$target_recall, 0.9)
    expect_identical(fit_umap$graph$target_recall, 0.9)
    expect_true(is.na(fit_umap$measured_recall))
    fit_tsne <- tsne(source, massive = "landmark", landmarks = 20L,
        perplexity = 3, early_exaggeration_iter = 5L, n_iter = 5L,
        transform_perplexity = 3, transform_iter = 5L, backend = "cpu",
        output = file.path(directory, "tsne.f32"), target_recall = 0.9)
    expect_identical(fit_tsne$target_recall, 0.9)
    expect_identical(fit_tsne$graph$target_recall, 0.9)
    expect_true(is.na(fit_tsne$measured_recall))
    reused <- tsne(source, massive = "landmark", landmarks = 20L,
        nn = fit_umap, perplexity = 3, transform_k = 5L,
        early_exaggeration_iter = 5L, n_iter = 5L,
        transform_perplexity = 3, transform_iter = 5L, backend = "cpu",
        output = file.path(directory, "reused.f32"))
    expect_true(reused$graph_reused)
    expect_identical(reused$target_recall, 0.9)
    expect_error(tsne(source, massive = "landmark", landmarks = 20L,
        nn = fit_umap, target_recall = 0.95), "Precomputed KNN")
})

test_that("resident landmark UMAP and t-SNE keep the requested tier", {
    set.seed(120)
    x <- matrix(rnorm(64L * 4L), ncol = 4L)
    fit_umap <- umap(x, landmarks = 0.5, n_neighbors = 5L,
        backend = "cpu", target_recall = 0.9)
    expect_identical(fit_umap$parameters$target_recall, 0.9)
    expect_true(is.na(fit_umap$parameters$reference_knn_recall))
    expect_true(is.na(fit_umap$parameters$projection_knn_recall))
    fit_tsne <- tsne(x, landmarks = 0.5, perplexity = 3,
        early_exaggeration_iter = 5L, n_iter = 5L,
        transform_perplexity = 3, transform_iter = 5L,
        backend = "cpu", target_recall = 0.9)
    expect_identical(fit_tsne$parameters$target_recall, 0.9)
    expect_true(is.na(fit_tsne$parameters$reference_knn_recall))
    expect_true(is.na(fit_tsne$parameters$projection_knn_recall))
})

test_that("Metal landmark KNN retains each tier on the device route", {
    skip_if_not(fastEmbedR:::fastembedr_metal_knn_available())
    set.seed(121)
    query <- massive_matrix(matrix(rnorm(24L * 4L), ncol = 4L))
    reference <- massive_matrix(matrix(rnorm(48L * 4L), ncol = 4L))
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    for (tier in c(0.9, 0.95, 0.99)) {
        fit <- massive_landmark_knn(query, reference, 4L,
            file.path(directory, paste0("metal-", tier)),
            backend = "metal", method = "exact", target_recall = tier)
        expect_identical(fit$backend, "metal")
        expect_identical(fit$knn_provider, "faissR")
        expect_identical(fit$target_recall, tier)
        expect_identical(fit$recall_status, "exact_by_construction")
        expect_true(is.na(fit$pilot_recall))
    }
})
