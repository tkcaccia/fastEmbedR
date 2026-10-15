test_that("bucketed UMAP consumes only a package-built KNN pair", {
    root <- tempfile("knn_consume_")
    dir.create(root)
    on.exit(unlink(root, recursive = TRUE), add = TRUE)
    x <- cbind(seq_len(32L), (seq_len(32L) %% 7L) / 3)
    source <- massive_matrix(x)
    make_knn <- function(name) massive_full_knn_graph(
        source, 5L, file.path(root, name),
        memory_limit = "512MB", chunk_rows = 8L,
        reference_chunk_rows = 16L)
    retained <- make_knn("retained")
    consumed <- make_knn("consumed")
    imported <- massive_knn_graph(file.path(root, "retained"),
        32L, 5L)
    expect_error(massive_umap_fuzzy_graph(imported,
        file.path(root, "imported_graph"), "64MB",
        sort_method = "bucketed", consume_knn = TRUE),
        "package-built")
    weighted <- massive_umap_memberships(retained)
    expect_error(massive_umap_fuzzy_graph(weighted,
        file.path(root, "global_graph"), "64MB",
        consume_knn = TRUE), "package-built")
    regular <- massive_umap_fuzzy_graph(retained,
        file.path(root, "regular"), "64MB",
        sort_method = "bucketed")
    compact <- massive_umap_fuzzy_graph(consumed,
        file.path(root, "compact"), "64MB",
        sort_method = "bucketed", consume_knn = TRUE)
    suffix <- c(".offsets.u64", ".indices.u32", ".weights.f32")
    expect_identical(unname(tools::md5sum(paste0(
        file.path(root, "regular"), suffix))),
        unname(tools::md5sum(paste0(
            file.path(root, "compact"), suffix))))
    expect_true(all(file.exists(c(retained$indices_path,
        retained$values_path))))
    expect_false(any(file.exists(c(consumed$indices_path,
        consumed$values_path))))
    expect_equal(compact$consumed_knn_bytes, 32 * 5 * 8)
    expect_equal(regular$consumed_knn_bytes, 0)
    expect_lte(compact$peak_graph_stage_bytes,
        regular$peak_graph_stage_bytes)
    reopened <- massive_fuzzy_graph(file.path(root, "compact"), 32L)
    expect_equal(reopened$consumed_knn_bytes, 32 * 5 * 8)
    expect_equal(reopened$peak_graph_stage_bytes,
        compact$peak_graph_stage_bytes)
})

test_that("bucketed t-SNE consumes KNN after staging", {
    root <- tempfile("affinity_consume_")
    dir.create(root)
    on.exit(unlink(root, recursive = TRUE), add = TRUE)
    x <- cbind(seq_len(32L), (seq_len(32L) %% 5L) / 4)
    source <- massive_matrix(x)
    make_knn <- function(name) massive_full_knn_graph(
        source, 5L, file.path(root, name),
        memory_limit = "512MB", chunk_rows = 8L,
        reference_chunk_rows = 16L)
    retained <- make_knn("retained")
    consumed <- make_knn("consumed")
    regular <- massive_tsne_affinities(retained, 3,
        file.path(root, "regular"), "64MB",
        sort_method = "bucketed")
    compact <- massive_tsne_affinities(consumed, 3,
        file.path(root, "compact"), "64MB",
        sort_method = "bucketed", consume_knn = TRUE)
    suffix <- c(".offsets.u64", ".indices.u32", ".weights.f32")
    expect_identical(unname(tools::md5sum(paste0(
        file.path(root, "regular"), suffix))),
        unname(tools::md5sum(paste0(
            file.path(root, "compact"), suffix))))
    expect_false(any(file.exists(c(consumed$indices_path,
        consumed$values_path))))
    expect_equal(compact$consumed_knn_bytes, 32 * 5 * 8)
    expect_lt(compact$peak_graph_stage_bytes,
        regular$peak_graph_stage_bytes)
    reopened <- massive_affinity_graph(file.path(root, "compact"), 32L)
    expect_equal(reopened$consumed_knn_bytes, 32 * 5 * 8)
})

test_that("bucketed sorting releases processed KNN blocks exactly", {
    root <- tempfile("knn_stream_reclaim_")
    dir.create(root)
    on.exit(unlink(root, recursive = TRUE), add = TRUE)
    n <- 5000L
    k <- 30L
    ids <- outer(seq_len(n), seq_len(k), function(row, col)
        (row + col - 1L) %% n + 1L)
    indices <- as.integer(as.vector(t(ids)))
    distances <- rep(seq_len(k) / k, n)
    build <- function(method, consume) {
        prefix <- file.path(root, paste0(method, consume))
        input <- paste0(prefix, c(".knn.indices.u32",
            ".knn.distances.f32"))
        writeBin(indices, input[[1L]], size = 4L, endian = "little")
        writeBin(distances, input[[2L]], size = 4L, endian = "little")
        output <- paste0(prefix, ".graph")
        release <- if (consume) function() {
            expect_equal(unname(file.info(input)$size), c(0, 0))
            expect_equal(unlink(input), 0L)
            as.double(8 * n * k)
        } else NULL
        result <- fastEmbedR:::massive_symmetrize_graph_cpp(
            input[[1L]], input[[2L]], n, k, output,
            64 * 1024^2, paste0(method, "_bucketed"),
            if (method == "tsne") 20 else 0, 1L, FALSE,
            0, character(), "pairs_input", 0, 0L, NULL, release)
        list(result = result, output = output, input = input)
    }
    for (method in c("umap", "tsne")) {
        regular <- build(method, FALSE)
        compact <- build(method, TRUE)
        suffix <- c(".offsets.u64", ".indices.u32", ".weights.f32")
        expect_identical(unname(tools::md5sum(paste0(
            regular$output, suffix))), unname(tools::md5sum(paste0(
            compact$output, suffix))))
        expect_false(any(file.exists(compact$input)))
        expect_lt(compact$result$peak_with_knn_bytes,
            regular$result$peak_with_knn_bytes)
        expect_equal(compact$result$released_input_bytes,
            8 * n * k)
    }
})

test_that("failed streamed staging removes its partial KNN pair", {
    root <- tempfile("knn_partial_failure_")
    dir.create(root)
    on.exit(unlink(root, recursive = TRUE), add = TRUE)
    n <- 3000L
    k <- 30L
    paths <- file.path(root, c("knn.indices.u32",
        "knn.distances.f32"))
    ids <- outer(seq_len(n), seq_len(k), function(row, col)
        (row + col - 1L) %% n + 1L)
    indices <- as.integer(as.vector(t(ids)))
    indices[[1L]] <- 1L
    writeBin(indices, paths[[1L]], size = 4L,
        endian = "little")
    writeBin(rep(0.5, n * k), paths[[2L]], size = 4L,
        endian = "little")
    graph <- list(indices_path = paths[[1L]],
        values_path = paths[[2L]], n_vertices = n, k = k,
        file_bytes_each = 4 * n * k)
    plan <- list(prefix = file.path(root, "affinity"),
        limit = 64 * 1024^2)
    expect_error(fastEmbedR:::massive_graph_sort_run(graph, plan,
        "tsne_bucketed", 20, 1L, FALSE, FALSE, 100L,
        function() stop("release should not run")),
        "invalid or self edges")
    expect_false(any(file.exists(paths)))
})
