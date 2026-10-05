test_that("physical postings preserve all source rows and IDs", {
    values <- matrix(seq_len(120) / 17, nrow = 30L, ncol = 4L)
    values[seq_len(15L), 1L] <- values[seq_len(15L), 1L] - 100
    values[seq.int(16L, 30L), 1L] <-
        values[seq.int(16L, 30L), 1L] + 100
    centers <- values[c(2L, 17L, 29L), , drop = FALSE]
    input <- tempfile(fileext = ".f32")
    prefix <- tempfile()
    writeBin(as.vector(t(values)), input, size = 4L,
        endian = "little")
    source <- massive_matrix(input, nrow(values), ncol(values))
    index <- massive_coarse_postings(source, centers, prefix,
        chunk_rows = 7L, n.cores = 2L, memory_limit = "256MB")
    reopened <- massive_open_postings(prefix)
    expect_identical(reopened$centers, centers)
    expect_identical(reopened$postings$counts, index$counts)
    expect_true(file.exists(index$manifest_path))
    expect_s3_class(index, "fastEmbedR_massive_postings")
    expect_identical(index$method, "nearest_center_physical_postings")
    expect_equal(sum(index$counts), nrow(values))
    expect_equal(tail(index$offsets, 1L), nrow(values))
    postings <- lapply(seq_len(index$nlist), function(bucket) {
        if (index$counts[[bucket]] == 0) return(NULL)
        massive_read_posting(index, bucket, n = index$counts[[bucket]])
    })
    ids <- unlist(lapply(postings, `[[`, "row_id"), use.names = FALSE)
    data <- do.call(rbind, lapply(postings, `[[`, "data"))
    expect_setequal(ids, seq_len(nrow(values)))
    expect_equal(data, values[ids, , drop = FALSE], tolerance = 1e-5)
    expected <- vapply(seq_len(nrow(values)), function(i) {
        which.min(rowSums((centers - matrix(values[i, ],
            nrow(centers), ncol(values), byrow = TRUE))^2))
    }, integer(1L))
    observed <- rep(seq_len(index$nlist), index$counts)
    expect_identical(observed[order(ids)], expected)
    expect_error(massive_coarse_postings(source, centers, prefix),
        "already exist")
    con <- file(index$ids_path, "r+b")
    writeBin(0L, con, size = 4L, endian = "little")
    close(con)
    expect_error(massive_read_posting(index, 1L), "changed")
    expect_error(massive_open_postings(prefix), "changed")
})

test_that("posting manifests reject missing or altered centers", {
    expect_error(massive_open_postings(tempfile()), "manifest")
    values <- matrix(seq_len(60) / 9, 15L, 4L)
    centers <- values[c(1L, 15L), , drop = FALSE]
    input <- tempfile(fileext = ".f32")
    writeBin(as.vector(t(values)), input, size = 4L,
        endian = "little")
    source <- massive_matrix(input, 15L, 4L)
    prefix <- tempfile()
    index <- massive_coarse_postings(source, centers, prefix,
        memory_limit = "128MB")
    saved <- readRDS(index$manifest_path)
    saved$centers[1L, 1L] <- saved$centers[1L, 1L] + 1
    saveRDS(saved, index$manifest_path)
    expect_error(massive_open_postings(prefix), "manifest")
    saveRDS(1L, index$manifest_path)
    expect_error(massive_open_postings(prefix), "manifest")
})

test_that("physical postings reject unsupported sources and budgets", {
    values <- matrix(as.double(seq_len(24)), 6L, 4L)
    centers <- values[c(1L, 6L), , drop = FALSE]
    source <- massive_matrix(values)
    expect_error(massive_coarse_postings(source, centers, tempfile()),
        "file-backed")
    input <- tempfile(fileext = ".f32")
    writeBin(as.vector(t(values)), input, size = 4L,
        endian = "little")
    source <- massive_matrix(input, nrow(values), ncol(values))
    expect_error(massive_coarse_postings(source, centers, tempfile(),
        memory_limit = "1MB"), "memory_limit")
    expect_error(massive_coarse_postings(source, centers[, 1:2],
        tempfile()), "feature-compatible")
})

test_that("physical postings accept fbin and mmap without changing rows", {
    values <- matrix(as.double(seq_len(48)) / 7, 12L, 4L)
    centers <- values[c(2L, 9L), , drop = FALSE]
    input <- tempfile(fileext = ".fbin")
    con <- file(input, "wb")
    writeBin(as.integer(dim(values)), con, size = 4L,
        endian = "little")
    writeBin(as.vector(t(values)), con, size = 4L,
        endian = "little")
    close(con)
    source <- massive_matrix(input, access = "mmap")
    index <- massive_coarse_postings(source, centers, tempfile(),
        chunk_rows = 3L, n.cores = 2L, memory_limit = "128MB")
    observed <- lapply(seq_len(index$nlist), function(bucket) {
        massive_read_posting(index, bucket, n = index$counts[[bucket]])
    })
    ids <- unlist(lapply(observed, `[[`, "row_id"), use.names = FALSE)
    data <- do.call(rbind, lapply(observed, `[[`, "data"))
    expect_setequal(ids, seq_len(nrow(values)))
    expect_equal(data, values[ids, , drop = FALSE], tolerance = 1e-5)
})

test_that("physical postings resume an interrupted write pass", {
    values <- matrix(seq_len(120) / 13, 30L, 4L)
    values[1:15, 1L] <- values[1:15, 1L] - 50
    centers <- values[c(2L, 17L, 29L), , drop = FALSE]
    input <- tempfile(fileext = ".f32")
    writeBin(as.vector(t(values)), input, size = 4L,
        endian = "little")
    source <- massive_matrix(input, 30L, 4L)
    prefix <- tempfile()
    original <- fastEmbedR:::massive_checkpoint_write
    interrupted <- function(state, path) {
        original(state, path)
        if (!is.null(state$completed_rows) &&
            state$completed_rows >= 7) stop("forced interruption")
    }
    with_mocked_bindings(
        massive_checkpoint_write = interrupted,
        expect_error(massive_coarse_postings(source, centers, prefix,
            chunk_rows = 7L, memory_limit = "128MB",
            checkpoint = TRUE, checkpoint_every = 1L),
            "forced interruption"),
        .package = "fastEmbedR"
    )
    sidecar <- paste0(prefix, ".checkpoint.rds")
    expect_true(file.exists(sidecar))
    state <- readRDS(sidecar)
    expect_equal(state$completed_rows, 7)
    expect_error(massive_coarse_postings(source, centers, prefix,
        chunk_rows = 7L, memory_limit = "128MB",
        checkpoint = TRUE, resume = TRUE,
        checkpoint_every = 2L), "does not match")
    resumed <- massive_coarse_postings(source, centers, prefix,
        chunk_rows = 7L, memory_limit = "128MB",
        checkpoint = TRUE, resume = TRUE, checkpoint_every = 1L)
    fresh <- massive_coarse_postings(source, centers, tempfile(),
        chunk_rows = 7L, memory_limit = "128MB")
    for (suffix in c(".features.f32", ".ids.u32",
        ".offsets.u64")) {
        left <- paste0(prefix, suffix)
        right <- paste0(sub("\\.features.f32$", "",
            fresh$features$path), suffix)
        expect_identical(unname(tools::md5sum(left)),
            unname(tools::md5sum(right)))
    }
    expect_false(file.exists(sidecar))
    expect_equal(sum(resumed$counts), 30)
})

test_that("completed postings resume interrupted final renames", {
    values <- matrix(seq_len(120) / 13, 30L, 4L)
    centers <- values[c(2L, 17L, 29L), , drop = FALSE]
    input <- tempfile(fileext = ".f32")
    writeBin(as.vector(t(values)), input, size = 4L,
        endian = "little")
    source <- massive_matrix(input, 30L, 4L)
    baseline <- massive_coarse_postings(source, centers, tempfile(),
        chunk_rows = 7L, memory_limit = "128MB")
    suffixes <- c(".features.f32", ".ids.u32", ".offsets.u64")
    expected <- paste0(sub("\\.features\\.f32$", "",
        baseline$features$path), suffixes)
    original <- fastEmbedR:::massive_checkpoint_write
    for (renamed in 0:4) {
        prefix <- tempfile()
        interrupted <- function(state, path) {
            original(state, path)
            if (identical(state$completed_rows, 30)) {
                stop("forced finalization interruption")
            }
        }
        with_mocked_bindings(
            massive_checkpoint_write = interrupted,
            expect_error(massive_coarse_postings(source, centers,
                prefix, chunk_rows = 7L, memory_limit = "128MB",
                checkpoint = TRUE, checkpoint_every = 1L),
                "forced finalization interruption"),
            .package = "fastEmbedR")
        outputs <- paste0(prefix, suffixes)
        if (renamed >= 1) file.rename(paste0(outputs[[1L]], ".part"),
            outputs[[1L]])
        if (renamed >= 2) file.rename(paste0(outputs[[2L]], ".part"),
            outputs[[2L]])
        if (renamed >= 3) {
            offsets <- c(0, cumsum(readRDS(paste0(prefix,
                ".checkpoint.rds"))$counts))
            words <- as.integer(c(rbind(offsets,
                rep.int(0L, length(offsets)))))
            writeBin(words, outputs[[3L]], size = 4L,
                endian = "little")
        }
        if (renamed == 4) {
            writeBin(rep.int(1L, 2L * length(offsets)),
                outputs[[3L]], size = 4L, endian = "little")
            expect_error(massive_coarse_postings(source, centers,
                prefix, chunk_rows = 7L, memory_limit = "128MB",
                checkpoint = TRUE, resume = TRUE,
                checkpoint_every = 1L), "offsets disagree")
            next
        }
        resumed <- massive_coarse_postings(source, centers, prefix,
            chunk_rows = 7L, memory_limit = "128MB",
            checkpoint = TRUE, resume = TRUE, checkpoint_every = 1L)
        expect_identical(unname(tools::md5sum(outputs)),
            unname(tools::md5sum(expected)))
        expect_equal(sum(resumed$counts), 30)
        expect_false(file.exists(paste0(prefix, ".checkpoint.rds")))
    }
})

test_that("posting scans match full exact KNN when all lists are probed", {
    values <- matrix(seq_len(180) / 23, 45L, 4L)
    values[1:15, 1L] <- values[1:15, 1L] - 100
    values[31:45, 1L] <- values[31:45, 1L] + 100
    centers <- values[c(3L, 22L, 40L), , drop = FALSE]
    input <- tempfile(fileext = ".f32")
    writeBin(as.vector(t(values)), input, size = 4L,
        endian = "little")
    source <- massive_matrix(input, 45L, 4L)
    prefix <- tempfile()
    massive_coarse_postings(source, centers, prefix,
        chunk_rows = 7L, memory_limit = "128MB")
    restored <- massive_open_postings(prefix)
    postings <- restored$postings
    centers <- restored$centers
    observed <- massive_search_postings(source, postings, centers,
        first = 4L, n = 9L, k = 3L, nprobe = 3L,
        reference_chunk = 5L, n.cores = 2L,
        memory_limit = "128MB", exclude_self = TRUE)
    expected <- fastEmbedR:::massive_exact_graph_batch_cpp(
        source, 4, 9L, 3L, 5L, 2L)
    expect_identical(observed$indices, expected$indices)
    expect_equal(observed$distances, expected$distances,
        tolerance = 1e-6)
    expect_true(observed$exact)
    expect_true(all(observed$candidate_count == 44))
    expect_identical(observed$backend_used,
        "native_cpu_posting_scan")
    expect_equal(observed$posting_read_bytes,
        45 * (4 * ncol(values) + 4))
    grouped <- massive_search_postings(postings$features,
        postings, centers, first = 1L, n = 45L, k = 3L,
        nprobe = 3L, reference_chunk = 5L, n.cores = 2L,
        memory_limit = "128MB", exclude_self = TRUE)
    connection <- file(postings$ids_path, "rb")
    row_ids <- readBin(connection, integer(), n = 45L,
        size = 4L, endian = "little")
    close(connection)
    all_exact <- fastEmbedR:::massive_exact_graph_batch_cpp(
        source, 1, 45L, 3L, 5L, 2L)
    expect_identical(grouped$indices[order(row_ids), ],
        all_exact$indices)
    expect_equal(grouped$distances[order(row_ids), ],
        all_exact$distances, tolerance = 1e-6)
    expect_true(all(grouped$candidate_count == 44))
    approximate <- massive_search_postings(source, postings, centers,
        first = 4L, n = 9L, k = 1L, nprobe = 1L,
        reference_chunk = 5L, memory_limit = "128MB",
        exclude_self = TRUE)
    expect_false(approximate$exact)
    expect_true(all(approximate$candidate_count < 44))
    expect_gt(approximate$posting_read_bytes, 0)
    expect_lte(approximate$posting_read_bytes,
        observed$posting_read_bytes)
    expect_error(massive_search_postings(source, postings,
        centers + 1, n = 1L, k = 1L), "do not match")
    other <- massive_matrix(values)
    expect_error(massive_search_postings(other, postings, centers,
        n = 1L, k = 1L, nprobe = 1L, exclude_self = TRUE),
        "original or grouped posting source")
})

test_that("posting recall pilot detects weak probes before graph build", {
    values <- cbind(as.double(seq_len(8L)), rep(0, 8L))
    centers <- values[c(1L, 8L), , drop = FALSE]
    input <- tempfile(fileext = ".f32")
    writeBin(as.vector(t(values)), input, size = 4L,
        endian = "little")
    source <- massive_matrix(input, 8L, 2L)
    postings <- massive_coarse_postings(source, centers,
        tempfile(), memory_limit = "128MB")
    weak <- massive_posting_recall_pilot(source, postings, centers,
        k = 3L, nprobe = 1L, sample_rows = 8L,
        memory_limit = "128MB")
    strong <- massive_posting_recall_pilot(source, postings, centers,
        k = 3L, nprobe = 2L, sample_rows = 8L,
        memory_limit = "128MB")
    boundary <- massive_posting_recall_pilot(source, postings,
        centers, k = 3L, nprobe = 1L, sample_rows = 2L,
        memory_limit = "128MB")
    expect_identical(weak$sample_rows, seq_len(8L))
    expect_identical(boundary$sample_rows, c(2L, 5L))
    expect_false(boundary$sample_target_met)
    expect_identical(boundary$audit_sampling,
        "evenly_spaced_and_distant_center_and_posting_lists")
    expect_identical(boundary$posting_lists_sampled, 1L)
    expect_equal(boundary$selection_source_bytes,
        source$nrow * source$ncol * 4)
    expect_lt(weak$minimum_row_recall, 1)
    expect_false(weak$sample_target_met)
    expect_equal(strong$row_recall, rep(1, 8L))
    expect_true(strong$sample_target_met)
    expect_equal(strong$observed_recall, 1)
    expect_identical(strong$posting_backend,
        "native_cpu_posting_scan")
    expect_identical(strong$reference_backend,
        "native_cpu_exact_stream")
    expect_true(all(strong$candidate_count == 7))
    expect_error(massive_posting_recall_pilot(source, postings,
        centers, sample_rows = 9L), "Invalid recall pilot")
    expect_error(massive_posting_recall_pilot(source, postings,
        centers, target_recall = 2), "Invalid recall pilot")
    query <- source
    query$selected_rows <- c(0, 1)
    expect_error(fastEmbedR:::massive_posting_search_cpp(query,
        postings$features$path, postings$ids_path,
        postings$offsets_path, centers, 1, 2L, 1L, 2L,
        4L, 1L, TRUE, postings$nrow, FALSE),
        "Selected query row")
})

test_that("top-probe routing preserves center distance order", {
    set.seed(42L)
    values <- matrix(rnorm(160L * 5L), 160L, 5L)
    centers <- values[seq.int(1L, 160L, by = 10L), , drop = FALSE]
    input <- tempfile(fileext = ".f32")
    writeBin(as.vector(t(values)), input, size = 4L,
        endian = "little")
    source <- massive_matrix(input, 160L, 5L)
    postings <- massive_coarse_postings(source, centers,
        tempfile(), memory_limit = "128MB")
    actual <- massive_read_rows(source, n = 160L)
    assigned <- vapply(seq_len(nrow(actual)), function(i) {
        delta <- sweep(centers, 2L, actual[i, ])
        which.min(rowSums(delta * delta))
    }, integer(1L))
    found <- massive_search_postings(source, postings, centers,
        n = 8L, k = 3L, nprobe = 4L,
        memory_limit = "128MB", exclude_self = TRUE)
    for (i in seq_len(8L)) {
        delta <- sweep(centers, 2L, actual[i, ])
        selected <- order(rowSums(delta * delta))[seq_len(4L)]
        candidate <- which(assigned %in% selected &
            seq_len(nrow(actual)) != i)
        gap <- sweep(actual[candidate, , drop = FALSE],
            2L, actual[i, ])
        distance <- rowSums(gap * gap)
        expected <- candidate[order(distance, candidate)][seq_len(3L)]
        expect_identical(as.integer(found$indices[i, ]),
            as.integer(expected))
    }
})

test_that("posting search writes a reusable exact full graph", {
    values <- matrix(seq_len(180) / 19, 45L, 4L)
    values[1:15, 1L] <- values[1:15, 1L] - 100
    values[31:45, 1L] <- values[31:45, 1L] + 100
    centers <- values[c(3L, 22L, 40L), , drop = FALSE]
    input <- tempfile(fileext = ".f32")
    writeBin(as.vector(t(values)), input, size = 4L,
        endian = "little")
    source <- massive_matrix(input, 45L, 4L)
    postings <- massive_coarse_postings(source, centers, tempfile(),
        memory_limit = "128MB")
    plan <- fastEmbedR:::massive_posting_search_plan(source,
        postings, 1L, 3L, 3L, 5L, 2L, "128MB")
    expect_equal(plan$estimated_ram_bytes,
        64 * 1024^2 + 12 * length(centers) +
            (32 + 16 * 2) * 3 +
            (4 * 4 + 32 * 3 + 16 * 3 + 64) +
            5 * (4 * 4 + 4))
    graph <- massive_full_knn_graph(source, 3L, tempfile(),
        method = "coarse_postings", postings = postings,
        centers = centers, nprobe = 3L, chunk_rows = 7L,
        reference_chunk_rows = 5L, n.cores = 2L,
        memory_limit = "128MB", audit_rows = 5L)
    auto <- massive_full_knn_graph(source, 3L, tempfile(),
        method = "coarse_postings", postings = postings,
        centers = centers, nprobe = 3L,
        reference_chunk_rows = 5L, n.cores = 2L,
        memory_limit = "128MB")
    grouped <- massive_full_knn_graph(source, 3L, tempfile(),
        method = "coarse_postings", postings = postings,
        centers = centers, nprobe = 3L, chunk_rows = 7L,
        reference_chunk_rows = 5L, n.cores = 2L,
        memory_limit = "128MB", query_order = "posting",
        audit_rows = 5L)
    expect_equal(auto$resources$chunk_rows, nrow(values))
    expect_equal(graph$resources$scan_bytes_upper,
        ceiling(nrow(values) / 7L) * nrow(values) *
            (4 * ncol(values) + 4))
    expect_equal(graph$resources$posting_read_bytes,
        graph$resources$scan_bytes_upper)
    expect_equal(auto$resources$posting_read_bytes,
        nrow(values) * (4 * ncol(values) + 4))
    expect_identical(unname(tools::md5sum(graph$indices_path)),
        unname(tools::md5sum(auto$indices_path)))
    expect_identical(unname(tools::md5sum(graph$distances_path)),
        unname(tools::md5sum(auto$distances_path)))
    expect_identical(unname(tools::md5sum(graph$indices_path)),
        unname(tools::md5sum(grouped$indices_path)))
    expect_identical(unname(tools::md5sum(graph$distances_path)),
        unname(tools::md5sum(grouped$distances_path)))
    expect_identical(grouped$query_order, "posting")
    expect_identical(grouped$audit_sample_rows,
        graph$audit_sample_rows)
    expect_true(is.finite(grouped$reorder_seconds))
    expect_false(file.exists(paste0(
        sub("\\.indices\\.u32$", "", grouped$indices_path),
        ".grouped.indices.u32")))
    expect_identical(graph$method, "native_cpu_posting_scan")
    expect_true(graph$exact)
    expect_true(graph$recall_audited)
    expect_equal(graph$observed_recall, 1)
    expected <- fastEmbedR:::massive_exact_graph_batch_cpp(
        source, 1, 45L, 3L, 5L, 2L)
    for (row in c(1L, 15L, 30L, 45L)) {
        found <- massive_read_graph_edges(graph, row, 1L)
        expect_identical(found$to, expected$indices[row, ])
        expect_equal(found$distance, expected$distances[row, ],
            tolerance = 1e-6)
    }
    expect_error(massive_full_knn_graph(source, 3L, tempfile(),
        backend = "cuda", method = "coarse_postings",
        postings = postings, centers = centers), "no CUDA fallback")
    expect_error(massive_full_knn_graph(source, 3L, tempfile(),
        method = "exact", postings = postings),
        "Posting controls require")
    expect_error(massive_full_knn_graph(source, 3L, tempfile(),
        method = "exact", query_order = "posting"),
        "requires posting search")
})

test_that("sample recall target checks weak rows, not only the mean", {
    stats <- fastEmbedR:::massive_recall_summary(
        c(rep(1, 99L), 0), 0.99)
    expect_equal(stats$observed_recall, 0.99)
    expect_equal(stats$minimum_row_recall, 0)
    expect_equal(stats$sampled_rows_below_target, 1L)
    expect_false(stats$sample_target_met)
})

test_that("posting audit includes hard-to-route rows", {
    values <- cbind(c(-11, -10, -9, -8, -7, -6, -5, -4,
        0, 1, 2, 3, 4, 5, 6, 20), seq_len(16L) / 10)
    centers <- values[c(2L, 13L), , drop = FALSE]
    input <- tempfile(fileext = ".f32")
    writeBin(as.vector(t(values)), input, size = 4L,
        endian = "little")
    source <- massive_matrix(input, 16L, 2L)
    postings <- massive_coarse_postings(source, centers,
        tempfile(), memory_limit = "128MB")
    nearest <- apply(values, 1L, function(row) min(rowSums(
        (centers - matrix(row, 2L, 2L, byrow = TRUE))^2)))
    ranked <- order(nearest, decreasing = TRUE)
    distant <- fastEmbedR:::massive_posting_distant_rows_cpp(
        source, centers, 4L, 3L, 2L)
    expect_setequal(distant, ranked[1:4])
    expect_equal(fastEmbedR:::massive_posting_distant_rows_cpp(
        source, centers, 1L, 5L, 1L), ranked[[1L]])
    expect_error(fastEmbedR:::massive_posting_distant_rows_cpp(
        source, centers, 17L, 5L, 1L), "invalid")
    expect_error(fastEmbedR:::massive_posting_distant_rows_cpp(
        source, centers, 4L, 5L, 1L, 3L), "invalid")
    expect_setequal(fastEmbedR:::massive_posting_distant_rows_cpp(
        source, centers, 4L, 5L, 1L, 16L), distant)
    graph <- massive_full_knn_graph(source, 2L, tempfile(),
        method = "coarse_postings", postings = postings,
        centers = centers, nprobe = 2L, n.cores = 2L,
        memory_limit = "128MB", audit_rows = 4L)
    expect_identical(graph$audit_sample_rows, c(4L, 8L, 9L, 16L))
    expect_identical(graph$audit_sampling,
        "evenly_spaced_and_distant_center_and_posting_lists")
    expect_identical(graph$audit_posting_lists_sampled, 1L)
    expect_length(graph$audit_row_recall, 4L)
    expect_equal(graph$observed_recall, 1)
})

test_that("posting audit samples a minority list", {
    values <- cbind(c(rep(0, 95L), rep(100, 5L)),
        seq_len(100L) / 100)
    input <- tempfile(fileext = ".f32")
    writeBin(as.vector(t(values)), input, size = 4L,
        endian = "little")
    source <- massive_matrix(input, 100L, 2L)
    centers <- values[c(1L, 96L), , drop = FALSE]
    postings <- massive_coarse_postings(source, centers, tempfile(),
        memory_limit = "128MB")
    expect_identical(postings$counts, c(95, 5))
    selection <- fastEmbedR:::massive_posting_audit_rows(
        source, centers, 4L, 16L, 1L, postings)
    expect_identical(selection$posting_lists_sampled, 1L)
    expect_true(any(selection$rows %in% 96:100))
    expect_length(selection$rows, 4L)
})

test_that("large posting audit scores bounded candidate rows", {
    source <- fastEmbedR:::massive_synthetic_matrix(1e9, 4L)
    centers <- rbind(rep(0, 4L), rep(1, 4L))
    selection <- fastEmbedR:::massive_posting_audit_rows(
        source, centers, 4L, 1024L, 2L)
    expect_identical(selection$scored_rows, 16L)
    expect_identical(selection$sampling,
        "evenly_spaced_and_distant_candidates")
    expect_length(selection$rows, 4L)
    expect_equal(length(unique(selection$rows)), 4L)
    expect_true(all(selection$rows >= 1L & selection$rows <= 1e9))
})

test_that("grouped posting graph resumes the existing query checkpoint", {
    values <- matrix(seq_len(180) / 19, 45L, 4L)
    values[1:15, 1L] <- values[1:15, 1L] - 100
    values[31:45, 1L] <- values[31:45, 1L] + 100
    centers <- values[c(3L, 22L, 40L), , drop = FALSE]
    input <- tempfile(fileext = ".f32")
    writeBin(as.vector(t(values)), input, size = 4L,
        endian = "little")
    source <- massive_matrix(input, 45L, 4L)
    postings <- massive_coarse_postings(source, centers, tempfile(),
        memory_limit = "128MB")
    prefix <- tempfile()
    original <- fastEmbedR:::massive_checkpoint_write
    interrupted <- function(state, path) {
        original(state, path)
        if (!is.null(state$completed_rows) &&
            state$completed_rows >= 7) stop("forced interruption")
    }
    with_mocked_bindings(
        massive_checkpoint_write = interrupted,
        expect_error(massive_full_knn_graph(source, 3L, prefix,
            method = "coarse_postings", postings = postings,
            centers = centers, nprobe = 1L, chunk_rows = 7L,
            reference_chunk_rows = 5L, memory_limit = "128MB",
            query_order = "posting", checkpoint = TRUE,
            checkpoint_every = 1L), "forced interruption"),
        .package = "fastEmbedR"
    )
    grouped <- massive_full_knn_graph(source, 3L, prefix,
        method = "coarse_postings", postings = postings,
        centers = centers, nprobe = 1L, chunk_rows = 7L,
        reference_chunk_rows = 5L, memory_limit = "128MB",
        query_order = "posting", checkpoint = TRUE,
        checkpoint_every = 1L, resume = TRUE)
    original_order <- massive_full_knn_graph(source, 3L, tempfile(),
        method = "coarse_postings", postings = postings,
        centers = centers, nprobe = 1L, chunk_rows = 7L,
        reference_chunk_rows = 5L, memory_limit = "128MB")
    expect_identical(unname(tools::md5sum(grouped$indices_path)),
        unname(tools::md5sum(original_order$indices_path)))
    expect_identical(unname(tools::md5sum(grouped$distances_path)),
        unname(tools::md5sum(original_order$distances_path)))
    expect_false(file.exists(paste0(prefix,
        ".grouped.checkpoint.rds")))
})

test_that("native grouped reorder validates rows and neighbors", {
    n <- 17L
    k <- 2L
    order <- c(8L, 1L, 17L, 4L, 11L, 2L, 15L, 6L, 13L,
        3L, 12L, 5L, 16L, 7L, 14L, 9L, 10L)
    ids <- cbind(order %% n + 1L, (order + 1L) %% n + 1L)
    distances <- cbind(order / 10, order / 5)
    root <- tempfile()
    dir.create(root)
    paths <- file.path(root, c("rows.u32", "ids.u32", "dist.f32",
        "output.ids.u32", "output.dist.f32"))
    write_files <- function(row_ids = order, neighbors = ids) {
        writeBin(as.integer(row_ids), paths[[1L]], size = 4L,
            endian = "little")
        writeBin(as.integer(t(neighbors)), paths[[2L]], size = 4L,
            endian = "little")
        writeBin(as.vector(t(distances)), paths[[3L]], size = 4L,
            endian = "little")
    }
    run <- function() fastEmbedR:::massive_reorder_posting_graph_cpp(
        paths[[1L]], paths[[2L]], paths[[3L]], paths[[4L]],
        paths[[5L]], n, k, 2L, 3L, FALSE)
    write_files()
    run()
    restored <- base::order(order)
    expected <- as.integer(t(ids[restored, , drop = FALSE]))
    observed <- readBin(paths[[4L]], "integer", n * k, size = 4L,
        endian = "little")
    expect_identical(observed, expected)
    expect_false(file.exists(paste0(paths[[4L]], ".reorder-work")))
    bad <- order
    bad[[1L]] <- bad[[2L]]
    write_files(row_ids = bad)
    expect_error(run(), "missing or duplicated")
    broken <- ids
    broken[1L, 1L] <- order[[1L]]
    write_files(neighbors = broken)
    expect_error(run(), "invalid neighbors")
})

test_that("grouped graph retries reorder without repeating search", {
    values <- matrix(seq_len(180) / 19, 45L, 4L)
    values[1:15, 1L] <- values[1:15, 1L] - 100
    values[31:45, 1L] <- values[31:45, 1L] + 100
    centers <- values[c(3L, 22L, 40L), , drop = FALSE]
    input <- tempfile(fileext = ".f32")
    writeBin(as.vector(t(values)), input, size = 4L,
        endian = "little")
    source <- massive_matrix(input, 45L, 4L)
    postings <- massive_coarse_postings(source, centers, tempfile(),
        memory_limit = "128MB")
    prefix <- tempfile()
    run <- function(resume = FALSE) massive_full_knn_graph(
        source, 3L, prefix, method = "coarse_postings",
        postings = postings, centers = centers, nprobe = 3L,
        chunk_rows = 7L, reference_chunk_rows = 5L,
        memory_limit = "128MB", query_order = "posting",
        checkpoint = TRUE, resume = resume)
    with_mocked_bindings(
        massive_reorder_posting_graph_cpp = function(...) {
            stop("forced reorder interruption")
        },
        expect_error(run(), "forced reorder interruption"),
        .package = "fastEmbedR"
    )
    expect_true(file.exists(paste0(prefix,
        ".grouped.indices.u32")))
    expect_true(file.exists(paste0(prefix,
        ".grouped.grouped-manifest.rds")))
    graph <- with_mocked_bindings(
        massive_full_knn_stream = function(...) {
            stop("search was repeated")
        },
        run(resume = TRUE), .package = "fastEmbedR")
    expect_identical(graph$query_order, "posting")
    expect_false(file.exists(paste0(prefix,
        ".grouped.indices.u32")))
})

test_that("grouped graph survives interruption after file commit", {
    values <- matrix(seq_len(180) / 19, 45L, 4L)
    centers <- values[c(3L, 22L, 40L), , drop = FALSE]
    input <- tempfile(fileext = ".f32")
    writeBin(as.vector(t(values)), input, size = 4L,
        endian = "little")
    source <- massive_matrix(input, 45L, 4L)
    postings <- massive_coarse_postings(source, centers, tempfile(),
        memory_limit = "128MB")
    prefix <- tempfile()
    run <- function(resume = FALSE) massive_full_knn_graph(
        source, 3L, prefix, method = "coarse_postings",
        postings = postings, centers = centers, nprobe = 3L,
        chunk_rows = 7L, reference_chunk_rows = 5L,
        memory_limit = "128MB", query_order = "posting",
        checkpoint = TRUE, resume = resume)
    original <- fastEmbedR:::massive_knn_finish
    with_mocked_bindings(
        massive_knn_finish = function(indices, distances, parts,
                paths, nrow, k) {
            original(indices, distances, parts, paths, nrow, k)
            if (grepl("\\.grouped\\.indices\\.u32$",
                    paths[["indices"]])) stop("forced post-commit stop")
        },
        expect_error(run(), "forced post-commit stop"),
        .package = "fastEmbedR"
    )
    expect_true(file.exists(paste0(prefix,
        ".grouped.indices.u32")))
    expect_true(file.exists(paste0(prefix,
        ".grouped.grouped-manifest.rds")))
    graph <- with_mocked_bindings(
        massive_full_knn_stream = function(...) {
            stop("search was repeated")
        },
        run(resume = TRUE), .package = "fastEmbedR")
    expect_identical(graph$query_order, "posting")
    expect_false(file.exists(paste0(prefix,
        ".grouped.indices.u32")))
    prefix <- tempfile()
    with_mocked_bindings(
        massive_knn_finish = function(indices, distances, parts,
                paths, nrow, k) {
            if (grepl("\\.grouped\\.indices\\.u32$",
                    paths[["indices"]])) stop("forced pre-commit stop")
            original(indices, distances, parts, paths, nrow, k)
        },
        expect_error(run(), "forced pre-commit stop"),
        .package = "fastEmbedR"
    )
    partial <- paste0(prefix, ".grouped.indices.u32.part")
    expect_true(file.exists(partial))
    modified <- file.info(partial)$mtime
    expect_true(Sys.setFileTime(partial, modified + 5))
    expect_error(run(resume = TRUE), "partial files changed")
    expect_true(Sys.setFileTime(partial, modified))
    if (identical(as.numeric(file.info(partial)$mtime),
            as.numeric(modified))) {
        graph <- with_mocked_bindings(
            massive_posting_graph_batch = function(...) {
                stop("search was repeated")
            },
            run(resume = TRUE), .package = "fastEmbedR")
        expect_identical(graph$query_order, "posting")
    } else {
        expect_error(run(resume = TRUE), "partial files changed")
    }
    prefix <- tempfile()
    with_mocked_bindings(
        massive_knn_finish = function(indices, distances, parts,
                paths, nrow, k) {
            if (grepl("\\.grouped\\.indices\\.u32$",
                    paths[["indices"]])) {
                expect_true(file.rename(parts[["indices"]],
                    paths[["indices"]]))
                stop("forced half-rename stop")
            }
            original(indices, distances, parts, paths, nrow, k)
        },
        expect_error(run(), "forced half-rename stop"),
        .package = "fastEmbedR"
    )
    expect_true(file.exists(paste0(prefix,
        ".grouped.indices.u32")))
    expect_true(file.exists(paste0(prefix,
        ".grouped.distances.f32.part")))
    graph <- with_mocked_bindings(
        massive_posting_graph_batch = function(...) {
            stop("search was repeated")
        },
        run(resume = TRUE), .package = "fastEmbedR")
    expect_identical(graph$query_order, "posting")
})

test_that("reordered graph recovers after its first final rename", {
    values <- matrix(seq_len(180) / 19, 45L, 4L)
    centers <- values[c(3L, 22L, 40L), , drop = FALSE]
    input <- tempfile(fileext = ".f32")
    writeBin(as.vector(t(values)), input, size = 4L,
        endian = "little")
    source <- massive_matrix(input, 45L, 4L)
    postings <- massive_coarse_postings(source, centers, tempfile(),
        memory_limit = "128MB")
    prefix <- tempfile()
    run <- function(resume = FALSE, checkpoint = TRUE)
        massive_full_knn_graph(
        source, 3L, prefix, method = "coarse_postings",
        postings = postings, centers = centers, nprobe = 3L,
        chunk_rows = 7L, reference_chunk_rows = 5L,
        memory_limit = "128MB", query_order = "posting",
        checkpoint = checkpoint, resume = resume)
    original <- fastEmbedR:::massive_knn_finish
    half_finish <- function(indices, distances, parts,
            paths, nrow, k) {
        if (!grepl("\\.grouped\\.indices\\.u32$",
                paths[["indices"]])) {
            expect_true(file.rename(parts[["indices"]],
                paths[["indices"]]))
            stop("forced final half-rename")
        }
        original(indices, distances, parts, paths, nrow, k)
    }
    with_mocked_bindings(
        massive_knn_finish = half_finish,
        expect_error(run(), "forced final half-rename"),
        .package = "fastEmbedR"
    )
    expect_true(file.exists(paste0(prefix, ".indices.u32")))
    expect_true(file.exists(paste0(prefix, ".distances.f32.part")))
    expect_true(file.exists(paste0(prefix, ".reorder-manifest.rds")))
    graph <- with_mocked_bindings(
        massive_posting_graph_batch = function(...) {
            stop("search was repeated")
        },
        massive_reorder_posting_graph_cpp = function(...) {
            stop("reorder was repeated")
        },
        run(resume = TRUE), .package = "fastEmbedR")
    expect_identical(graph$query_order, "posting")
    expect_true(file.exists(graph$indices_path))
    expect_true(file.exists(graph$distances_path))
    reopened <- with_mocked_bindings(
        massive_posting_graph_batch = function(...) {
            stop("search was repeated")
        },
        massive_reorder_posting_graph_cpp = function(...) {
            stop("reorder was repeated")
        },
        run(resume = TRUE), .package = "fastEmbedR")
    expect_identical(reopened$output_identity,
        graph$output_identity)
    stamp <- file.info(graph$indices_path)$mtime
    expect_true(Sys.setFileTime(graph$indices_path, stamp + 5))
    expect_error(run(resume = TRUE),
        "Completed posting graph does not match")
    expect_true(Sys.setFileTime(graph$indices_path, stamp))
    center <- centers[1L, 1L]
    centers[1L, 1L] <- centers[1L, 1L] + 1
    restored <- identical(as.numeric(file.info(graph$indices_path)$mtime),
        as.numeric(stamp))
    expect_error(run(resume = TRUE), if (restored) {
        "centers do not match"
    } else "Completed posting graph does not match")
    centers[1L, 1L] <- center
    prefix <- tempfile()
    complete_mark <- fastEmbedR:::massive_posting_complete_mark
    with_mocked_bindings(
        massive_posting_complete_mark = function(...) {
            complete_mark(...)
            stop("forced after completion mark")
        },
        expect_error(run(), "forced after completion mark"),
        .package = "fastEmbedR"
    )
    grouped <- paste0(prefix, ".grouped.indices.u32")
    expect_true(file.exists(grouped))
    modified <- file.info(grouped)$mtime
    expect_true(Sys.setFileTime(grouped, modified + 5))
    expect_error(run(resume = TRUE), "work files changed")
    expect_true(Sys.setFileTime(grouped, modified))
    if (identical(as.numeric(file.info(grouped)$mtime),
            as.numeric(modified))) {
        reopened <- with_mocked_bindings(
            massive_posting_graph_batch = function(...) {
                stop("search was repeated")
            },
            massive_reorder_posting_graph_cpp = function(...) {
                stop("reorder was repeated")
            },
            run(resume = TRUE), .package = "fastEmbedR")
        expect_identical(reopened$query_order, "posting")
        expect_false(file.exists(grouped))
    } else {
        expect_error(run(resume = TRUE), "work files changed")
    }
    prefix <- tempfile()
    with_mocked_bindings(
        massive_knn_finish = half_finish,
        expect_error(run(checkpoint = FALSE),
            "forced final half-rename"),
        .package = "fastEmbedR"
    )
    leftovers <- paste0(prefix, c(".indices.u32",
        ".distances.f32.part", ".reorder-manifest.rds"))
    expect_false(any(file.exists(leftovers)))
})

test_that("posting graph shrinks query blocks to its RAM budget", {
    values <- matrix(seq_len(4000) / 31, 1000L, 4L)
    centers <- values[c(50L, 500L, 950L), , drop = FALSE]
    input <- tempfile(fileext = ".f32")
    writeBin(as.vector(t(values)), input, size = 4L,
        endian = "little")
    source <- massive_matrix(input, 1000L, 4L)
    postings <- massive_coarse_postings(source, centers, tempfile(),
        memory_limit = "128MB")
    graph <- massive_full_knn_graph(source, 3L, tempfile(),
        method = "coarse_postings", postings = postings,
        centers = centers, nprobe = 3L, chunk_rows = 1000L,
        reference_chunk_rows = 256L, n.cores = 2L,
        memory_limit = "91.5MB")
    expect_lt(graph$resources$chunk_rows, 1000L)
    expect_gt(graph$resources$chunk_rows, 1L)
    expect_equal(graph$resources$reference_chunk_rows, 256L)
    expect_lte(graph$resources$peak_ram_bytes,
        0.7 * graph$resources$memory_limit_bytes)
    expect_true(graph$exact)
    expect_error(massive_full_knn_graph(source, 3L, tempfile(),
        method = "coarse_postings", postings = postings,
        centers = centers, nprobe = 3L, chunk_rows = 1000L,
        memory_limit = "90MB"), "memory_limit")
})

test_that("approximate posting graph checkpoints bind probe settings", {
    values <- matrix(seq_len(180) / 19, 45L, 4L)
    values[1:15, 1L] <- values[1:15, 1L] - 100
    values[31:45, 1L] <- values[31:45, 1L] + 100
    centers <- values[c(3L, 22L, 40L), , drop = FALSE]
    input <- tempfile(fileext = ".f32")
    writeBin(as.vector(t(values)), input, size = 4L,
        endian = "little")
    source <- massive_matrix(input, 45L, 4L)
    postings <- massive_coarse_postings(source, centers, tempfile(),
        memory_limit = "128MB")
    prefix <- tempfile()
    original <- fastEmbedR:::massive_checkpoint_write
    interrupted <- function(state, path) {
        original(state, path)
        if (!is.null(state$completed_rows) &&
            state$completed_rows >= 7) stop("forced interruption")
    }
    with_mocked_bindings(
        massive_checkpoint_write = interrupted,
        expect_error(massive_full_knn_graph(source, 3L, prefix,
            method = "coarse_postings", postings = postings,
            centers = centers, nprobe = 1L, chunk_rows = 7L,
            reference_chunk_rows = 5L, memory_limit = "128MB",
            checkpoint = TRUE, checkpoint_every = 1L),
            "forced interruption"),
        .package = "fastEmbedR"
    )
    expect_error(massive_full_knn_graph(source, 3L, prefix,
        method = "coarse_postings", postings = postings,
        centers = centers, nprobe = 2L, chunk_rows = 7L,
        reference_chunk_rows = 5L, memory_limit = "128MB",
        checkpoint = TRUE, checkpoint_every = 1L,
        resume = TRUE), "do not match")
    graph <- massive_full_knn_graph(source, 3L, prefix,
        method = "coarse_postings", postings = postings,
        centers = centers, nprobe = 1L, chunk_rows = 7L,
        reference_chunk_rows = 5L, memory_limit = "128MB",
        checkpoint = TRUE, checkpoint_every = 1L,
        resume = TRUE, audit_rows = 5L)
    expect_false(graph$exact)
    expect_true(graph$recall_audited)
    expect_true(graph$observed_recall >= 0)
    expect_true(graph$observed_recall <= 1)
    starts <- seq.int(1L, nrow(values), by = 7L)
    expected_reads <- sum(vapply(starts, function(first) {
        massive_search_postings(source, postings, centers,
            first = first, n = min(7L, nrow(values) - first + 1L),
            k = 3L, nprobe = 1L, reference_chunk = 5L,
            memory_limit = "128MB", exclude_self = TRUE)$posting_read_bytes
    }, 0.0))
    expect_equal(graph$resources$posting_read_bytes,
        expected_reads)
    expect_false(file.exists(paste0(prefix, ".checkpoint.rds")))
})
