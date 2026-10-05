test_that("file-backed Leiden refinement preserves parent groups", {
    skip_on_os("windows")
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    source <- file.path(directory, "knn")
    fuzzy <- file.path(directory, "fuzzy")
    parent_path <- file.path(directory, "parent.u32")
    first <- file.path(directory, "refined.u32")
    second <- file.path(directory, "repeat.u32")
    neighbors <- t(vapply(seq_len(8L), function(row) {
        setdiff(if (row <= 4L) 1:4 else 5:8, row)
    }, integer(3L)))
    distances <- matrix(rep(c(0.1, 0.3, 0.7), 8L),
        nrow = 8L, byrow = TRUE)
    writeBin(as.integer(t(neighbors)), paste0(source,
        ".indices.u32"), size = 4L, endian = "little")
    writeBin(as.numeric(t(distances)), paste0(source,
        ".distances.f32"), size = 4L, endian = "little")
    knn <- massive_knn_graph(source, 8L, 3L)
    graph <- massive_umap_fuzzy_graph(
        massive_umap_memberships(knn), fuzzy, memory_limit = "64MB")
    parent <- massive_louvain_level(graph, parent_path, seed = 4L,
        chunk_rows = 2L, memory_limit = "128MB")
    expect_error(massive_leiden_refine_level(graph, parent, first,
        backend = "cuda"), "no fallback")
    fit <- massive_leiden_refine_level(graph, parent, first,
        seed = 4L, chunk_rows = 2L, memory_limit = "128MB")
    again <- massive_leiden_refine_level(graph, parent, second,
        seed = 4L, chunk_rows = 2L, memory_limit = "128MB")
    read_labels <- function(path, count) {
        readBin(path, integer(), n = count, size = 4L,
            endian = "little")
    }
    labels <- read_labels(fit$membership_path, 8L)
    mapping <- read_labels(fit$parent_mapping_path, fit$n_refined)
    expect_identical(labels,
        read_labels(again$membership_path, 8L))
    expect_identical(mapping[labels],
        massive_read_louvain_rows(parent, 1L, 8L))
    expect_gte(fit$n_refined, parent$n_communities)
    expect_lte(fit$n_refined, 8L)
    expect_identical(fit$backend, "cpu")
    expect_identical(fit$method, "leiden_refinement_level")
    expect_false(file.exists(paste0(first, ".sizes.part")))
    expect_error(massive_leiden_refine_level(graph, parent, first),
        "already exists")
})

test_that("file-backed Leiden contracts and preserves original rows", {
    skip_on_os("windows")
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    n <- 512L
    k <- 6L
    source <- file.path(directory, "knn")
    fuzzy <- file.path(directory, "fuzzy")
    rows <- seq_len(n) - 1L
    neighbors <- outer(rows, seq_len(k), function(row, step) {
        (row + step) %% n + 1L
    })
    writeBin(as.integer(t(neighbors)), paste0(source,
        ".indices.u32"), size = 4L, endian = "little")
    writeBin(rep(as.numeric(seq_len(k)), n), paste0(source,
        ".distances.f32"), size = 4L, endian = "little")
    graph <- massive_umap_fuzzy_graph(massive_umap_memberships(
        massive_knn_graph(source, n, k)), fuzzy,
        memory_limit = "64MB")
    output <- file.path(directory, "leiden.u32")
    fit <- massive_leiden(graph, output, seed = 4L,
        chunk_rows = 64L, memory_limit = "128MB")
    labels <- massive_read_leiden_rows(fit, 1L, n)
    expect_gt(length(fit$levels), 1L)
    expect_identical(length(labels), n)
    expect_identical(sort(unique(labels)),
        seq_len(fit$n_communities))
    expect_gt(fit$modularity_final, 0.5)
    expect_equal(fit$modularity_final,
        massive_graph_modularity(graph, output,
            fit$n_communities, memory_limit = "128MB")$modularity,
        tolerance = 1e-6)
    edges <- massive_read_graph_edges(graph, 1L, n)
    resident <- graph_cluster(list(from = edges$from, to = edges$to,
        weight = edges$weight, n_vertices = n),
        method = "leiden", seed = 4L)
    expect_gte(fit$modularity_final, resident$modularity - 0.05)
    contingency <- table(labels, resident$membership)
    pairs <- function(counts) sum(counts * (counts - 1) / 2)
    left <- pairs(rowSums(contingency))
    right <- pairs(colSums(contingency))
    expected <- left * right / choose(n, 2)
    ari <- (pairs(contingency) - expected) /
        (0.5 * (left + right) - expected)
    expect_gt(ari, 0.2)
    adjacency <- split(edges$to, edges$from)
    for (group in split(seq_len(n), labels)) {
        seen <- rep(FALSE, n)
        queue <- group[[1L]]
        seen[queue] <- TRUE
        position <- 1L
        while (position <= length(queue)) {
            nearby <- adjacency[[queue[[position]]]]
            nearby <- nearby[labels[nearby] == labels[queue[[1L]]]]
            nearby <- nearby[!seen[nearby]]
            seen[nearby] <- TRUE
            queue <- c(queue, nearby)
            position <- position + 1L
        }
        expect_true(all(seen[group]))
    }
    expect_error(massive_leiden(graph, tempfile(),
        backend = "cuda"), "no fallback")
    public_path <- file.path(directory, "public.u32")
    public <- massive_cluster(graph, method = "leiden",
        massive = "out_of_core_graph", output = public_path,
        n_iterations = 10L, seed = 4L, memory_limit = "128MB",
        checkpoint = TRUE)
    expect_identical(public$mode, "out_of_core_graph")
    expect_false(file.exists(paste0(public_path,
        ".checkpoint.rds")))
    expect_identical(head(fit, 4L), labels[1:4])
})

test_that("Leiden resumes a verified completed graph level", {
    skip_on_os("windows")
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    n <- 96L
    source <- file.path(directory, "knn")
    neighbors <- outer(seq_len(n) - 1L, 1:4, function(row, step)
        (row + step) %% n + 1L)
    writeBin(as.integer(t(neighbors)), paste0(source,
        ".indices.u32"), size = 4L, endian = "little")
    writeBin(rep(as.numeric(1:4), n), paste0(source,
        ".distances.f32"), size = 4L, endian = "little")
    graph <- massive_umap_fuzzy_graph(massive_umap_memberships(
        massive_knn_graph(source, n, 4L)),
        file.path(directory, "fuzzy"), memory_limit = "64MB")
    output <- file.path(directory, "resumed.u32")
    controls <- fastEmbedR:::massive_louvain_controls(10L, 1, 4L)
    resources <- fastEmbedR:::massive_leiden_output(graph, output,
        "128MB", FALSE)
    signature <- fastEmbedR:::massive_leiden_signature(graph,
        output, controls, 32L, "128MB")
    prefix <- paste0(output, ".level0")
    step <- fastEmbedR:::massive_leiden_run_level(graph, NULL,
        prefix, 0L, 10L, 1, 4L, 32L, "128MB")
    expect_false(is.null(step$refined))
    part <- paste0(output, ".part")
    expect_true(file.copy(step$mapping, part))
    contracted <- fastEmbedR:::massive_louvain_contract(graph,
        step$refined, prefix, resources)
    current <- fastEmbedR:::massive_louvain_coarse_csr(
        contracted, paste0(prefix, ".csr"),
        resources$memory_limit_bytes)
    levels <- list(list(vertices = n,
        parent = step$parent$n_communities,
        parent_modularity = step$parent$modularity_final,
        refined = step$refined$n_refined))
    fastEmbedR:::massive_leiden_save(output, signature, "next",
        1L, current, step$refined, part, levels,
        step$parent$modularity_final)
    incomplete <- paste0(output, ".level1.parent.u32.part")
    expect_true(file.create(incomplete))
    expect_error(massive_leiden(graph, output, seed = 4L,
        chunk_rows = 32L, memory_limit = "128MB",
        checkpoint = TRUE, resume = TRUE), "conflicts")
    unlink(incomplete)
    expect_error(massive_leiden(graph, output, seed = 4L,
        max_passes = 9L, chunk_rows = 32L,
        memory_limit = "128MB", checkpoint = TRUE,
        resume = TRUE), "checkpoint")
    fit <- massive_leiden(graph, output, seed = 4L,
        chunk_rows = 32L, memory_limit = "128MB",
        checkpoint = TRUE, resume = TRUE)
    fresh <- massive_leiden(graph,
        file.path(directory, "fresh.u32"), seed = 4L,
        chunk_rows = 32L, memory_limit = "128MB")
    expect_identical(massive_read_leiden_rows(fit, 1L, n),
        massive_read_leiden_rows(fresh, 1L, n))
    expect_equal(fit$modularity_final, fresh$modularity_final)
    expect_false(file.exists(paste0(output, ".checkpoint.rds")))
    ready_path <- file.path(directory, "ready.u32")
    ready_resources <- fastEmbedR:::massive_leiden_output(
        graph, ready_path, "128MB", FALSE)
    ready_signature <- fastEmbedR:::massive_leiden_signature(
        graph, ready_path, controls, 32L, "128MB")
    fastEmbedR:::massive_leiden_hierarchy(graph, ready_path,
        controls, 32L, "128MB", ready_resources,
        ready_signature)
    expect_identical(readRDS(paste0(ready_path,
        ".checkpoint.rds"))$stage, "ready")
    ready <- massive_leiden(graph, ready_path, seed = 4L,
        chunk_rows = 32L, memory_limit = "128MB",
        checkpoint = TRUE, resume = TRUE)
    expect_identical(massive_read_leiden_rows(ready, 1L, n),
        massive_read_leiden_rows(fresh, 1L, n))
    expect_false(file.exists(paste0(ready_path,
        ".checkpoint.rds")))
})
