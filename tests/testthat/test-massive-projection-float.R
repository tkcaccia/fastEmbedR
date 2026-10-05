test_that("saved float32 reference dimensions are registered", {
    skip_if_not_installed("float")
    package_dir <- find.package("fastEmbedR")
    if (!file.exists(file.path(package_dir, "Meta", "package.rds"))) {
        skip("Requires an installed fastEmbedR package")
    }
    layout <- float::fl(matrix(seq_len(20L), ncol = 2L))
    fit <- structure(list(method = "umap", layout = layout),
        class = "fastEmbedR_embedding")
    graph <- structure(list(n_reference = 10L, ncol = 3L,
        metric = "euclidean", backend = "cpu"),
        class = "fastEmbedR_massive_knn")
    paths <- tempfile(fileext = c(".rds", ".rds", ".R"))
    on.exit(unlink(paths))
    saveRDS(fit, paths[[1L]])
    saveRDS(graph, paths[[2L]])
    writeLines(c(
        "args <- commandArgs(TRUE)",
        "library(fastEmbedR, lib.loc = args[[1L]])",
        "fit <- readRDS(args[[2L]])",
        "graph <- readRDS(args[[3L]])",
        "stopifnot(!isNamespaceLoaded('float'))",
        "controls <- fastEmbedR:::massive_projection_controls(",
        "    fit, graph, 'cpu', 0L, 0L, 5, 4L)",
        "stopifnot(controls$epochs == 0L, nrow(fit$layout) == 10L)"
    ), paths[[3L]])
    binary <- if (.Platform$OS.type == "windows") {
        "Rscript.exe"
    } else {
        "Rscript"
    }
    result <- suppressWarnings(system2(
        file.path(R.home("bin"), binary),
        c("--vanilla", shQuote(paths[[3L]]),
            shQuote(dirname(find.package("fastEmbedR"))),
            shQuote(paths[[1L]]), shQuote(paths[[2L]])),
        stdout = TRUE, stderr = TRUE))
    expect_identical(attr(result, "status"), NULL,
        info = paste(result, collapse = "\n"))
})
