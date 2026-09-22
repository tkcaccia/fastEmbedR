test_that("JSS manuscript examples execute against the installed package", {
    path <- system.file(
        "examples", "jss_manuscript.R", package = "fastEmbedR"
    )
    expect_true(nzchar(path))

    example_env <- new.env(parent = globalenv())
    expect_silent(sys.source(path, envir = example_env))
    expect_s3_class(example_env$fit_tsne, "fastEmbedR_embedding")
    expect_s3_class(example_env$fit_umap, "fastEmbedR_embedding")
    expect_s3_class(example_env$knn, "fastEmbedR_knn")
    expect_s3_class(example_env$model, "fastEmbedR_landmark_model")
})
