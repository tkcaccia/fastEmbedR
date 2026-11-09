test_that("Metal 3D t-SNE tiles retain global offsets past int32", {
    compiler <- Sys.which("c++")
    skip_if(!nzchar(compiler), "C++ compiler unavailable")
    source <- testthat::test_path("..",
        "test_metal_tsne_3d_tiles.cpp")
    skip_if(!file.exists(source), "native tile fixture unavailable")
    output <- tempfile("metal-tsne-tiles-")
    on.exit(unlink(output), add = TRUE)
    build <- suppressWarnings(system2(compiler,
        c("-std=c++17", "-O2", shQuote(source), "-o",
            shQuote(output)), stdout = TRUE, stderr = TRUE))
    status <- attr(build, "status")
    if (is.null(status)) status <- 0L
    expect_identical(status, 0L,
        info = paste(build, collapse = "\n"))
    if (!file.exists(output)) return(invisible())
    run <- suppressWarnings(system2(output,
        stdout = TRUE, stderr = TRUE))
    status <- attr(run, "status")
    if (is.null(status)) status <- 0L
    expect_identical(status, 0L,
        info = paste(run, collapse = "\n"))
})
