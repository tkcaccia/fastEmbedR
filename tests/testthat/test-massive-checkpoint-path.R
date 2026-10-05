test_that("checkpoint paths use one cross-platform identity", {
    first <- tempfile("fastembedr_path_a_")
    second <- tempfile("fastembedr_path_b_")
    on.exit(unlink(c(first, second)))
    writeBin(as.raw(1L), first)
    writeBin(as.raw(2L), second)
    forward <- normalizePath(first, winslash = "/")
    native <- normalizePath(first, winslash = "\\")
    expect_true(fastEmbedR:::massive_checkpoint_same_paths(
        forward, native))
    expect_false(fastEmbedR:::massive_checkpoint_same_paths(
        first, second))
})
