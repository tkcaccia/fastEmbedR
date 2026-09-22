args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
    stop("usage: run-testthat-timed.R <output-directory>", call. = FALSE)
}

out_dir <- normalizePath(args[[1L]], mustWork = TRUE)
limit <- suppressWarnings(as.numeric(
    Sys.getenv("FASTEMBEDR_TESTTHAT_MAX_SECONDS", "180")
))
if (!is.finite(limit) || limit <= 0) {
    stop("FASTEMBEDR_TESTTHAT_MAX_SECONDS must be positive.", call. = FALSE)
}

timing <- system.time({
    testthat::test_dir(
        "tests/testthat",
        reporter = "summary",
        package = "fastEmbedR",
        load_package = "installed",
        stop_on_failure = TRUE
    )
})
summary <- data.frame(
    elapsed_sec = unname(timing[["elapsed"]]),
    user_sec = unname(timing[["user.self"]]),
    system_sec = unname(timing[["sys.self"]]),
    limit_sec = limit
)
utils::write.csv(
    summary,
    file.path(out_dir, "testthat-timing.csv"),
    row.names = FALSE
)
if (summary$elapsed_sec > limit) {
    stop(
        sprintf(
            "testthat elapsed %.3f seconds; limit is %.3f seconds.",
            summary$elapsed_sec,
            limit
        ),
        call. = FALSE
    )
}
