test_that("local seed restores the caller's RNG state", {
    had_seed <- exists(".Random.seed", envir = .GlobalEnv,
        inherits = FALSE)
    old_seed <- if (had_seed) get(".Random.seed", envir = .GlobalEnv)
    on.exit({
        if (had_seed) {
            assign(".Random.seed", old_seed, envir = .GlobalEnv)
        } else if (exists(".Random.seed", envir = .GlobalEnv,
            inherits = FALSE)) {
            rm(".Random.seed", envir = .GlobalEnv)
        }
    }, add = TRUE)

    set.seed(123)
    prior <- .Random.seed
    restore <- set_local_seed(99)
    observed <- stats::runif(3)
    restore()
    expect_identical(.Random.seed, prior)
    set.seed(99)
    expect_identical(observed, stats::runif(3))

    rm(".Random.seed", envir = .GlobalEnv)
    restore <- set_local_seed(99)
    expect_true(exists(".Random.seed", envir = .GlobalEnv,
        inherits = FALSE))
    restore()
    expect_false(exists(".Random.seed", envir = .GlobalEnv,
        inherits = FALSE))
})
