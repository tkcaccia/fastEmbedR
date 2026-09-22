test_that(paste(
    "float32 exact t-SNE forces agree with an independent",
    "float64 reference"
), {
    set.seed(701)
    x <- matrix(rnorm(48L * 6L), 48L, 6L)
    knn <- test_exact_knn(x, k = 15L, exclude_self = TRUE)
    layout <- matrix(rnorm(48L * 2L, sd = 0.3), 48L, 2L)
    reference_probability <- tsne_reference_probabilities(
        knn$indices, knn$distances,
        perplexity = 5
    )
    reference <- tsne_reference_forces(reference_probability, layout)
    diagnostic <- fastEmbedR:::opentsne_force_diagnostic_cpp(
        knn$indices, knn$distances, layout, 5, 1, 64L, 2L
    )

    expect_equal(
        tsne_csr_probability_matrix(diagnostic),
        reference_probability,
        tolerance = 2e-6
    )
    expect_lt(tsne_relative_l2(
        diagnostic$attractive_force, reference$attractive_force
    ), 5e-5)
    expect_lt(tsne_relative_l2(
        diagnostic$repulsive_force_exact, reference$repulsive_force
    ), 5e-5)
    expect_lt(tsne_relative_l2(
        diagnostic$objective_gradient_exact, reference$objective_gradient
    ), 1e-4)
    expect_equal(diagnostic$sum_q, reference$sum_q, tolerance = 2e-5)
    expect_equal(diagnostic$kl, reference$kl, tolerance = 5e-6)
})

test_that("the exact objective gradient passes a finite-difference check", {
    set.seed(702)
    x <- matrix(rnorm(18L * 5L), 18L, 5L)
    knn <- test_exact_knn(x, k = 9L, exclude_self = TRUE)
    layout <- matrix(rnorm(18L * 2L, sd = 0.2), 18L, 2L)
    probability <- tsne_reference_probabilities(
        knn$indices, knn$distances,
        perplexity = 3
    )
    reference <- tsne_reference_forces(probability, layout)
    diagnostic <- fastEmbedR:::opentsne_force_diagnostic_cpp(
        knn$indices, knn$distances, layout, 3, 1, 64L, 1L
    )

    epsilon <- 1e-6
    numerical <- matrix(0, nrow(layout), ncol(layout))
    for (index in seq_along(layout)) {
        upper <- lower <- layout
        upper[index] <- upper[index] + epsilon
        lower[index] <- lower[index] - epsilon
        numerical[index] <- (
            tsne_reference_forces(probability, upper)$kl -
                tsne_reference_forces(probability, lower)$kl
        ) / (2 * epsilon)
    }

    expect_lt(tsne_relative_l2(reference$objective_gradient, numerical), 2e-7)
    expect_lt(tsne_relative_l2(
        diagnostic$objective_gradient_exact, numerical
    ), 2e-4)
})

test_that("FFT-grid repulsion converges to exact forces as the grid grows", {
    set.seed(703)
    x <- matrix(rnorm(48L * 6L), 48L, 6L)
    knn <- test_exact_knn(x, k = 15L, exclude_self = TRUE)
    layout <- matrix(rnorm(48L * 2L, sd = 0.3), 48L, 2L)
    reference_probability <- tsne_reference_probabilities(
        knn$indices, knn$distances,
        perplexity = 5
    )
    reference <- tsne_reference_forces(reference_probability, layout)

    limits <- c(`32` = 1e-2, `64` = 3e-3, `128` = 1e-3)
    errors <- numeric(length(limits))
    for (position in seq_along(limits)) {
        grid <- as.integer(names(limits)[position])
        diagnostic <- fastEmbedR:::opentsne_force_diagnostic_cpp(
            knn$indices, knn$distances, layout, 5, 1, grid, 2L
        )
        errors[position] <- tsne_relative_l2(
            diagnostic$repulsive_force_fft, reference$repulsive_force
        )
        expect_lt(errors[position], limits[position])
        expect_gt(cor(
            c(diagnostic$repulsive_force_fft), c(reference$repulsive_force)
        ), 0.999)
    }
    expect_true(all(diff(errors) < 0))
})

test_that("exact force agreement holds across perplexity and support widths", {
    set.seed(704)
    x <- matrix(rnorm(40L * 6L), 40L, 6L)
    layout <- matrix(rnorm(40L * 2L, sd = 0.25), 40L, 2L)
    for (perplexity in c(2L, 4L, 8L)) {
        for (multiplier in c(1L, 3L, 4L)) {
            k <- perplexity * multiplier
            knn <- test_exact_knn(x, k = k, exclude_self = TRUE)
            probability <- tsne_reference_probabilities(
                knn$indices, knn$distances, perplexity
            )
            reference <- tsne_reference_forces(probability, layout)
            diagnostic <- fastEmbedR:::opentsne_force_diagnostic_cpp(
                knn$indices, knn$distances, layout, perplexity, 1, 64L, 2L
            )
            expect_lt(tsne_relative_l2(
                diagnostic$objective_gradient_exact,
                reference$objective_gradient
            ), 2e-4)
        }
    }
})

test_that("t-SNE diagnostics remain finite on pathological valid inputs", {
    set.seed(705)
    x <- rbind(
        matrix(0, 8L, 4L),
        matrix(1, 8L, 4L),
        matrix(rnorm(8L * 4L, sd = 1e-8), 8L, 4L)
    )
    knn <- test_exact_knn(x, k = 6L, exclude_self = TRUE)
    disconnected_x <- rbind(
        matrix(rnorm(12L * 4L, -20, 0.01), 12L, 4L),
        matrix(rnorm(12L * 4L, 20, 0.01), 12L, 4L)
    )
    disconnected_knn <- test_exact_knn(
        disconnected_x,
        k = 6L, exclude_self = TRUE
    )
    cases <- list(
        duplicated = list(
            knn = knn,
            layout = matrix(rnorm(24L * 2L, sd = 0.1), 24L, 2L)
        ),
        coincident = list(knn = knn, layout = matrix(0, 24L, 2L)),
        extreme = list(
            knn = knn,
            layout = matrix(rep(c(-1e4, 1e4), 24L), 24L, 2L, byrow = TRUE)
        ),
        disconnected = list(
            knn = disconnected_knn,
            layout = matrix(rnorm(24L * 2L, sd = 0.1), 24L, 2L)
        )
    )
    for (case in cases) {
        diagnostic <- fastEmbedR:::opentsne_force_diagnostic_cpp(
            case$knn$indices, case$knn$distances, case$layout, 2, 1, 64L, 2L
        )
        expect_true(all(is.finite(diagnostic$objective_gradient_exact)))
        expect_true(all(is.finite(diagnostic$objective_gradient_fft)))
        expect_true(is.finite(diagnostic$kl))
        expect_gt(diagnostic$sum_q, 0)
    }

    available <- c(
        cpu = TRUE,
        metal = isTRUE(fastEmbedR:::embedding_metal_available_cpp()) &&
            isTRUE(fastEmbedR:::metal_opentsne_native_available()),
        cuda = isTRUE(fastEmbedR:::embedding_cuda_available_cpp()) &&
            isTRUE(fastEmbedR:::cuda_opentsne_native_available())
    )
    for (backend in names(available)[available]) {
        for (case in cases) {
            case_knn <- case$knn
            if (backend == "cuda") {
                expect_true(requireNamespace("float", quietly = TRUE))
                case_knn$distances <- float::fl(case_knn$distances)
            }
            observed <- tsne_knn(
                case_knn,
                perplexity = 2,
                Y_init = case$layout,
                early_exaggeration_iter = 0L,
                n_iter = 1L,
                early_exaggeration = 1,
                exaggeration = 1,
                learning_rate = 1,
                initial_momentum = 0,
                final_momentum = 0,
                max_step_norm = 1e6,
                negative_gradient_method = "fft",
                auto_config = FALSE,
                backend = backend,
                n.cores = 2L,
                seed = 705L
            )
            materialized <- if (inherits(observed, "float32")) {
                float::dbl(observed)
            } else {
                as.matrix(observed)
            }
            expect_true(all(is.finite(materialized)))
            expect_identical(
                attr(observed, "fastEmbedR_config")$backend,
                backend
            )
        }
    }

    invalid <- cases$duplicated$layout
    invalid[1, 1] <- Inf
    expect_error(
        fastEmbedR:::opentsne_force_diagnostic_cpp(
            knn$indices, knn$distances, invalid, 2, 1, 64L, 1L
        ),
        "finite"
    )
    for (backend in names(available)[available]) {
        invalid_knn <- knn
        if (backend == "cuda") invalid_knn$distances <- float::fl(knn$distances)
        expect_error(
            tsne_knn(
                invalid_knn,
                perplexity = 2,
                Y_init = invalid,
                early_exaggeration_iter = 0L,
                n_iter = 1L,
                negative_gradient_method = "fft",
                auto_config = FALSE,
                backend = backend,
                n.cores = 2L,
                seed = 705L
            ),
            "finite"
        )
    }
})

test_that("native Metal and CUDA match the CPU first optimizer step", {
    set.seed(706)
    x <- matrix(rnorm(64L * 6L), 64L, 6L)
    knn <- test_exact_knn(x, k = 15L, exclude_self = TRUE)
    initial <- matrix(rnorm(64L * 2L, sd = 1e-4), 64L, 2L)

    old_grid <- Sys.getenv("FASTEMBEDR_TSNE_FFT_GRID", unset = NA_character_)
    Sys.setenv(FASTEMBEDR_TSNE_FFT_GRID = "64")
    on.exit(
        {
            if (is.na(old_grid)) {
                Sys.unsetenv("FASTEMBEDR_TSNE_FFT_GRID")
            } else {
                Sys.setenv(FASTEMBEDR_TSNE_FFT_GRID = old_grid)
            }
        },
        add = TRUE
    )

    run_step <- function(backend, negative_gradient_method) {
        backend_knn <- knn
        if (backend == "cuda") {
            expect_true(requireNamespace("float", quietly = TRUE))
            backend_knn$distances <- float::fl(backend_knn$distances)
        }
        tsne_knn(
            backend_knn,
            perplexity = 5,
            Y_init = initial,
            early_exaggeration_iter = 0L,
            n_iter = 1L,
            early_exaggeration = 1,
            exaggeration = 1,
            learning_rate = 1,
            initial_momentum = 0,
            final_momentum = 0,
            max_step_norm = 1e6,
            negative_gradient_method = negative_gradient_method,
            auto_config = FALSE,
            backend = backend,
            n.cores = 2L,
            seed = 706L
        )
    }

    cpu <- list(
        exact = run_step("cpu", "exact"),
        fft = run_step("cpu", "fft")
    )
    available <- list(
        metal = isTRUE(fastEmbedR:::embedding_metal_available_cpp()) &&
            isTRUE(fastEmbedR:::metal_opentsne_native_available()),
        cuda = isTRUE(fastEmbedR:::embedding_cuda_available_cpp()) &&
            isTRUE(fastEmbedR:::cuda_opentsne_native_available())
    )

    for (backend in names(available)[unlist(available)]) {
        fft <- run_step(backend, "fft")
        if (backend != "cuda") {
            exact <- run_step(backend, "exact")
            expect_lt(tsne_relative_l2(exact, cpu$exact), 5e-4,
                label = paste(backend, "exact first-step relative L2")
            )
        }
        expect_lt(tsne_relative_l2(fft, cpu$fft), 2e-3,
            label = paste(backend, "FFT first-step relative L2")
        )
    }

    if (!any(unlist(available))) {
        succeed(paste(
            "GPU first-step checks require a compiled Metal or",
            "CUDA backend"
        ))
    }
})

test_that("FFT t-SNE retains exact-objective agreement over a long run", {
    set.seed(719)
    x <- matrix(rnorm(180L * 8L), 180L, 8L)
    knn <- test_exact_knn(x, k = 10L, exclude_self = TRUE)
    initial <- matrix(rnorm(180L * 2L, sd = 1e-4), 180L, 2L)

    old_grid <- Sys.getenv("FASTEMBEDR_TSNE_FFT_GRID", unset = NA_character_)
    Sys.setenv(FASTEMBEDR_TSNE_FFT_GRID = "64")
    on.exit(
        {
            if (is.na(old_grid)) {
                Sys.unsetenv("FASTEMBEDR_TSNE_FFT_GRID")
            } else {
                Sys.setenv(FASTEMBEDR_TSNE_FFT_GRID = old_grid)
            }
        },
        add = TRUE
    )

    run_long <- function(backend, method) {
        tsne_knn(
            knn,
            perplexity = 10,
            Y_init = initial,
            seed = 719L,
            backend = backend,
            n.cores = 2L,
            learning_rate = 15,
            early_exaggeration_iter = 75L,
            early_exaggeration = 12,
            n_iter = 225L,
            exaggeration = 1,
            initial_momentum = 0.8,
            final_momentum = 0.8,
            max_step_norm = "auto",
            negative_gradient_method = method,
            auto_config = FALSE
        )
    }
    objective <- function(layout) {
        fastEmbedR:::opentsne_kl_diagnostic_cpp(
            knn$indices, knn$distances, layout, 10, 2L
        )
    }

    reference_kl <- objective(run_long("cpu", "exact"))
    cpu_fft_kl <- objective(run_long("cpu", "fft"))
    expect_true(is.finite(reference_kl))
    expect_lte(cpu_fft_kl, 1.05 * reference_kl)

    if (isTRUE(fastEmbedR:::embedding_metal_available_cpp())) {
        metal_fft_kl <- objective(run_long("metal", "fft"))
        expect_lte(metal_fft_kl, 1.05 * reference_kl)
    }
})

test_that("automatic t-SNE step clipping is backend invariant", {
    expect_identical(
        fastEmbedR:::resolve_opentsne_max_step_norm("auto"),
        5
    )
})

test_that("explicit small-data CPU FFT records the validated grid floor", {
    set.seed(707)
    x <- matrix(rnorm(64L * 6L), 64L, 6L)
    knn <- test_exact_knn(x, k = 15L, exclude_self = TRUE)
    initial <- matrix(rnorm(64L * 2L, sd = 1e-4), 64L, 2L)

    old_grid <- Sys.getenv("FASTEMBEDR_TSNE_FFT_GRID", unset = NA_character_)
    Sys.unsetenv("FASTEMBEDR_TSNE_FFT_GRID")
    on.exit(
        {
            if (is.na(old_grid)) {
                Sys.unsetenv("FASTEMBEDR_TSNE_FFT_GRID")
            } else {
                Sys.setenv(FASTEMBEDR_TSNE_FFT_GRID = old_grid)
            }
        },
        add = TRUE
    )

    fit <- tsne_knn(
        knn,
        perplexity = 5,
        Y_init = initial,
        early_exaggeration_iter = 0L,
        n_iter = 1L,
        early_exaggeration = 1,
        exaggeration = 1,
        learning_rate = 1,
        initial_momentum = 0,
        final_momentum = 0,
        max_step_norm = 1e6,
        negative_gradient_method = "fft",
        auto_config = FALSE,
        backend = "cpu",
        n.cores = 1L,
        seed = 707L
    )

    expect_identical(
        attr(fit, "fastEmbedR_config")$fft_grid_size,
        128L
    )
})

test_that("explicit small-data Metal FFT records the validated grid floor", {
    skip_if_not(isTRUE(fastEmbedR:::embedding_metal_available_cpp()))

    set.seed(708)
    x <- matrix(rnorm(64L * 6L), 64L, 6L)
    knn <- test_exact_knn(x, k = 15L, exclude_self = TRUE)
    initial <- matrix(rnorm(64L * 2L, sd = 1e-4), 64L, 2L)

    old_grid <- Sys.getenv("FASTEMBEDR_TSNE_FFT_GRID", unset = NA_character_)
    Sys.unsetenv("FASTEMBEDR_TSNE_FFT_GRID")
    on.exit(
        {
            if (is.na(old_grid)) {
                Sys.unsetenv("FASTEMBEDR_TSNE_FFT_GRID")
            } else {
                Sys.setenv(FASTEMBEDR_TSNE_FFT_GRID = old_grid)
            }
        },
        add = TRUE
    )

    fit <- tsne_knn(
        knn,
        perplexity = 5,
        Y_init = initial,
        early_exaggeration_iter = 0L,
        n_iter = 1L,
        early_exaggeration = 1,
        exaggeration = 1,
        learning_rate = 1,
        initial_momentum = 0,
        final_momentum = 0,
        max_step_norm = 1e6,
        negative_gradient_method = "fft",
        auto_config = FALSE,
        backend = "metal",
        n.cores = 1L,
        seed = 708L
    )

    expect_identical(
        attr(fit, "fastEmbedR_config")$fft_grid_size,
        128L
    )
})
