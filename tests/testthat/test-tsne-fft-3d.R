test_that("3D FFT repulsion agrees with exact forces", {
    set.seed(419)
    n <- 256L
    k <- 12L
    indices <- outer(
        seq_len(n), seq_len(k),
        function(i, j) (i + j - 1L) %% n + 1L
    )
    distances <- matrix(rep(seq_len(k) / 10, each = n), n, k)
    for (grid in c(16L, 32L)) {
        for (spread in c(1, 10, 50)) {
            layout <- matrix(rnorm(n * 3L, sd = spread), n, 3L)
            result <- fastEmbedR:::tsne_fft_3d_force_diagnostic_cpp(
                indices, distances, layout, 5, 1, grid, 2L
            )
            error <- sqrt(sum((result$repulsive_fft -
                result$repulsive_exact)^2)) /
                sqrt(sum(result$repulsive_exact^2))
            expect_lt(error, 0.03)
            expect_lt(abs(result$sum_q_fft / result$sum_q_exact - 1),
                if (grid == 16L) 0.003 else 0.001)
        }
    }
    duplicate <- fastEmbedR:::tsne_fft_3d_force_diagnostic_cpp(
        indices, distances, matrix(0, n, 3L), 5, 1, 16L, 2L
    )
    expect_equal(duplicate$sum_q_fft, n * (n - 1L), tolerance = 1e-4)
    expect_lt(max(abs(duplicate$repulsive_fft)), 1e-7)
})

test_that("3D CPU auto routing uses FFT", {
    n <- 3201L
    k <- 6L
    indices <- outer(
        seq_len(n), seq_len(k),
        function(i, j) (i + j - 1L) %% n + 1L
    )
    distances <- matrix(rep(seq_len(k) / 10, each = n), n, k)
    set.seed(420)
    init <- matrix(rnorm(n * 3L, sd = 10), n, 3L)
    fit <- tsne_knn(
        indices, distances, n_components = 3L, perplexity = 5,
        Y_init = init, early_exaggeration_iter = 0L, n_iter = 1L,
        n.cores = 2L, seed = 420L
    )
    expect_equal(dim(fit), c(n, 3L))
    expect_true(all(is.finite(fit)))
    expect_identical(
        attr(fit, "fastEmbedR_config")$repulsion, "fft_grid_3d"
    )
    expect_equal(attr(fit, "fastEmbedR_config")$fft_grid_size, 16L)
    expect_gt(attr(fit, "fastEmbedR_config")$fft_elapsed_sec, 0)

    small <- tsne_knn(
        indices[1:100, , drop = FALSE] %% 100L + 1L,
        distances[1:100, , drop = FALSE],
        n_components = 3L, perplexity = 5,
        Y_init = init[1:100, , drop = FALSE],
        early_exaggeration_iter = 0L, n_iter = 1L
    )
    expect_identical(
        attr(small, "fastEmbedR_config")$repulsion, "fft_grid_3d"
    )
    expect_error(
        tsne_knn(indices, distances, perplexity = 5,
            negative_gradient_method = "unsupported_method"),
        "must be one of"
    )
})

test_that("3D FFT reaches comparable KL on a fixed trajectory", {
    set.seed(421)
    n <- 1000L
    groups <- rep(seq_len(4L), each = n / 4L)
    x <- matrix(rnorm(n * 6L), n, 6L)
    x[, 1L] <- x[, 1L] + 2 * groups
    knn <- test_exact_knn(x, k = 15L, backend = "cpu")
    init <- matrix(rnorm(n * 3L, sd = 10), n, 3L)
    fit <- function(method) {
        tsne_knn(
            knn, perplexity = 10, n_components = 3L,
            Y_init = init, negative_gradient_method = method,
            early_exaggeration_iter = 25L, n_iter = 100L,
            auto_config = FALSE, record_costs = TRUE,
            n.cores = 1L, seed = 421L
        )
    }
    exact <- fit("exact")
    fft <- fit("fft")
    exact_kl <- tail(attr(exact, "itercosts"), 1L)
    fft_kl <- tail(attr(fft, "itercosts"), 1L)
    expect_true(is.finite(exact_kl) && is.finite(fft_kl))
    expect_lte(fft_kl, 1.05 * exact_kl)
    expect_identical(
        attr(fft, "fastEmbedR_config")$repulsion, "fft_grid_3d"
    )
})

test_that("3D Metal FFT follows the CPU one-step trajectory", {
    skip_if_not(embedding_metal_available_cpp())
    set.seed(422)
    n <- 80L
    x <- matrix(rnorm(n * 5L), n, 5L)
    knn <- test_exact_knn(x, k = 10L, backend = "cpu")
    init <- matrix(rnorm(n * 3L, sd = 2), n, 3L)
    run <- function(backend) {
        tsne_knn(
            knn, backend = backend, n_components = 3L,
            perplexity = 5, Y_init = init,
            early_exaggeration_iter = 0L, n_iter = 1L,
            learning_rate = 1, auto_config = FALSE,
            negative_gradient_method = "fft", seed = 422L
        )
    }
    cpu <- run("cpu")
    metal <- run("metal")
    expect_identical(dim(metal), c(n, 3L))
    expect_true(all(is.finite(metal)))
    expect_identical(
        attr(metal, "fastEmbedR_config")$repulsion,
        "fft_grid_3d_metal"
    )
    centered_init <- sweep(init, 2L, colMeans(init))
    relative <- sqrt(
        sum((metal - cpu)^2) /
            sum((cpu - centered_init)^2)
    )
    expect_lt(relative, 0.05)
    expect_error(
        tsne_knn(
            knn, backend = "metal", n_components = 3L,
            perplexity = 5, Y_init = init,
            negative_gradient_method = "exact",
            early_exaggeration_iter = 0L, n_iter = 1L
        ),
        "requires FFT repulsion"
    )
})

test_that("3D Metal FFT reaches comparable neighborhood quality", {
    skip_if_not(embedding_metal_available_cpp())
    set.seed(423)
    n <- 400L
    groups <- rep(seq_len(4L), each = n / 4L)
    x <- matrix(rnorm(n * 8L), n, 8L)
    x[, 1L] <- x[, 1L] + 3 * groups
    knn <- test_exact_knn(x, k = 15L, backend = "cpu")
    init <- matrix(rnorm(n * 3L, sd = 1), n, 3L)
    fit <- function(backend) {
        tsne_knn(
            knn, backend = backend, n_components = 3L,
            perplexity = 10, Y_init = init,
            early_exaggeration_iter = 25L, n_iter = 100L,
            auto_config = FALSE,
            negative_gradient_method = "fft", seed = 423L
        )
    }
    cpu <- fit("cpu")
    metal <- fit("metal")
    cpu_quality <- evaluate_embedding(x, cpu, k = 15L)
    metal_quality <- evaluate_embedding(x, metal, k = 15L)
    expect_gt(metal_quality$trustworthiness, 0.9)
    expect_lt(
        abs(cpu_quality$trustworthiness -
            metal_quality$trustworthiness), 0.02
    )
    normalized <- fastEmbedR:::normalize_opentsne_knn_input(
        knn, NULL, 10L
    )
    kl <- function(y) fastEmbedR:::opentsne_kl_diagnostic_cpp(
        normalized$indices, normalized$distances, y, 10, 1L
    )
    expect_true(is.finite(kl(cpu)) && is.finite(kl(metal)))
    expect_lte(kl(metal), 1.05 * kl(cpu))
    expect_identical(
        attr(metal, "fastEmbedR_config")$backend,
        "metal"
    )
})

test_that("3D CUDA FFT follows the CPU one-step trajectory", {
    skip_if_not(embedding_cuda_available_cpp())
    set.seed(424)
    n <- 80L
    x <- matrix(rnorm(n * 5L), n, 5L)
    knn <- test_exact_knn(x, k = 10L, backend = "cpu")
    init <- matrix(rnorm(n * 3L, sd = 2), n, 3L)
    run <- function(backend) {
        tsne_knn(
            knn, backend = backend, n_components = 3L,
            perplexity = 5, Y_init = init,
            early_exaggeration_iter = 0L, n_iter = 1L,
            learning_rate = 1, auto_config = FALSE,
            negative_gradient_method = "fft", seed = 424L
        )
    }
    cpu <- run("cpu")
    cuda <- run("cuda")
    expect_identical(dim(cuda), c(n, 3L))
    expect_true(all(is.finite(cuda)))
    expect_identical(
        attr(cuda, "fastEmbedR_config")$repulsion,
        "fft_grid_3d_cuda_cufft"
    )
    centered_init <- sweep(init, 2L, colMeans(init))
    relative <- sqrt(
        sum((cuda - cpu)^2) /
            sum((cpu - centered_init)^2)
    )
    expect_lt(relative, 0.05)
    expect_error(
        tsne_knn(
            knn, backend = "cuda", n_components = 3L,
            perplexity = 5, Y_init = init,
            negative_gradient_method = "exact",
            early_exaggeration_iter = 0L, n_iter = 1L
        ),
        "supports.*fft"
    )
})

test_that("3D CUDA FFT preserves neighborhood quality", {
    skip_if_not(embedding_cuda_available_cpp())
    set.seed(427)
    n <- 400L
    groups <- rep(seq_len(4L), each = n / 4L)
    x <- matrix(rnorm(n * 8L), n, 8L)
    x[, 1L] <- x[, 1L] + 3 * groups
    knn <- test_exact_knn(x, k = 15L, backend = "cpu")
    init <- matrix(rnorm(n * 3L, sd = 1), n, 3L)
    fit <- function(backend) {
        tsne_knn(
            knn, backend = backend, n_components = 3L,
            perplexity = 10, Y_init = init,
            early_exaggeration_iter = 25L, n_iter = 100L,
            auto_config = FALSE,
            negative_gradient_method = "fft", seed = 427L
        )
    }
    cpu <- fit("cpu")
    cuda <- fit("cuda")
    cpu_quality <- evaluate_embedding(x, cpu, k = 15L)
    cuda_quality <- evaluate_embedding(x, cuda, k = 15L)
    expect_gt(cuda_quality$trustworthiness, 0.9)
    expect_lt(
        abs(cpu_quality$trustworthiness -
            cuda_quality$trustworthiness), 0.02
    )
    normalized <- fastEmbedR:::normalize_opentsne_knn_input(
        knn, NULL, 10L
    )
    kl <- function(y) fastEmbedR:::opentsne_kl_diagnostic_cpp(
        normalized$indices, normalized$distances, y, 10, 1L
    )
    expect_true(is.finite(kl(cpu)) && is.finite(kl(cuda)))
    expect_lte(kl(cuda), 1.05 * kl(cpu))
    expect_identical(
        attr(cuda, "fastEmbedR_config")$backend,
        "cuda"
    )
})

test_that("3D CUDA uses the large FFT grid above 5000 points", {
    skip_if_not(embedding_cuda_available_cpp())
    n <- 5200L
    k <- 8L
    indices <- outer(
        seq_len(n), seq_len(k),
        function(i, j) (i + j - 1L) %% n + 1L
    )
    distances <- matrix(rep(seq_len(k) / 10, each = n), n, k)
    set.seed(77)
    init <- matrix(rnorm(n * 3L), n, 3L)
    fit <- tsne_knn(
        indices, distances, backend = "cuda", n_components = 3L,
        perplexity = 5, Y_init = init, early_exaggeration_iter = 0L,
        n_iter = 2L, auto_config = FALSE,
        negative_gradient_method = "fft", seed = 77L
    )
    config <- attr(fit, "fastEmbedR_config")
    expect_true(all(is.finite(fit)))
    expect_gt(sd(fit[, 3L]), 0)
    expect_identical(config$repulsion, "fft_grid_3d_cuda_cufft")
    expect_identical(config$fft_grid_size, 64L)
})

test_that("3D Metal one-call fit retains native PCA and FFT", {
    skip_if_not(embedding_metal_available_cpp())
    set.seed(425)
    x <- matrix(rnorm(120L * 8L), 120L, 8L)
    fit <- tsne(
        x, backend = "metal", n_components = 3L,
        perplexity = 5, early_exaggeration_iter = 3L,
        n_iter = 5L, auto_config = FALSE, seed = 425L
    )
    expect_identical(dim(fit$layout), c(120L, 3L))
    expect_true(all(is.finite(fit$layout)))
    expect_identical(fit$parameters$nn_backend, "metal")
    expect_identical(fit$parameters$init_backend, "metal_mps_rsvd")
    expect_identical(fit$parameters$repulsion, "fft_grid_3d_metal")
})

test_that("3D CUDA one-call fit retains native PCA and FFT", {
    skip_if_not(embedding_cuda_available_cpp())
    set.seed(426)
    x <- matrix(rnorm(120L * 8L), 120L, 8L)
    fit <- tsne(
        x, backend = "cuda", n_components = 3L,
        perplexity = 5, early_exaggeration_iter = 3L,
        n_iter = 5L, auto_config = FALSE, seed = 426L
    )
    expect_identical(dim(fit$layout), c(120L, 3L))
    expect_true(all(is.finite(fit$layout)))
    expect_match(fit$parameters$nn_backend, "^native_cuda_")
    expect_match(fit$parameters$init_backend, "^cuda_")
    expect_identical(fit$parameters$repulsion, "fft_grid_3d_cuda_cufft")
})
