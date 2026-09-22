#!/usr/bin/env Rscript

# Release-level numerical validation for the native t-SNE kernels.
# Run from the fastEmbedR source root after installing the source being tested.

arguments <- commandArgs(trailingOnly = TRUE)
argument_value <- function(name, default) {
  prefix <- paste0("--", name, "=")
  match <- arguments[startsWith(arguments, prefix)]
  if (!length(match)) return(default)
  sub(prefix, "", match[[length(match)]], fixed = TRUE)
}

source_root <- normalizePath(
  argument_value("source-root", Sys.getenv("FASTEMBEDR_SOURCE_ROOT", ".")),
  mustWork = TRUE
)
output_dir <- argument_value(
  "out-dir",
  file.path(source_root, "results", "tsne_numerical_validation")
)
requested_backends <- unique(strsplit(
  argument_value("backends", "cpu,metal,cuda"), ",", fixed = TRUE
)[[1]])
requested_backends <- trimws(requested_backends)
requested_backends <- requested_backends[
  requested_backends %in% c("cpu", "metal", "cuda")
]
if (!length(requested_backends)) stop("No valid backend was requested.")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
source(file.path(source_root, "tests", "testthat", "helper-knn.R"))
source(file.path(source_root, "tests", "testthat", "helper-tsne-reference.R"))
suppressPackageStartupMessages(library(fastEmbedR))

write_result <- function(value, filename) {
  utils::write.csv(value, file.path(output_dir, filename), row.names = FALSE)
}
bind_rows <- function(rows) {
  if (!length(rows)) return(data.frame())
  columns <- unique(unlist(lapply(rows, names), use.names = FALSE))
  normalized <- lapply(rows, function(row) {
    missing <- setdiff(columns, names(row))
    for (name in missing) row[[name]] <- NA
    as.data.frame(row[columns], stringsAsFactors = FALSE)
  })
  do.call(rbind, normalized)
}
backend_available <- function(backend) {
  switch(
    backend,
    cpu = TRUE,
    metal = isTRUE(fastEmbedR:::embedding_metal_available_cpp()) &&
      isTRUE(fastEmbedR:::metal_opentsne_native_available()),
    cuda = isTRUE(fastEmbedR:::embedding_cuda_available_cpp()) &&
      isTRUE(fastEmbedR:::cuda_opentsne_native_available()),
    FALSE
  )
}
backend_method_supported <- function(backend, method) {
  !(identical(backend, "cuda") && identical(method, "exact"))
}

thresholds <- data.frame(
  check = c(
    "affinity_float32_vs_float64_max_abs",
    "attractive_force_float32_vs_float64_relative_l2",
    "repulsive_force_float32_vs_float64_relative_l2",
    "objective_gradient_float32_vs_float64_relative_l2",
    "kl_float32_vs_float64_absolute",
    "finite_difference_float64_relative_l2",
    "finite_difference_float32_relative_l2",
    "fft_repulsive_grid32_relative_l2",
    "fft_repulsive_grid64_relative_l2",
    "fft_repulsive_grid128_relative_l2",
    "fft_force_correlation_minimum",
    "backend_exact_first_step_relative_l2",
    "backend_fft_first_step_relative_l2",
    "backend_exact_trajectory_relative_kl",
    "backend_fft_trajectory_relative_kl",
    "pathological_valid_input_finite",
    "public_backend_pathological_inputs"
  ),
  direction = c(
    rep("maximum", 10L), "minimum", rep("maximum", 4L),
    "required", "required"
  ),
  threshold = c(
    2e-6, 5e-5, 5e-5, 1e-4, 5e-6, 2e-7, 2e-4,
    1e-2, 3e-3, 1e-3, 0.999, 5e-4, 2e-3, 2e-2, 5e-2, 1, 1
  ),
  rationale = c(
    "float32 affinity mass against independent dense float64 oracle",
    "float32 sparse attractive force against float64 oracle",
    "float32 exact repulsion against float64 all-pairs oracle",
    "complete exact objective gradient against float64 oracle",
    "common-affinity exact KL against float64 oracle",
    "central finite difference against analytic float64 gradient",
    "production float32 gradient against central finite difference",
    "coarse FFT grid approximation against exact repulsion",
    "medium FFT grid approximation against exact repulsion",
    "production-scale diagnostic grid against exact repulsion",
    "FFT and exact force direction",
    "identical-state CPU/GPU exact update",
    "identical-state CPU/GPU FFT update",
    "common-affinity KL trajectory relative to CPU exact",
    "common-affinity KL trajectory relative to CPU FFT",
    "all returned gradients, normalizers, and KL must be finite",
    "each available public backend must return finite layouts or reject Inf"
  ),
  stringsAsFactors = FALSE
)
write_result(thresholds, "acceptance_thresholds.csv")

# Exact float32 implementation against an independent dense float64 oracle.
set.seed(1701)
x <- matrix(rnorm(48L * 6L), 48L, 6L)
knn <- test_exact_knn(x, k = 15L, exclude_self = TRUE)
layout <- matrix(rnorm(48L * 2L, sd = 0.3), 48L, 2L)
probability <- tsne_reference_probabilities(knn$indices, knn$distances, 5)
reference <- tsne_reference_forces(probability, layout)
diagnostic <- fastEmbedR:::opentsne_force_diagnostic_cpp(
  knn$indices, knn$distances, layout, 5, 1, 64L, 2L
)
observed_probability <- tsne_csr_probability_matrix(diagnostic)
force_reference <- data.frame(
  metric = c(
    "affinity_max_abs", "attractive_relative_l2", "repulsive_relative_l2",
    "objective_gradient_relative_l2", "sum_q_absolute", "kl_absolute"
  ),
  value = c(
    max(abs(observed_probability - probability)),
    tsne_relative_l2(diagnostic$attractive_force, reference$attractive_force),
    tsne_relative_l2(
      diagnostic$repulsive_force_exact, reference$repulsive_force
    ),
    tsne_relative_l2(
      diagnostic$objective_gradient_exact, reference$objective_gradient
    ),
    abs(diagnostic$sum_q - reference$sum_q),
    abs(diagnostic$kl - reference$kl)
  ),
  precision_tested = "float32 production vs float64 independent reference",
  stringsAsFactors = FALSE
)
write_result(force_reference, "exact_force_float32_vs_float64.csv")

# Central finite-difference check of the objective gradient.
set.seed(1702)
x_fd <- matrix(rnorm(18L * 5L), 18L, 5L)
knn_fd <- test_exact_knn(x_fd, k = 9L, exclude_self = TRUE)
layout_fd <- matrix(rnorm(18L * 2L, sd = 0.2), 18L, 2L)
probability_fd <- tsne_reference_probabilities(
  knn_fd$indices, knn_fd$distances, 3
)
reference_fd <- tsne_reference_forces(probability_fd, layout_fd)
diagnostic_fd <- fastEmbedR:::opentsne_force_diagnostic_cpp(
  knn_fd$indices, knn_fd$distances, layout_fd, 3, 1, 64L, 1L
)
epsilon <- 1e-6
numerical <- matrix(0, nrow(layout_fd), ncol(layout_fd))
for (index in seq_along(layout_fd)) {
  upper <- lower <- layout_fd
  upper[index] <- upper[index] + epsilon
  lower[index] <- lower[index] - epsilon
  numerical[index] <- (
    tsne_reference_forces(probability_fd, upper)$kl -
      tsne_reference_forces(probability_fd, lower)$kl
  ) / (2 * epsilon)
}
finite_difference <- data.frame(
  implementation = c("float64_reference", "fastEmbedR_float32_exact"),
  relative_l2 = c(
    tsne_relative_l2(reference_fd$objective_gradient, numerical),
    tsne_relative_l2(diagnostic_fd$objective_gradient_exact, numerical)
  ),
  epsilon = epsilon,
  stringsAsFactors = FALSE
)
write_result(finite_difference, "finite_difference_gradient.csv")

# FFT-grid approximation against exact repulsion.
grid_rows <- list()
for (grid in c(32L, 64L, 128L, 256L)) {
  observed <- fastEmbedR:::opentsne_force_diagnostic_cpp(
    knn$indices, knn$distances, layout, 5, 1, grid, 2L
  )
  grid_rows[[length(grid_rows) + 1L]] <- data.frame(
    grid_size = grid,
    repulsive_relative_l2 = tsne_relative_l2(
      observed$repulsive_force_fft, reference$repulsive_force
    ),
    force_correlation = stats::cor(
      c(observed$repulsive_force_fft), c(reference$repulsive_force)
    ),
    stringsAsFactors = FALSE
  )
}
fft_grid <- bind_rows(grid_rows)
write_result(fft_grid, "fft_grid_force_convergence.csv")

# Exact gradients across perplexities and candidate-support widths.
set.seed(1703)
x_support <- matrix(rnorm(40L * 6L), 40L, 6L)
layout_support <- matrix(rnorm(40L * 2L, sd = 0.25), 40L, 2L)
support_rows <- list()
for (perplexity in c(2L, 4L, 8L)) {
  for (multiplier in c(1L, 3L, 4L)) {
    width <- perplexity * multiplier
    knn_support <- test_exact_knn(
      x_support, k = width, exclude_self = TRUE
    )
    probability_support <- tsne_reference_probabilities(
      knn_support$indices, knn_support$distances, perplexity
    )
    reference_support <- tsne_reference_forces(
      probability_support, layout_support
    )
    observed_support <- fastEmbedR:::opentsne_force_diagnostic_cpp(
      knn_support$indices, knn_support$distances, layout_support,
      perplexity, 1, 64L, 2L
    )
    support_rows[[length(support_rows) + 1L]] <- data.frame(
      perplexity = perplexity,
      support_multiplier = multiplier,
      support_width = width,
      objective_gradient_relative_l2 = tsne_relative_l2(
        observed_support$objective_gradient_exact,
        reference_support$objective_gradient
      ),
      kl_absolute = abs(observed_support$kl - reference_support$kl),
      stringsAsFactors = FALSE
    )
  }
}
write_result(bind_rows(support_rows), "perplexity_support_sweep.csv")

# Valid pathological cases and one invalid nonfinite-coordinate case.
set.seed(1704)
x_path <- rbind(
  matrix(0, 8L, 4L),
  matrix(1, 8L, 4L),
  matrix(rnorm(8L * 4L, sd = 1e-8), 8L, 4L)
)
knn_path <- test_exact_knn(x_path, k = 6L, exclude_self = TRUE)
disconnected_x <- rbind(
  matrix(rnorm(12L * 4L, -20, 0.01), 12L, 4L),
  matrix(rnorm(12L * 4L, 20, 0.01), 12L, 4L)
)
knn_disconnected <- test_exact_knn(
  disconnected_x, k = 6L, exclude_self = TRUE
)
pathological_cases <- list(
  duplicated_and_zero_distances = list(
    knn = knn_path,
    layout = matrix(rnorm(24L * 2L, sd = 0.1), 24L, 2L)
  ),
  coincident_coordinates = list(knn = knn_path, layout = matrix(0, 24L, 2L)),
  extreme_coordinates = list(
    knn = knn_path,
    layout = matrix(rep(c(-1e4, 1e4), 24L), 24L, 2L, byrow = TRUE)
  ),
  disconnected_knn = list(
    knn = knn_disconnected,
    layout = matrix(rnorm(24L * 2L, sd = 0.1), 24L, 2L)
  )
)
pathology_rows <- list()
for (case_name in names(pathological_cases)) {
  case <- pathological_cases[[case_name]]
  result <- tryCatch(
    fastEmbedR:::opentsne_force_diagnostic_cpp(
      case$knn$indices, case$knn$distances, case$layout,
      2, 1, 64L, 2L
    ),
    error = identity
  )
  finite <- !inherits(result, "error") &&
    all(is.finite(result$objective_gradient_exact)) &&
    all(is.finite(result$objective_gradient_fft)) &&
    is.finite(result$sum_q) && result$sum_q > 0 && is.finite(result$kl)
  pathology_rows[[length(pathology_rows) + 1L]] <- data.frame(
    case = case_name,
    expected = "finite_success",
    passed = finite,
    error = if (inherits(result, "error")) conditionMessage(result) else NA,
    stringsAsFactors = FALSE
  )
}
invalid_layout <- pathological_cases[[1L]]$layout
invalid_layout[1, 1] <- Inf
invalid_result <- tryCatch(
  fastEmbedR:::opentsne_force_diagnostic_cpp(
    knn_path$indices, knn_path$distances, invalid_layout,
    2, 1, 64L, 2L
  ),
  error = identity
)
pathology_rows[[length(pathology_rows) + 1L]] <- data.frame(
  case = "nonfinite_coordinates",
  expected = "explicit_error",
  passed = inherits(invalid_result, "error"),
  error = if (inherits(invalid_result, "error")) {
    conditionMessage(invalid_result)
  } else {
    NA
  },
  stringsAsFactors = FALSE
)
pathology_results <- bind_rows(pathology_rows)
write_result(pathology_results, "pathological_inputs.csv")

# Identical-state one-step and objective-trajectory backend comparisons.
old_grid <- Sys.getenv("FASTEMBEDR_TSNE_FFT_GRID", unset = NA_character_)
Sys.setenv(FASTEMBEDR_TSNE_FFT_GRID = "64")

set.seed(1705)
x_backend <- matrix(rnorm(64L * 6L), 64L, 6L)
knn_backend <- test_exact_knn(x_backend, k = 15L, exclude_self = TRUE)
initial_backend <- matrix(rnorm(64L * 2L, sd = 1e-4), 64L, 2L)
run_prefix <- function(backend, method, iterations, learning_rate = 1) {
  knn_input <- knn_backend
  if (identical(backend, "cuda")) {
    if (!requireNamespace("float", quietly = TRUE)) {
      stop("CUDA numerical validation requires the suggested `float` package.")
    }
    knn_input$distances <- float::fl(knn_input$distances)
  }
  result <- tsne_knn(
    knn_input,
    perplexity = 5,
    Y_init = initial_backend,
    early_exaggeration_iter = 0L,
    n_iter = as.integer(iterations),
    early_exaggeration = 1,
    exaggeration = 1,
    learning_rate = learning_rate,
    initial_momentum = 0,
    final_momentum = 0,
    max_step_norm = 1e6,
    negative_gradient_method = method,
    auto_config = FALSE,
    backend = backend,
    n.cores = 2L,
    seed = 1705L
  )
  if (inherits(result, "float32")) float::dbl(result) else as.matrix(result)
}
available <- vapply(requested_backends, backend_available, logical(1))

backend_pathology_rows <- list()
for (backend in requested_backends) {
  if (!available[[backend]]) {
    backend_pathology_rows[[length(backend_pathology_rows) + 1L]] <- data.frame(
      backend = backend, case = NA_character_, status = "unavailable",
      expected = NA_character_, actual_backend = NA_character_, passed = NA,
      error = NA_character_, stringsAsFactors = FALSE
    )
    next
  }
  for (case_name in names(pathological_cases)) {
    case <- pathological_cases[[case_name]]
    case_knn <- case$knn
    if (identical(backend, "cuda")) {
      if (!requireNamespace("float", quietly = TRUE)) {
        stop("CUDA numerical validation requires the suggested `float` package.")
      }
      case_knn$distances <- float::fl(case_knn$distances)
    }
    result <- tryCatch(
      tsne_knn(
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
        seed = 1704L
      ),
      error = identity
    )
    actual_backend <- if (inherits(result, "error")) {
      NA_character_
    } else {
      attr(result, "fastEmbedR_config")$backend
    }
    materialized <- if (inherits(result, "error")) {
      NULL
    } else if (inherits(result, "float32")) {
      float::dbl(result)
    } else {
      as.matrix(result)
    }
    finite <- !is.null(materialized) && all(is.finite(materialized))
    backend_pathology_rows[[length(backend_pathology_rows) + 1L]] <- data.frame(
      backend = backend, case = case_name, status = if (finite) "success" else "failed",
      expected = "finite_success", actual_backend = actual_backend,
      passed = finite && identical(actual_backend, backend),
      error = if (inherits(result, "error")) conditionMessage(result) else NA,
      stringsAsFactors = FALSE
    )
  }

  invalid_knn <- knn_path
  if (identical(backend, "cuda")) {
    invalid_knn$distances <- float::fl(invalid_knn$distances)
  }
  invalid_public <- tryCatch(
    tsne_knn(
      invalid_knn,
      perplexity = 2,
      Y_init = invalid_layout,
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
      seed = 1704L
    ),
    error = identity
  )
  rejected <- inherits(invalid_public, "error")
  backend_pathology_rows[[length(backend_pathology_rows) + 1L]] <- data.frame(
    backend = backend, case = "nonfinite_coordinates",
    status = if (rejected) "success" else "failed",
    expected = "explicit_error", actual_backend = NA_character_,
    passed = rejected,
    error = if (rejected) conditionMessage(invalid_public) else NA,
    stringsAsFactors = FALSE
  )
}
backend_pathology <- bind_rows(backend_pathology_rows)
write_result(backend_pathology, "backend_pathological_inputs.csv")

one_step_rows <- list()
cpu_steps <- list()
for (method in c("exact", "fft")) {
  cpu_steps[[method]] <- run_prefix("cpu", method, 1L)
  for (backend in requested_backends) {
    if (!available[[backend]]) {
      one_step_rows[[length(one_step_rows) + 1L]] <- data.frame(
        backend = backend, method = method, status = "unavailable",
        relative_l2_vs_cpu = NA_real_, max_abs_vs_cpu = NA_real_,
        threshold = if (method == "exact") 5e-4 else 2e-3,
        passed = NA, stringsAsFactors = FALSE
      )
      next
    }
    if (!backend_method_supported(backend, method)) {
      one_step_rows[[length(one_step_rows) + 1L]] <- data.frame(
        backend = backend, method = method, status = "unsupported",
        relative_l2_vs_cpu = NA_real_, max_abs_vs_cpu = NA_real_,
        threshold = if (method == "exact") 5e-4 else 2e-3,
        passed = NA, stringsAsFactors = FALSE
      )
      next
    }
    observed <- if (backend == "cpu") cpu_steps[[method]] else
      run_prefix(backend, method, 1L)
    error <- tsne_relative_l2(observed, cpu_steps[[method]])
    threshold <- if (method == "exact") 5e-4 else 2e-3
    one_step_rows[[length(one_step_rows) + 1L]] <- data.frame(
      backend = backend, method = method, status = "success",
      relative_l2_vs_cpu = error,
      max_abs_vs_cpu = max(abs(observed - cpu_steps[[method]])),
      threshold = threshold,
      passed = error <= threshold,
      stringsAsFactors = FALSE
    )
  }
}
one_step <- bind_rows(one_step_rows)
write_result(one_step, "backend_first_step.csv")

trajectory_rows <- list()
checkpoints <- c(1L, 10L, 25L, 50L, 100L)
for (method in c("exact", "fft")) {
  for (iteration in checkpoints) {
    for (backend in requested_backends) {
      if (!available[[backend]]) {
        trajectory_rows[[length(trajectory_rows) + 1L]] <- data.frame(
          backend = backend, method = method, iteration = iteration,
          status = "unavailable", kl = NA_real_, stringsAsFactors = FALSE
        )
        next
      }
      if (!backend_method_supported(backend, method)) {
        trajectory_rows[[length(trajectory_rows) + 1L]] <- data.frame(
          backend = backend, method = method, iteration = iteration,
          status = "unsupported", kl = NA_real_, stringsAsFactors = FALSE
        )
        next
      }
      observed <- run_prefix(backend, method, iteration, learning_rate = 5)
      objective <- fastEmbedR:::opentsne_force_diagnostic_cpp(
        knn_backend$indices, knn_backend$distances, observed,
        5, 1, 64L, 2L
      )$kl
      trajectory_rows[[length(trajectory_rows) + 1L]] <- data.frame(
        backend = backend, method = method, iteration = iteration,
        status = "success", kl = objective, stringsAsFactors = FALSE
      )
    }
  }
}
trajectory <- bind_rows(trajectory_rows)
cpu_key <- paste(trajectory$method, trajectory$iteration)[
  trajectory$backend == "cpu" & trajectory$status == "success"
]
cpu_kl <- trajectory$kl[
  trajectory$backend == "cpu" & trajectory$status == "success"
]
reference_position <- match(
  paste(trajectory$method, trajectory$iteration), cpu_key
)
trajectory$cpu_kl <- cpu_kl[reference_position]
trajectory$relative_kl_difference_vs_cpu <- abs(
  trajectory$kl - trajectory$cpu_kl
) / pmax(abs(trajectory$cpu_kl), .Machine$double.eps)
trajectory$threshold <- ifelse(trajectory$method == "exact", 2e-2, 5e-2)
trajectory$passed <- ifelse(
  trajectory$status == "success",
  trajectory$relative_kl_difference_vs_cpu <= trajectory$threshold,
  NA
)
write_result(trajectory, "objective_trajectories.csv")

# Collate the predeclared gates into one release-readable table.
summary_rows <- list(
  data.frame(
    check = "float32_exact_force_reference",
    available = TRUE,
    passed = force_reference$value[force_reference$metric == "affinity_max_abs"] <= 2e-6 &&
      force_reference$value[force_reference$metric == "attractive_relative_l2"] <= 5e-5 &&
      force_reference$value[force_reference$metric == "repulsive_relative_l2"] <= 5e-5 &&
      force_reference$value[force_reference$metric == "objective_gradient_relative_l2"] <= 1e-4 &&
      force_reference$value[force_reference$metric == "kl_absolute"] <= 5e-6,
    stringsAsFactors = FALSE
  ),
  data.frame(
    check = "finite_difference_gradient",
    available = TRUE,
    passed = finite_difference$relative_l2[1] <= 2e-7 &&
      finite_difference$relative_l2[2] <= 2e-4,
    stringsAsFactors = FALSE
  ),
  data.frame(
    check = "fft_grid_convergence",
    available = TRUE,
    passed = fft_grid$repulsive_relative_l2[fft_grid$grid_size == 32L] <= 1e-2 &&
      fft_grid$repulsive_relative_l2[fft_grid$grid_size == 64L] <= 3e-3 &&
      fft_grid$repulsive_relative_l2[fft_grid$grid_size == 128L] <= 1e-3 &&
      min(fft_grid$force_correlation) >= 0.999 &&
      all(diff(fft_grid$repulsive_relative_l2) < 0),
    stringsAsFactors = FALSE
  ),
  data.frame(
    check = "perplexity_support_sweep",
    available = TRUE,
    passed = all(bind_rows(support_rows)$objective_gradient_relative_l2 <= 2e-4),
    stringsAsFactors = FALSE
  ),
  data.frame(
    check = "pathological_inputs",
    available = TRUE,
    passed = all(pathology_results$passed),
    stringsAsFactors = FALSE
  )
)
for (backend in requested_backends) {
  if (backend == "cpu") next
  summary_rows[[length(summary_rows) + 1L]] <- data.frame(
    check = paste0(backend, "_first_step"),
    available = available[[backend]],
    passed = if (!available[[backend]]) {
      NA
    } else {
      rows <- one_step$backend == backend & one_step$status == "success"
      any(rows) && all(one_step$passed[rows])
    },
    stringsAsFactors = FALSE
  )
  summary_rows[[length(summary_rows) + 1L]] <- data.frame(
    check = paste0(backend, "_pathological_inputs"),
    available = available[[backend]],
    passed = if (!available[[backend]]) {
      NA
    } else {
      rows <- backend_pathology$backend == backend &
        backend_pathology$status != "unavailable"
      any(rows) && all(backend_pathology$passed[rows])
    },
    stringsAsFactors = FALSE
  )
  summary_rows[[length(summary_rows) + 1L]] <- data.frame(
    check = paste0(backend, "_objective_trajectory"),
    available = available[[backend]],
    passed = if (!available[[backend]]) {
      NA
    } else {
      rows <- trajectory$backend == backend & trajectory$status == "success"
      any(rows) && all(trajectory$passed[rows])
    },
    stringsAsFactors = FALSE
  )
}
validation_summary <- bind_rows(summary_rows)
write_result(validation_summary, "validation_summary.csv")

has_git_metadata <- file.exists(file.path(source_root, ".git"))
git_commit <- if (has_git_metadata) {
  tryCatch(
    system2("git", c("-C", source_root, "rev-parse", "HEAD"), stdout = TRUE),
    error = function(error) "unavailable"
  )
} else {
  "unavailable"
}
git_status <- if (has_git_metadata) {
  tryCatch(
    system2(
      "git", c("-C", source_root, "status", "--porcelain=v1"), stdout = TRUE
    ),
    error = function(error) "unavailable"
  )
} else {
  "unavailable"
}
manifest <- c(
  paste0("timestamp_utc=", format(Sys.time(), tz = "UTC", usetz = TRUE)),
  paste0("source_root=", source_root),
  paste0("package_version=", as.character(utils::packageVersion("fastEmbedR"))),
  paste0("installed_package=", find.package("fastEmbedR")),
  paste0("git_commit=", paste(git_commit, collapse = "")),
  paste0("git_dirty=", length(git_status) > 0L),
  paste0("git_dirty_file_count=", if (identical(git_status, "unavailable")) {
    NA_integer_
  } else {
    length(git_status)
  }),
  paste0("requested_backends=", paste(requested_backends, collapse = ",")),
  paste0("available_backends=", paste(
    requested_backends[available], collapse = ","
  )),
  paste0("R_version=", R.version.string),
  paste0("platform=", R.version$platform),
  paste0("machine=", Sys.info()[["machine"]]),
  "trajectory_protocol=independent deterministic prefixes from identical optimizer state",
  "cuda_unavailable_is_reported_not_passed"
)
writeLines(manifest, file.path(output_dir, "validation_manifest.txt"))

if (is.na(old_grid)) {
  Sys.unsetenv("FASTEMBEDR_TSNE_FFT_GRID")
} else {
  Sys.setenv(FASTEMBEDR_TSNE_FFT_GRID = old_grid)
}

print(validation_summary, row.names = FALSE)
if (any(validation_summary$available & !validation_summary$passed)) {
  quit(status = 1L)
}
