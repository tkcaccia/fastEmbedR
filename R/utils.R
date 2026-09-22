cache_file <- function(cache_dir, prefix, dataset, n, p, key) {
    safe_dataset <- gsub("[^A-Za-z0-9_.-]+", "_", dataset)
    filename <- sprintf(
        "%s_%s_n%s_p%s_%s.rds",
        prefix,
        safe_dataset,
        n,
        p,
        key
    )
    file.path(cache_dir, filename)
}

capture_error <- function(expr) {
    tryCatch(
        list(value = force(expr), error = NA_character_),
        error = function(e) {
            list(value = NULL, error = conditionMessage(e))
        }
    )
}

numeric_scalar <- function(x, default = NA_real_) {
    if (length(x) != 1L || is.na(x)) {
        return(default)
    }
    if (is.factor(x)) {
        x <- as.character(x)
    }
    if (is.numeric(x) || is.logical(x)) {
        return(as.numeric(x))
    }
    if (!is.character(x)) {
        return(default)
    }
    value <- trimws(x)
    numeric_pattern <- paste0(
        "^[+-]?(?:(?:[0-9]+(?:\\.[0-9]*)?)|(?:\\.[0-9]+))",
        "(?:[eE][+-]?[0-9]+)?$"
    )
    if (!grepl(numeric_pattern, value, perl = TRUE)) {
        return(default)
    }
    as.numeric(value)
}

integer_scalar <- function(x, default = NA_integer_) {
    value <- numeric_scalar(x)
    if (!is.finite(value) ||
        value < -.Machine$integer.max - 1 ||
        value > .Machine$integer.max) {
        return(default)
    }
    rounded <- round(value)
    tolerance <- sqrt(.Machine$double.eps) * max(1, abs(value))
    if (abs(value - rounded) > tolerance) {
        return(default)
    }
    as.integer(rounded)
}

set_local_seed <- function(seed) {
    had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    old_seed <- if (had_seed) {
        get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    } else {
        NULL
    }
    seeded_state <- withr::with_seed(
        as.integer(seed),
        get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    )
    assign(".Random.seed", seeded_state, envir = .GlobalEnv)
    function() {
        if (had_seed) {
            assign(".Random.seed", old_seed, envir = .GlobalEnv)
        } else if (exists(".Random.seed",
            envir = .GlobalEnv,
            inherits = FALSE
        )) {
            rm(".Random.seed", envir = .GlobalEnv)
        }
        invisible(NULL)
    }
}
# Evaluate a function call while retaining both its value and elapsed timing.
timed_do_call <- function(fun, args) {
    value <- NULL
    elapsed <- system.time({
        value <- do.call(fun, args)
    })
    list(value = value, time = elapsed)
}
