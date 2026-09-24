#!/usr/bin/env Rscript

max_function_lines <- 50L
max_line_width <- 80L

source_files <- function() {
    paths <- c(
        list.files("R", "[.][Rr]$", full.names = TRUE),
        list.files("tests/testthat", "[.][Rr]$", full.names = TRUE),
        list.files("vignettes", "[.](Rmd|Rmarkdown)$", full.names = TRUE)
    )
    paths <- paths[file.exists(paths)]
    paths[basename(paths) != "RcppExports.R"]
}

line_width_failures <- function(paths) {
    failures <- character()
    for (path in paths) {
        lines <- readLines(path, warn = FALSE)
        bad <- which(nchar(lines, type = "width") > max_line_width)
        if (length(bad)) {
            failures <- c(
                failures,
                sprintf(
                    "%s:%d: %d columns",
                    path,
                    bad,
                    nchar(lines[bad], type = "width")
                )
            )
        }
    }
    failures
}

function_lengths <- function(path) {
    parsed <- parse(path, keep.source = TRUE)
    data <- getParseData(parsed)
    data <- data[data$terminal & data$parent > -1L, ]
    function_rows <- which(data$token %in% c("FUNCTION", "'\\\\'"))
    if (!length(function_rows)) {
        return(data.frame())
    }

    rows <- lapply(function_rows, function(index) {
        parent <- data$parent[index]
        start <- data$line1[index]
        name <- "_anonymous_"
        if (index >= 3L &&
            data$token[index - 1L] %in% c("EQ_ASSIGN", "LEFT_ASSIGN") &&
            data$token[index - 2L] == "SYMBOL") {
            name <- data$text[index - 2L]
            start <- data$line1[index - 2L]
        }
        tail_parent <- data$parent[seq.int(index + 1L, nrow(data))]
        exit <- which(tail_parent > parent)
        end_index <- if (length(exit)) index + exit[1L] - 1L else nrow(data)
        end <- data$line2[end_index]
        data.frame(
            file = path,
            name = name,
            start = start,
            end = end,
            lines = end - start + 1L
        )
    })
    do.call(rbind, rows)
}

function_length_failures <- function(paths) {
    paths <- paths[
        grepl("^R/.*[.][Rr]$", paths)
    ]
    lengths <- do.call(rbind, lapply(paths, function_lengths))
    if (is.null(lengths) || !nrow(lengths)) {
        return(character())
    }
    bad <- lengths[lengths$lines > max_function_lines, , drop = FALSE]
    sprintf(
        "%s:%d: %s() spans %d lines",
        bad$file,
        bad$start,
        bad$name,
        bad$lines
    )
}

tab_failures <- function(paths) {
    failures <- character()
    for (path in paths) {
        lines <- readLines(path, warn = FALSE)
        bad <- which(grepl("\\t", lines))
        if (length(bad)) {
            failures <- c(failures, sprintf("%s:%d: tab", path, bad))
        }
    }
    failures
}

indentation_failures <- function(paths) {
    failures <- character()
    paths <- paths[grepl("[.](R|r|Rd)$", paths)]
    for (path in paths) {
        lines <- readLines(path, warn = FALSE)
        indentation <- nchar(sub("^( *)[^ ].*$", "\\1", lines))
        bad <- which(indentation > 0L & indentation %% 4L != 0L)
        if (length(bad)) {
            failures <- c(
                failures,
                sprintf("%s:%d: indentation is not a multiple of 4", path, bad)
            )
        }
    }
    failures
}

paths <- source_files()
failures <- c(
    line_width_failures(paths),
    function_length_failures(paths),
    tab_failures(paths),
    indentation_failures(paths)
)

if (length(failures)) {
    cat(paste(failures, collapse = "\n"), "\n", sep = "")
    quit(status = 1L)
}

cat(
    "Source style passed: functions <= 50 lines, lines <= 80 columns, ",
    "four-space indentation, and no tabs.\n",
    sep = ""
)
