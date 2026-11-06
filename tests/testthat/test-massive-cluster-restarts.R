test_that("automatic resident clustering uses requested CPU restarts", {
    set.seed(904)
    x <- rbind(matrix(rnorm(48L, 0, 0.1), ncol = 2L),
        matrix(rnorm(48L, 4, 0.1), ncol = 2L),
        matrix(rnorm(48L, 8, 0.1), ncol = 2L))
    for (method in c("leiden", "louvain")) {
        serial <- massive_cluster(x, massive = "auto", method = method,
            k = 4L, memory_limit = "1GB", n_runs = 4L,
            n.cores = 1L, seed = 19L)
        parallel <- massive_cluster(x, massive = "auto", method = method,
            k = 4L, memory_limit = "1GB", n_runs = 4L,
            n.cores = 2L, seed = 19L)
        expect_identical(parallel$membership, serial$membership)
        expect_identical(parallel$modularity, serial$modularity)
        expect_identical(parallel$effective_cpu_workers, 2L)
        expect_identical(parallel$parameters$n.cores, 2L)
        single <- massive_cluster(x, massive = "auto", method = method,
            k = 4L, memory_limit = "1GB", n_runs = 1L,
            n.cores = 2L, seed = 19L)
        expect_identical(single$effective_cpu_workers, 1L)
    }
})

test_that("file-backed landmark fit uses and reports CPU restarts", {
    set.seed(905)
    x <- rbind(matrix(rnorm(48L, 0, 0.1), ncol = 2L),
        matrix(rnorm(48L, 4, 0.1), ncol = 2L),
        matrix(rnorm(48L, 8, 0.1), ncol = 2L))
    source_path <- tempfile(fileext = ".f32")
    directory <- tempfile()
    dir.create(directory)
    on.exit(unlink(c(source_path, directory), recursive = TRUE))
    writeBin(as.numeric(t(x)), source_path, size = 4L,
        endian = "little")
    source <- massive_matrix(source_path, nrow(x), ncol(x))
    for (method in c("leiden", "louvain")) {
        serial <- massive_cluster(source, massive = "landmark",
            method = method, landmarks = 24L, k = 4L,
            output = file.path(directory, paste0(method, "-serial")),
            memory_limit = "512MB", n_runs = 4L,
            n.cores = 1L, seed = 19L)
        parallel <- massive_cluster(source, massive = "landmark",
            method = method, landmarks = 24L, k = 4L,
            output = file.path(directory, paste0(method, "-parallel")),
            memory_limit = "512MB", n_runs = 4L,
            n.cores = 2L, seed = 19L)
        expect_identical(parallel$reference_membership,
            serial$reference_membership)
        expect_identical(parallel$reference_modularity,
            serial$reference_modularity)
        expect_identical(parallel$fitting_cpu_workers, 2L)
        expect_identical(parallel$resources$fitting_cpu_workers_budgeted,
            2L)
        observed <- massive_read_cluster_rows(parallel, 1L, nrow(x))
        expected <- massive_read_cluster_rows(serial, 1L, nrow(x))
        expect_identical(observed$membership, expected$membership)
        single <- massive_cluster(source, massive = "landmark",
            method = method, landmarks = 24L, k = 4L,
            output = file.path(directory, paste0(method, "-single")),
            memory_limit = "512MB", n_runs = 1L,
            n.cores = 2L, seed = 19L)
        expect_identical(single$fitting_cpu_workers, 1L)
    }
})
