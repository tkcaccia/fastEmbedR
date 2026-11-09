wide_cluster_edges <- function() {
    list(from = as.integer(c(rep(1L, 70L), 2:70)),
        to = as.integer(c(2:71, 3:71)),
        weight = rep(1, 139L))
}

wide_cluster_native <- function(backend, method, graph, force_wide) {
    native <- switch(backend,
        metal = fastEmbedR:::fastembedr_graph_cluster_metal_cpp,
        cuda = fastEmbedR:::fastembedr_graph_cluster_cuda_cpp)
    native(graph$from, graph$to, graph$weight, 71L,
        method, 1, 6L, 1L, 3, force_wide)
}

test_that("GPU clustering makes one decision across wide edge tiles", {
    available <- c(
        metal = isTRUE(fastEmbedR:::graph_clustering_metal_available_cpp()),
        cuda = isTRUE(fastEmbedR:::graph_clustering_cuda_available_cpp()))
    skip_if(!any(available))
    graph <- wide_cluster_edges()
    for (backend in names(available)[available]) {
        for (method in c("louvain", "leiden")) {
            narrow <- wide_cluster_native(backend, method, graph, FALSE)
            wide <- wide_cluster_native(backend, method, graph, TRUE)
            expect_identical(narrow$csr_offset_dtype, "int32")
            expect_identical(wide$csr_offset_dtype, "uint64")
            expect_identical(wide$graph_storage,
                "float32_csr_uint64")
            if (backend == "metal") {
                expect_identical(wide$metal_graph_input,
                    "bounded_resident_csr_tiles")
            }
            expect_equal(wide$modularity, narrow$modularity,
                tolerance = 1e-5)
            expect_identical(
                outer(wide$membership, wide$membership, "=="),
                outer(narrow$membership, narrow$membership, "=="))
        }
    }
})

test_that("public clustering summary reports the selected CSR width", {
    skip_if(!isTRUE(
        fastEmbedR:::graph_clustering_metal_available_cpp()))
    graph <- wide_cluster_edges()
    graph$n_vertices <- 71L
    fit <- graph_cluster(graph, method = "louvain",
        backend = "metal", n_iterations = 6L, seed = 3L)
    expect_identical(summary(fit)$graph_offset_dtype, "int32")
})
