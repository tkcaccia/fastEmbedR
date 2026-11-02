three_clique_graph <- function() {
    groups <- split(seq_len(18L), rep(seq_len(3L), each = 6L))
    from <- to <- integer()
    weight <- numeric()
    for (group in groups) {
        pairs <- utils::combn(group, 2L)
        from <- c(from, pairs[1L, ])
        to <- c(to, pairs[2L, ])
        weight <- c(weight, rep(1, ncol(pairs)))
    }
    from <- c(from, 6L, 12L)
    to <- c(to, 7L, 13L)
    weight <- c(weight, 0.15, 0.15)
    structure(
        list(
            from = from, to = to, weight = weight,
            n_vertices = 18L, n_edges = length(from)
        ),
        class = c("fastEmbedR_graph", "list")
    )
}
