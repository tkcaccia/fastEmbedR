faissr_exact_graph_batch <- function(source, first, count, k,
        reference_chunk, threads) {
    fastEmbedR:::faissr_matrix_knn_batch_cpp(source, source,
        as.double(seq.int(first, length.out = count)), k,
        "cpu", "stream_exact", reference_chunk,
        256 * 1024^2, threads, TRUE)
}

faissr_exact_reference_batch <- function(query, reference, first,
        count, k, reference_chunk, threads) {
    if (identical(query$format, "synthetic")) {
        query <- massive_matrix(massive_read_rows(query, first, count))
        first <- 1L
    }
    fastEmbedR:::faissr_matrix_knn_batch_cpp(query, reference,
        as.double(seq.int(first, length.out = count)), k,
        "cpu", "stream_exact", reference_chunk,
        256 * 1024^2, threads, FALSE)
}

faissr_exact_sample_batch <- function(query, reference, rows, k,
        reference_chunk, threads, exclude_self) {
    fastEmbedR:::faissr_matrix_knn_batch_cpp(query, reference,
        as.double(rows), k, "cpu", "stream_exact", reference_chunk,
        256 * 1024^2, threads, exclude_self)
}
