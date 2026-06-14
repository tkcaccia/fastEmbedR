#!/usr/bin/env Rscript

out_root <- "/Users/stefano/Documents/fastEmbedR/Data"
download_root <- file.path(out_root, "_downloads")
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)
dir.create(download_root, recursive = TRUE, showWarnings = FALSE)

message("Writing benchmark datasets to: ", out_root)

need_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop("Package `", pkg, "` is required for this preparation script.", call. = FALSE)
  }
}

download_if_missing <- function(url, dest, mode = "wb", timeout = 1800) {
  dir.create(dirname(dest), recursive = TRUE, showWarnings = FALSE)
  if (!file.exists(dest) || file.info(dest)$size == 0) {
    old_timeout <- getOption("timeout")
    on.exit(options(timeout = old_timeout), add = TRUE)
    options(timeout = max(timeout, old_timeout))
    message("Downloading ", url)
    utils::download.file(url, destfile = dest, mode = mode, quiet = FALSE)
  }
  dest
}

save_benchmark_dataset <- function(name, data, labels, material_methods, metadata = list()) {
  folder <- file.path(out_root, name)
  dir.create(folder, recursive = TRUE, showWarnings = FALSE)
  dataset <- list(
    data = data,
    labels = labels,
    material_methods = material_methods,
    metadata = c(
      list(
        name = name,
        n = nrow(data),
        p = ncol(data),
        label_levels = if (is.null(labels)) NULL else sort(unique(as.character(labels))),
        created = as.character(Sys.time())
      ),
      metadata
    )
  )
  save(dataset, file = file.path(folder, paste0(name, ".RData")), compress = "gzip")
  writeLines(
    c(
      paste0("# ", name),
      "",
      material_methods,
      "",
      paste0("Saved object: `dataset`"),
      paste0("Rows: ", nrow(data)),
      paste0("Columns: ", ncol(data)),
      paste0("Labels: ", if (is.null(labels)) "none" else length(unique(labels)))
    ),
    con = file.path(folder, "MATERIALS_AND_METHODS.md")
  )
  invisible(file.path(folder, paste0(name, ".RData")))
}

trimap_reference <- function(dataset) {
  descriptions <- list(
    MNIST = paste(
      "TriMap benchmark context: Amid and Warmuth (2019),",
      "'TriMap: Large-scale Dimensionality Reduction Using Triplets',",
      "listed MNIST as a 70K-image handwritten digit benchmark with digits",
      "0-9 and 28 x 28 images."
    ),
    FashionMNIST = paste(
      "TriMap benchmark context: Amid and Warmuth (2019) listed",
      "Fashion-MNIST as a 70K gray-scale clothing-image benchmark, with",
      "items such as t-shirt, pullover, and bag, represented as 28 x 28 images."
    ),
    USPS = paste(
      "TriMap benchmark context: Amid and Warmuth (2019) listed USPS as",
      "an 11K-image handwritten digit benchmark with digits 0-9 and 16 x 16",
      "images."
    ),
    COIL20 = paste(
      "TriMap benchmark context: Amid and Warmuth (2019) listed COIL-20",
      "as a 1,440-image gray-scale object benchmark containing 20 objects in",
      "uniformly sampled orientations, 5 degrees of rotation and 72 images per",
      "object, with processed images having background removed and cropped to",
      "128 x 128."
    ),
    MetRef = paste(
      "MetRef benchmark context: this metabolomics reference dataset is used",
      "as a compact labelled sanity-check dataset for embedding and clustering",
      "benchmarks. The saved matrix follows the KODAMA preprocessing requested",
      "for fastEmbedR: zero-sum variables removed, KODAMA normalization, and",
      "KODAMA scaling."
    ),
    TabulaMuris = paste(
      "TriMap benchmark context: Amid and Warmuth (2019) listed Tabula",
      "Muris as an approximately 54K-cell single-cell transcriptome benchmark",
      "from mouse spanning 20 organs."
    )
  )
  descriptions[[dataset]]
}

read_idx_images <- function(path) {
  con <- gzfile(path, "rb")
  on.exit(close(con), add = TRUE)
  magic <- readBin(con, "integer", n = 1L, size = 4L, endian = "big")
  if (!identical(magic, 2051L)) stop("Unexpected IDX image magic in ", path)
  n <- readBin(con, "integer", n = 1L, size = 4L, endian = "big")
  rows <- readBin(con, "integer", n = 1L, size = 4L, endian = "big")
  cols <- readBin(con, "integer", n = 1L, size = 4L, endian = "big")
  values <- readBin(con, "integer", n = as.double(n) * rows * cols, size = 1L, signed = FALSE)
  matrix(as.integer(values), nrow = n, ncol = rows * cols, byrow = TRUE)
}

read_idx_labels <- function(path) {
  con <- gzfile(path, "rb")
  on.exit(close(con), add = TRUE)
  magic <- readBin(con, "integer", n = 1L, size = 4L, endian = "big")
  if (!identical(magic, 2049L)) stop("Unexpected IDX label magic in ", path)
  n <- readBin(con, "integer", n = 1L, size = 4L, endian = "big")
  as.integer(readBin(con, "integer", n = n, size = 1L, signed = FALSE))
}

prepare_mnist <- function() {
  cache <- file.path("/Users/stefano/Documents/umap/results/dataset_cache/mnist")
  urls <- c(
    train_images = "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    train_labels = "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    test_images = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    test_labels = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"
  )
  files <- file.path(cache, basename(urls))
  names(files) <- names(urls)
  for (i in names(urls)) download_if_missing(urls[[i]], files[[i]])
  data <- rbind(read_idx_images(files[["train_images"]]), read_idx_images(files[["test_images"]]))
  labels <- factor(c(read_idx_labels(files[["train_labels"]]), read_idx_labels(files[["test_labels"]])))
  material <- paste(
    "MNIST handwritten digit images were obtained from the public IDX files",
    "mirrored at `storage.googleapis.com/cvdf-datasets/mnist`.",
    "The 60,000 training and 10,000 test images were concatenated, flattened",
    "from 28 x 28 grayscale pixels to 784 integer pixel features, and labels",
    "were stored as digit factors. No scaling, PCA, or dimensionality reduction",
    "was applied in this saved file.",
    trimap_reference("MNIST")
  )
  save_benchmark_dataset("MNIST", data, labels, material, list(source_url = unname(urls)))
}

prepare_fashion_mnist <- function() {
  cache <- file.path("/Users/stefano/Documents/umap/results/extended_dr_cache/downloads/fashion_mnist")
  urls <- c(
    train_images = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz",
    train_labels = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz",
    test_images = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz",
    test_labels = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz"
  )
  files <- file.path(cache, basename(urls))
  names(files) <- names(urls)
  for (i in names(urls)) download_if_missing(urls[[i]], files[[i]])
  data <- rbind(read_idx_images(files[["train_images"]]), read_idx_images(files[["test_images"]]))
  labels <- factor(c(read_idx_labels(files[["train_labels"]]), read_idx_labels(files[["test_labels"]])))
  class_names <- c(
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
  )
  labels <- factor(class_names[as.integer(as.character(labels)) + 1L], levels = class_names)
  material <- paste(
    "Fashion-MNIST was obtained from the Zalando Research GitHub IDX files.",
    "The 60,000 training and 10,000 test images were concatenated, flattened",
    "from 28 x 28 grayscale pixels to 784 integer pixel features, and labels",
    "were mapped to the ten official clothing classes. No scaling, PCA, or",
    "dimensionality reduction was applied in this saved file.",
    trimap_reference("FashionMNIST")
  )
  save_benchmark_dataset("FashionMNIST", data, labels, material, list(source_url = unname(urls)))
}

prepare_usps <- function() {
  need_pkg("Rdimtools")
  data("usps", package = "Rdimtools", envir = environment())
  data <- as.matrix(usps$data)
  storage.mode(data) <- "double"
  labels <- factor(usps$label)
  material <- paste(
    "USPS handwritten digit data were loaded from the public dataset bundled",
    "with the CRAN package Rdimtools. The data contain 11,000 grayscale digit",
    "images flattened to 256 features. The saved matrix is the Rdimtools",
    "feature matrix with digit labels; no additional scaling, PCA, or",
    "dimensionality reduction was applied.",
    trimap_reference("USPS")
  )
  save_benchmark_dataset(
    "USPS",
    data,
    labels,
    material,
    list(source_package = "Rdimtools", original_dataset = "USPS handwritten digits")
  )
}

prepare_coil20 <- function() {
  need_pkg("png")
  url <- "https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip"
  zip_path <- file.path(download_root, "COIL20", "coil-20-proc.zip")
  download_if_missing(url, zip_path, timeout = 1800)
  extract_dir <- file.path(download_root, "COIL20", "coil-20-proc")
  if (!dir.exists(extract_dir) || length(list.files(extract_dir, recursive = TRUE)) == 0L) {
    dir.create(extract_dir, recursive = TRUE, showWarnings = FALSE)
    utils::unzip(zip_path, exdir = extract_dir)
  }
  files <- list.files(extract_dir, pattern = "\\.(png|pgm)$", recursive = TRUE, full.names = TRUE, ignore.case = TRUE)
  if (length(files) == 0L) stop("No COIL-20 images found after extracting ", zip_path)
  files <- sort(files)
  labels <- sub(".*obj([0-9]+).*", "\\1", basename(files))
  imgs <- lapply(files, function(f) {
    x <- png::readPNG(f)
    if (length(dim(x)) == 3L) x <- x[, , 1L]
    as.integer(round(as.vector(t(x)) * 255))
  })
  data <- do.call(rbind, imgs)
  labels <- factor(labels)
  material <- paste(
    "COIL-20 processed object images were downloaded from Columbia CAVE as",
    "`coil-20-proc.zip`. The 1,440 images were read from the processed PNG",
    "files, converted to grayscale if necessary, flattened to pixel features,",
    "and labelled by the object identifier parsed from each file name. No",
    "scaling, PCA, or dimensionality reduction was applied in this saved file.",
    trimap_reference("COIL20")
  )
  save_benchmark_dataset("COIL20", data, labels, material, list(source_url = url, files = basename(files)))
}

prepare_metref <- function() {
  need_pkg("KODAMA")
  data("MetRef", package = "KODAMA", envir = environment())
  u <- MetRef$data
  zero_sum <- which(colSums(u) == 0)
  if (length(zero_sum) > 0L) {
    u <- u[, -zero_sum, drop = FALSE]
  }
  u <- KODAMA::normalization(u)$newXtrain
  data <- KODAMA::scaling(u)$newXtrain
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  labels <- as.factor(MetRef$donor)
  material <- paste(
    "MetRef was loaded from the KODAMA R package using `data(MetRef)`. The",
    "preprocessing follows the requested benchmark recipe exactly: start from",
    "`MetRef$data`, remove variables whose column sum is zero, apply",
    "`KODAMA::normalization(u)$newXtrain`, then apply",
    "`KODAMA::scaling(u)$newXtrain`. Labels are donor identifiers stored as",
    "`as.factor(MetRef$donor)`. No PCA or embedding was applied in the saved",
    "benchmark matrix.",
    trimap_reference("MetRef")
  )
  save_benchmark_dataset(
    "MetRef",
    data,
    labels,
    material,
    list(
      source_package = "KODAMA",
      source_dataset = "MetRef",
      preprocessing = "remove zero-sum columns, KODAMA normalization, KODAMA scaling",
      removed_zero_sum_columns = length(zero_sum),
      donor = labels,
      gender = as.factor(MetRef$gender)
    )
  )
}

prepare_tabula_muris <- function() {
  if (requireNamespace("ExperimentHub", quietly = TRUE) &&
      requireNamespace("AnnotationHub", quietly = TRUE) &&
      requireNamespace("SingleCellExperiment", quietly = TRUE) &&
      requireNamespace("SummarizedExperiment", quietly = TRUE)) {
    return(prepare_tabula_muris_experimenthub())
  }
  prepare_tabula_muris_ebi()
}

preprocess_tabula_muris_counts <- function(counts) {
  need_pkg("Matrix")
  need_pkg("irlba")
  message("Preprocessing Tabula Muris: library-size normalization, log1p, top HVGs, PCA50")
  lib_size <- Matrix::colSums(counts)
  positive <- lib_size > 0
  if (!all(positive)) {
    counts <- counts[, positive, drop = FALSE]
    lib_size <- lib_size[positive]
  }
  scale_factor <- stats::median(lib_size) / lib_size
  x <- Matrix::t(Matrix::t(counts) * scale_factor)
  x <- log1p(x)
  means <- Matrix::rowMeans(x)
  vars <- Matrix::rowMeans(x ^ 2) - means ^ 2
  keep <- order(vars, decreasing = TRUE)[seq_len(min(2000L, length(vars)))]
  x_hvg_cells_by_genes <- Matrix::t(x[keep, , drop = FALSE])
  set.seed(1)
  pca <- irlba::prcomp_irlba(
    x_hvg_cells_by_genes,
    n = min(50L, ncol(x_hvg_cells_by_genes) - 1L),
    center = TRUE,
    scale. = TRUE
  )
  data <- as.matrix(pca$x)
  colnames(data) <- paste0("PC", seq_len(ncol(data)))
  list(data = data, keep_cells = positive, hvg = keep)
}

prepare_tabula_muris_experimenthub <- function() {
  need_pkg("Matrix")
  need_pkg("irlba")
  need_pkg("ExperimentHub")
  need_pkg("AnnotationHub")
  need_pkg("SingleCellExperiment")
  need_pkg("SummarizedExperiment")

  message("Loading Tabula Muris Droplet from ExperimentHub record EH1617")
  eh <- ExperimentHub::ExperimentHub()
  sce <- eh[["EH1617"]]
  counts <- SummarizedExperiment::assay(sce, "counts")
  coldata <- as.data.frame(SummarizedExperiment::colData(sce))
  labels <- coldata$cell_ontology_class
  if (is.null(labels) || all(is.na(labels))) labels <- coldata$free_annotation
  prep <- preprocess_tabula_muris_counts(counts)
  if (!all(prep$keep_cells)) {
    labels <- labels[prep$keep_cells]
    coldata <- coldata[prep$keep_cells, , drop = FALSE]
  }
  labels <- factor(labels)
  material <- paste(
    "Tabula Muris was loaded from the Bioconductor ExperimentHub record",
    "EH1617 via the TabulaMurisData resource `TabulaMurisDroplet`. This",
    "record provides processed 10x droplet single-cell RNA-seq data from",
    "the Tabula Muris Consortium as a SingleCellExperiment object with",
    "23,341 genes and 70,118 cells. Cells were labelled with the curated",
    "`cell_ontology_class` column from the object metadata. Counts were",
    "library-size normalized to the median library size, transformed with",
    "log1p, reduced to the top 2,000 highly variable genes by variance,",
    "and compressed to 50 principal components using irlba. The saved",
    "`$data` matrix is therefore a PCA50 benchmark matrix suitable for",
    "repeated UMAP/t-SNE KNN benchmarks without redoing single-cell",
    "preprocessing.",
    trimap_reference("TabulaMuris")
  )
  save_benchmark_dataset(
    "TabulaMuris",
    prep$data,
    labels,
    material,
    list(
      source = "Bioconductor ExperimentHub",
      source_package = "TabulaMurisData",
      experimenthub_id = "EH1617",
      object = "TabulaMurisDroplet",
      source_url = "https://bioconductor.org/packages/release/data/experiment/html/TabulaMurisData.html",
      original_source_url = "https://s3.amazonaws.com/czbiohub-tabula-muris/TM_droplet_mat.rds",
      preprocessing = "library-size normalization, log1p, top 2000 variance genes, irlba PCA50",
      tissue = factor(coldata$tissue),
      mouse_id = factor(coldata$mouse_id),
      method = factor(coldata$method)
    )
  )
}

prepare_tabula_muris_ebi <- function() {
  need_pkg("Matrix")
  need_pkg("data.table")
  need_pkg("irlba")
  base <- "https://www.ebi.ac.uk/gxa/sc/experiment/E-ENAD-15"
  design_url <- paste0(base, "/download?fileType=experiment-design&accessKey=")
  normalised_url <- paste0(base, "/download/zip?fileType=normalised&accessKey=")
  tab_dir <- file.path(download_root, "TabulaMuris")
  design_path <- file.path(tab_dir, "ExpDesign-E-ENAD-15.tsv")
  zip_path <- file.path(tab_dir, "E-ENAD-15-normalised.zip")
  download_if_missing(design_url, design_path, mode = "wb", timeout = 1800)
  download_if_missing(normalised_url, zip_path, mode = "wb", timeout = 3600)

  zip_files <- utils::unzip(zip_path, list = TRUE)
  mtx_name <- zip_files$Name[grepl("\\.mtx$", zip_files$Name, ignore.case = TRUE)][1L]
  col_name <- zip_files$Name[grepl("_cols$", zip_files$Name, ignore.case = TRUE)][1L]
  if (is.na(mtx_name) || is.na(col_name)) {
    stop("Could not find MatrixMarket matrix and column ID files in Tabula Muris archive.")
  }
  message("Reading Tabula Muris matrix from ZIP: ", mtx_name)
  mat <- Matrix::readMM(unz(zip_path, mtx_name, open = "r"))
  matrix_cols <- readLines(unz(zip_path, col_name, open = "r"), warn = FALSE)
  design <- data.table::fread(design_path, data.table = FALSE)
  label_col <- "Factor Value[inferred cell type - ontology labels]"
  if (!label_col %in% names(design)) {
    label_col <- "Sample Characteristic[inferred cell type - ontology labels]"
  }
  if (!"Assay" %in% names(design)) {
    stop("Tabula Muris experiment design does not contain an `Assay` column.")
  }
  row_match <- match(matrix_cols, design$Assay)
  if (anyNA(row_match)) {
    stop("Could not match all Tabula Muris matrix column IDs to experiment design `Assay` values.")
  }
  design <- design[row_match, , drop = FALSE]
  labels <- factor(design[[label_col]])
  tissue_col <- "Factor Value[organism part]"
  tissue <- if (tissue_col %in% names(design)) factor(design[[tissue_col]]) else NULL

  if (ncol(mat) == length(labels)) {
    cells_by_genes <- Matrix::t(mat)
  } else if (nrow(mat) == length(labels)) {
    cells_by_genes <- mat
  } else {
    stop("Tabula Muris matrix dimensions do not match experiment design rows.")
  }

  message("Preprocessing Tabula Muris: log1p, top HVGs, PCA50")
  x <- Matrix::drop0(cells_by_genes)
  x <- log1p(x)
  means <- Matrix::colMeans(x)
  vars <- Matrix::colMeans(x ^ 2) - means ^ 2
  keep <- order(vars, decreasing = TRUE)[seq_len(min(2000L, length(vars)))]
  x_hvg <- x[, keep, drop = FALSE]
  set.seed(1)
  pca <- irlba::prcomp_irlba(x_hvg, n = min(50L, ncol(x_hvg) - 1L), center = TRUE, scale. = TRUE)
  data <- as.matrix(pca$x)
  colnames(data) <- paste0("PC", seq_len(ncol(data)))
  material <- paste(
    "Tabula Muris was obtained from EMBL-EBI Single Cell Expression Atlas",
    "experiment E-ENAD-15, 'Single-cell RNA-seq analysis of 20 organs and",
    "tissues from individual mice creates a Tabula muris'. The normalised",
    "MatrixMarket archive and experiment-design metadata were downloaded from",
    "EBI. Cells were labelled by inferred cell type ontology labels. The",
    "expression matrix was oriented to cells x genes, transformed with log1p,",
    "reduced to the top 2,000 highly variable genes by variance, and compressed",
    "to 50 principal components using irlba. The saved `$data` matrix is",
    "therefore a PCA50 benchmark matrix suitable for repeated UMAP/t-SNE KNN",
    "benchmarks without redoing single-cell preprocessing.",
    trimap_reference("TabulaMuris")
  )
  save_benchmark_dataset(
    "TabulaMuris",
    data,
    labels,
    material,
    list(
      source_url = c(design_url, normalised_url),
      experiment = "E-ENAD-15",
      preprocessing = "log1p normalised values, top 2000 variance genes, irlba PCA50",
      tissue = tissue
    )
  )
}

write_dataset_manifest <- function(dataset_names) {
  rows <- lapply(dataset_names, function(name) {
    path <- file.path(out_root, name, paste0(name, ".RData"))
    if (!file.exists(path)) return(NULL)
    env <- new.env(parent = emptyenv())
    load(path, envir = env)
    if (!exists("dataset", envir = env, inherits = FALSE)) return(NULL)
    ds <- get("dataset", envir = env, inherits = FALSE)
    data.frame(
      dataset = name,
      path = normalizePath(path, mustWork = FALSE),
      n = nrow(ds$data),
      p = ncol(ds$data),
      labels = if (is.null(ds$labels)) NA_integer_ else length(unique(ds$labels)),
      file_mb = round(file.info(path)$size / 1024^2, 2),
      stringsAsFactors = FALSE
    )
  })
  manifest <- do.call(rbind, rows[!vapply(rows, is.null, logical(1))])
  write.csv(manifest, file.path(out_root, "dataset_manifest.csv"), row.names = FALSE)
  invisible(manifest)
}

main <- function() {
  requested <- commandArgs(trailingOnly = TRUE)
  known <- c("MNIST", "FashionMNIST", "USPS", "COIL20", "MetRef", "TabulaMuris")
  if (!length(requested)) requested <- known
  requested <- match.arg(requested, known, several.ok = TRUE)
  for (name in requested) {
    message("\n=== Preparing ", name, " ===")
    tryCatch(
      switch(
        name,
        MNIST = prepare_mnist(),
        FashionMNIST = prepare_fashion_mnist(),
        USPS = prepare_usps(),
        COIL20 = prepare_coil20(),
        MetRef = prepare_metref(),
        TabulaMuris = prepare_tabula_muris()
      ),
      error = function(e) {
        message("FAILED preparing ", name, ": ", conditionMessage(e))
      }
    )
  }
  write_dataset_manifest(known)
}

main()
