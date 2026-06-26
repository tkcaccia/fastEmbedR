#include <Rcpp.h>

using Rcpp::IntegerMatrix;
using Rcpp::IntegerVector;
using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;

bool embedding_cuda_available_impl() {
  return false;
}

NumericMatrix spectral_knn_init_cuda_impl(IntegerMatrix,
                                          NumericMatrix,
                                          int,
                                          int,
                                          int) {
  Rcpp::stop("CUDA spectral initialization is available only when the package is built with CUDA support.");
}

NumericMatrix knn_embed_cuda_impl(IntegerMatrix,
                                  NumericMatrix,
                                  NumericMatrix,
                                  std::string,
                                  int,
                                  int,
                                  double,
                                  double,
                                  int) {
  Rcpp::stop("CUDA embedding backend is available only when the package is built with CUDA support.");
}

NumericMatrix knn_umap_cuda_fused_impl(IntegerMatrix,
                                       NumericMatrix,
                                       int,
                                       int,
                                       double,
                                       double,
                                       double,
                                       int,
                                       int,
                                       int) {
  Rcpp::stop("CUDA fused UMAP is available only when the package is built with CUDA support.");
}

NumericMatrix knn_umap_cuda_fused_float_impl(IntegerMatrix,
                                             SEXP,
                                             int,
                                             int,
                                             double,
                                             double,
                                             double,
                                             int,
                                             int,
                                             int) {
  Rcpp::stop("CUDA float32 fused UMAP is available only when the package is built with CUDA support.");
}

List umap_cuda_graph_dump_impl(IntegerMatrix,
                               NumericMatrix) {
  Rcpp::stop("CUDA UMAP graph dump is available only when the package is built with CUDA support.");
}

NumericMatrix umap_cuda_optimize_coo_impl(IntegerVector,
                                          IntegerVector,
                                          SEXP,
                                          SEXP,
                                          NumericMatrix,
                                          int,
                                          int,
                                          double,
                                          double,
                                          double,
                                          int,
                                          int) {
  Rcpp::stop("CUDA COO UMAP optimizer is available only when the package is built with CUDA support.");
}

NumericMatrix knn_tsne_exact_cuda_impl(IntegerMatrix,
                                       NumericMatrix,
                                       NumericMatrix,
                                       int,
                                       double,
                                       double,
                                       int,
                                       int,
                                       double,
                                       double,
                                       double,
                                       int) {
  Rcpp::stop("CUDA exact t-SNE is available only when the package is built with CUDA support.");
}

List knn_tsne_opentsne_cuda_impl(IntegerMatrix,
                                 NumericMatrix,
                                 NumericMatrix,
                                 bool,
                                 int,
                                 double,
                                 int,
                                 int,
                                 double,
                                 double,
                                 double,
                                 bool,
                                 double,
                                 double,
                                 double,
                                 double,
                                 std::string,
                                 int,
                                 bool) {
  Rcpp::stop("CUDA openTSNE FFT-grid is available only when the package is built with the native CUDA openTSNE backend.");
}

List knn_tsne_opentsne_cuda_float_impl(IntegerMatrix,
                                       SEXP,
                                       NumericMatrix,
                                       bool,
                                       int,
                                       double,
                                       int,
                                       int,
                                       double,
                                       double,
                                       double,
                                       bool,
                                       double,
                                       double,
                                       double,
                                       double,
                                       std::string,
                                       int,
                                       bool) {
  Rcpp::stop("CUDA float32 openTSNE FFT-grid is available only when the package is built with the native CUDA openTSNE backend.");
}

List standardize_cuda_impl(NumericMatrix) {
  Rcpp::stop("CUDA preprocessing is available only when the package is built with CUDA support.");
}

NumericMatrix project_embedding_knn_cuda_impl(NumericMatrix,
                                              IntegerMatrix,
                                              NumericMatrix) {
  Rcpp::stop("CUDA projection is available only when the package is built with CUDA support.");
}

NumericMatrix interpolate_landmark_layout_cuda_impl(NumericMatrix,
                                                    IntegerVector,
                                                    IntegerMatrix,
                                                    NumericMatrix,
                                                    int) {
  Rcpp::stop("CUDA landmark interpolation is available only when the package is built with CUDA support.");
}

List landmark_project_interpolate_knn_confidence_cuda_impl(NumericMatrix,
                                                           NumericMatrix,
                                                           NumericMatrix,
                                                           IntegerVector,
                                                           int) {
  Rcpp::stop("CUDA fused landmark projection is available only when the package is built with CUDA support.");
}

NumericVector knn_structure_score_cuda_impl(NumericMatrix,
                                            IntegerMatrix,
                                            IntegerVector,
                                            int,
                                            IntegerVector,
                                            int) {
  Rcpp::stop("CUDA scoring is available only when the package is built with CUDA support.");
}

double silhouette_score_cuda_impl(NumericMatrix,
                                  IntegerVector,
                                  int) {
  Rcpp::stop("CUDA scoring is available only when the package is built with CUDA support.");
}

NumericMatrix rsvd_multiply_cuda_impl(NumericMatrix,
                                      NumericMatrix,
                                      bool) {
  Rcpp::stop("CUDA RSVD matrix multiply is available only when the package is built with CUDA support.");
}
