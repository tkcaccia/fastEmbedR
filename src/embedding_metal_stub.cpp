#include <Rcpp.h>

using Rcpp::IntegerMatrix;
using Rcpp::IntegerVector;
using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;

bool embedding_metal_available_impl() {
  return false;
}

NumericMatrix spectral_knn_init_metal_impl(IntegerMatrix,
                                           NumericMatrix,
                                           int,
                                           int,
                                           int) {
  Rcpp::stop("Metal spectral initialization is only available on macOS with Metal support.");
}

List standardize_metal_impl(NumericMatrix) {
  Rcpp::stop("Metal preprocessing backend is only available on macOS with Metal support.");
}

NumericMatrix project_embedding_knn_metal_impl(NumericMatrix,
                                               IntegerMatrix,
                                               NumericMatrix) {
  Rcpp::stop("Metal projection backend is only available on macOS with Metal support.");
}

List project_embedding_affine_metal_impl(NumericMatrix,
                                         NumericMatrix,
                                         NumericMatrix,
                                         IntegerMatrix,
                                         NumericMatrix,
                                         int,
                                         double,
                                         double) {
  Rcpp::stop("Metal affine projection backend is only available on macOS with Metal support.");
}

NumericMatrix interpolate_landmark_layout_metal_impl(NumericMatrix,
                                                     IntegerVector,
                                                     IntegerMatrix,
                                                     NumericMatrix,
                                                     int) {
  Rcpp::stop("Metal landmark interpolation backend is only available on macOS with Metal support.");
}

NumericMatrix landmark_project_interpolate_metal_impl(NumericMatrix,
                                                      NumericMatrix,
                                                      NumericMatrix,
                                                      IntegerVector,
                                                      int) {
  Rcpp::stop("Metal fused landmark projection backend is only available on macOS with Metal support.");
}

List landmark_project_interpolate_knn_confidence_metal_impl(NumericMatrix,
                                                            NumericMatrix,
                                                            NumericMatrix,
                                                            IntegerVector,
                                                            int) {
  Rcpp::stop("Metal fused landmark projection/confidence backend is only available on macOS with Metal support.");
}

NumericVector knn_structure_score_metal_impl(NumericMatrix,
                                             IntegerMatrix,
                                             IntegerVector,
                                             int,
                                             IntegerVector,
                                             int) {
  Rcpp::stop("Metal structure scoring backend is only available on macOS with Metal support.");
}

double silhouette_score_metal_impl(NumericMatrix,
                                   IntegerVector,
                                   int) {
  Rcpp::stop("Metal silhouette scoring backend is only available on macOS with Metal support.");
}

NumericMatrix knn_embed_metal_impl(IntegerMatrix,
                                   NumericMatrix,
                                   NumericMatrix,
                                   std::string,
                                   int,
                                   int,
                                   double,
                                   double,
                                   int) {
  Rcpp::stop("Metal embedding backend is only available on macOS with Metal support.");
}

NumericMatrix knn_embed_metal_csr_impl(IntegerVector,
                                       IntegerVector,
                                       NumericVector,
                                       NumericMatrix,
                                       int,
                                       int,
                                       double,
                                       double,
                                       double,
                                       double,
                                       int,
                                       int) {
  Rcpp::stop("Metal CSR embedding backend is only available on macOS with Metal support.");
}

NumericMatrix knn_umap_refine_rows_metal_impl(IntegerMatrix,
                                              NumericMatrix,
                                              IntegerVector,
                                              NumericMatrix,
                                              int,
                                              double,
                                              int,
                                              double,
                                              double,
                                              int) {
  Rcpp::stop("Metal UMAP landmark refinement backend is only available on macOS with Metal support.");
}

NumericMatrix rsvd_multiply_metal_impl(NumericMatrix,
                                       NumericMatrix,
                                       bool) {
  Rcpp::stop("Metal RSVD matrix multiply is only available on macOS with Metal support.");
}

List transform_tsne_metal_impl(NumericMatrix,
                               IntegerMatrix,
                               NumericMatrix,
                               NumericMatrix,
                               bool,
                               std::string,
                               double,
                               int,
                               int,
                               double,
                               double,
                               double,
                               double,
                               double,
                               double,
                               double,
                               int,
                               int,
                               int) {
  Rcpp::stop("Metal t-SNE transform is only available on macOS with Metal support.");
}

List knn_tsne_opentsne_metal_impl(IntegerMatrix,
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
  Rcpp::stop("Metal openTSNE optimizer is only available on macOS with Metal support.");
}

List metal_fft512_stockham_diagnostic_impl(int,
                                           bool,
                                           int) {
  Rcpp::stop("Metal FFT diagnostics are only available on macOS with Metal support.");
}

List metal_mpsgraph_fft_diagnostic_impl(int,
                                        int,
                                        int) {
  Rcpp::stop("MPSGraph FFT diagnostics are only available on macOS with Metal support.");
}

List metal_mpsgraph_convolution_diagnostic_impl(int,
                                                int,
                                                int) {
  Rcpp::stop("MPSGraph convolution diagnostics are only available on macOS with Metal support.");
}
