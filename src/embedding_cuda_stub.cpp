#include <Rcpp.h>

#include "massive_pca_cuda.h"

using Rcpp::IntegerMatrix;
using Rcpp::IntegerVector;
using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;

extern "C" void* fastembedr_massive_pca_cuda_create(
    int, int, int, const float*, const float*, const float*) {
  return nullptr;
}

extern "C" int fastembedr_massive_pca_cuda_cross(
    void*, const float*, int, float*) {
  return 1;
}

extern "C" int fastembedr_massive_pca_cuda_project(
    void*, const float*, int, float*) {
  return 1;
}

extern "C" int fastembedr_massive_pca_cuda_action(
    void*, const float*, int, float*) {
  return 1;
}

extern "C" void fastembedr_massive_pca_cuda_destroy(void*) {}

extern "C" const char* fastembedr_massive_pca_cuda_error() {
  return "EXPERIMENTAL out-of-core CUDA PCA is unavailable in this build.";
}

extern "C" int fastembedr_massive_pca_cuda_memory(
    std::size_t*, std::size_t*, int*) {
  return 1;
}

extern "C" int fastembedr_massive_cuda_device_count(int*) {
  return 1;
}

extern "C" int fastembedr_massive_cuda_select_device(int) {
  return 1;
}

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
                                       int,
                                       int,
                                       bool) {
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
                                             int,
                                             int,
                                             bool) {
  Rcpp::stop("CUDA float32 fused UMAP is available only when the package is built with CUDA support.");
}

NumericMatrix knn_umap_cuda_fused_gpu_impl(SEXP,
                                           int,
                                           int,
                                           int,
                                           double,
                                           double,
                                           double,
                                           int,
                                           int,
                                           int,
                                           bool,
                                           int) {
  Rcpp::stop("CUDA GPU-resident UMAP is available only when the package is built with CUDA support.");
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

List knn_tsne_opentsne_cuda_gpu_impl(SEXP,
                                     int,
                                     NumericMatrix,
                                     bool,
                                     SEXP,
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
  Rcpp::stop("CUDA GPU-resident openTSNE FFT-grid is available only when the package is built with the native CUDA openTSNE backend.");
}

List standardize_cuda_impl(NumericMatrix) {
  Rcpp::stop("CUDA preprocessing is available only when the package is built with CUDA support.");
}

NumericMatrix project_embedding_knn_cuda_impl(NumericMatrix,
                                              IntegerMatrix,
                                              NumericMatrix) {
  Rcpp::stop("CUDA projection is available only when the package is built with CUDA support.");
}

SEXP massive_cuda_projector_create_impl(NumericMatrix, int, int) {
  Rcpp::stop("Persistent CUDA projection is unavailable; no CPU fallback was used.");
}

NumericMatrix massive_cuda_projector_batch_impl(
    SEXP, IntegerMatrix, NumericMatrix) {
  Rcpp::stop("Persistent CUDA projection is unavailable; no CPU fallback was used.");
}

void massive_cuda_projector_release_impl(SEXP) {
  Rcpp::stop("Persistent CUDA projection is unavailable; no CPU fallback was used.");
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

List transform_tsne_cuda_impl(NumericMatrix,
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
  Rcpp::stop(
    "CUDA t-SNE transform is available only when the package is built "
    "with CUDA support."
  );
}

List landmark_tsne_transform_cuda_gpu_impl(SEXP,
                                           SEXP,
                                           SEXP,
                                           SEXP,
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
                                           int,
                                           int,
                                           double,
                                           double) {
  Rcpp::stop("CUDA GPU-resident landmark t-SNE is available only when the package is built with CUDA support.");
}

List landmark_umap_project_refine_cuda_gpu_impl(SEXP,
                                                SEXP,
                                                SEXP,
                                                SEXP,
                                                IntegerVector,
                                                IntegerVector,
                                                int,
                                                int,
                                                double,
                                                int,
                                                double,
                                                double,
                                                int,
                                                int,
                                                double,
                                                double) {
  Rcpp::stop("CUDA GPU-resident landmark UMAP is available only when the package is built with CUDA support.");
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

List pca_tsvd_cuda_impl(SEXP,
                        int,
                        bool,
                        bool,
                        int,
                        int,
                        int,
                        int) {
  Rcpp::stop("native RAPIDS RAFT TSVD PCA is available only when the package is built with CUDA and RAFT support.");
}
