#ifndef FASTEMBEDR_MASSIVE_PCA_CUDA_H
#define FASTEMBEDR_MASSIVE_PCA_CUDA_H

#include <cstddef>

extern "C" {
void* fastembedr_massive_pca_cuda_create(int rows, int columns, int rank,
                                         const float* center,
                                         const float* scale,
                                         const float* loadings);
int fastembedr_massive_pca_cuda_cross(void* context, const float* input,
                                      int rows, float* cross);
int fastembedr_massive_pca_cuda_project(void* context, const float* input,
                                        int rows, float* scores);
int fastembedr_massive_pca_cuda_action(void* context, const float* input,
                                       int rows, float* cross);
void fastembedr_massive_pca_cuda_destroy(void* context);
const char* fastembedr_massive_pca_cuda_error();
int fastembedr_massive_pca_cuda_memory(std::size_t* free_bytes,
                                       std::size_t* total_bytes,
                                       int* device);
int fastembedr_massive_cuda_device_count(int* count);
int fastembedr_massive_cuda_select_device(int device);
}

#endif
