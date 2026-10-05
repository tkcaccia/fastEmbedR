#ifndef FASTEMBEDR_MASSIVE_UMAP_CUDA_H
#define FASTEMBEDR_MASSIVE_UMAP_CUDA_H

extern "C" void* fastembedr_massive_umap_cuda_create(
    int vertices, int dimensions, int edge_capacity,
    int epochs, int negatives,
    float learning_rate, float a, float b, float repulsion,
    float max_weight, unsigned int seed, int managed_layout);
extern "C" int fastembedr_massive_umap_cuda_upload(
    void* context, int first, int rows, const float* layout);
extern "C" int fastembedr_massive_umap_cuda_step(
    void* context, const int* heads, const int* tails,
    const float* weights, const float* periods, int edges, int epoch);
extern "C" int fastembedr_massive_umap_cuda_finish_epoch(void* context);
extern "C" int fastembedr_massive_umap_cuda_download(
    void* context, int first, int rows, float* layout);
extern "C" void fastembedr_massive_umap_cuda_destroy(void* context);
extern "C" const char* fastembedr_cuda_embedding_last_error();

#endif
