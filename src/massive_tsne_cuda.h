#ifndef FASTEMBEDR_MASSIVE_TSNE_CUDA_H
#define FASTEMBEDR_MASSIVE_TSNE_CUDA_H

extern "C" void* fastembedr_massive_tsne_cuda_create(
    int vertices, int edge_capacity);
extern "C" int fastembedr_massive_tsne_cuda_upload(
    void* context, int field, int first, int rows, const float* values);
extern "C" int fastembedr_massive_tsne_cuda_begin(void* context);
extern "C" int fastembedr_massive_tsne_cuda_repulsion_ms(
    void* context, float* elapsed_ms);
extern "C" int fastembedr_massive_tsne_cuda_attract(
    void* context, const int* heads, const int* tails,
    const float* weights, int edges, float exaggeration);
extern "C" int fastembedr_massive_tsne_cuda_finish(
    void* context, float learning_rate, float momentum,
    float min_gain, float max_step_norm);
extern "C" int fastembedr_massive_tsne_cuda_download(
    void* context, int field, int first, int rows, float* values);
extern "C" void fastembedr_massive_tsne_cuda_destroy(void* context);
extern "C" int fastembedr_cuda_opentsne_fft_grid_size(int vertices);

#ifdef FASTEMBEDR_HAS_NCCL
extern "C" void* fastembedr_massive_tsne_multigpu_create(
    int vertices, int edge_capacity, const int* devices,
    int device_count);
extern "C" void fastembedr_massive_tsne_multigpu_destroy(
    void* context);
extern "C" int fastembedr_massive_tsne_multigpu_upload(
    void* context, int field, int first, int rows,
    const float* values);
extern "C" int fastembedr_massive_tsne_multigpu_begin(
    void* context);
extern "C" int fastembedr_massive_tsne_multigpu_repulsion_ms(
    void* context, float* elapsed_ms);
extern "C" int fastembedr_massive_tsne_multigpu_attract(
    void* context, int rank, const int* heads, const int* tails,
    const float* weights, int edges, float exaggeration);
extern "C" int fastembedr_massive_tsne_multigpu_finish(
    void* context, float learning_rate, float momentum,
    float min_gain, float max_step_norm);
extern "C" int fastembedr_massive_tsne_multigpu_gather_state(
    void* context);
extern "C" int fastembedr_massive_tsne_multigpu_download(
    void* context, int field, int first, int rows, float* values);
#endif

#endif
