#ifndef FASTEMBEDR_MASSIVE_UMAP_CUDA_H
#define FASTEMBEDR_MASSIVE_UMAP_CUDA_H

#include <cstddef>
#include <cstdint>

extern "C" void fastembedr_massive_spectral_cuda_destroy(void* context);
extern "C" void* fastembedr_massive_spectral_cuda_create(
    int vertices, int block, std::uint64_t edges, const double* inv_degree);
extern "C" int fastembedr_massive_spectral_cuda_upload(
    void* context, int field, std::size_t offset, std::size_t bytes,
    const void* values);
extern "C" int fastembedr_massive_spectral_cuda_multiply(
    void* context, const double* input, double* output);

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
extern "C" int fastembedr_massive_umap_cuda_cache_begin(
    void* context, std::uint64_t edges, double memory_limit);
extern "C" int fastembedr_massive_umap_cuda_cache_replay(
    void* context, int epoch);
extern "C" int fastembedr_massive_umap_cuda_finish_epoch(void* context);
extern "C" int fastembedr_massive_umap_cuda_download(
    void* context, int first, int rows, float* layout);
extern "C" void fastembedr_massive_umap_cuda_destroy(void* context);
extern "C" void* fastembedr_massive_umap_multigpu_create(
    int vertices, int dimensions, int edge_capacity, int epochs,
    int negatives,
    float learning_rate, float a, float b, float repulsion,
    float max_weight, unsigned int seed, const int* devices,
    int device_count, double delta_scale, void* retained_layouts = nullptr,
    double device_limit = 0.0);
extern "C" int fastembedr_massive_umap_multigpu_upload(
    void* context, int first, int rows, const float* layout);
extern "C" int fastembedr_massive_umap_multigpu_start_epoch(
    void* context);
extern "C" int fastembedr_massive_umap_multigpu_cache_begin(
    void* context, const unsigned long long* rank_edges);
extern "C" int fastembedr_massive_umap_multigpu_cache_replay(
    void* context, int rank, int epoch);
extern "C" int fastembedr_massive_umap_multigpu_step(
    void* context, int rank, const int* heads, const int* tails,
    const float* weights, const float* periods, int edges, int epoch);
extern "C" int fastembedr_massive_umap_multigpu_finish_epoch(
    void* context);
extern "C" double fastembedr_massive_umap_multigpu_collective_seconds(
    void* context);
extern "C" int fastembedr_massive_umap_multigpu_download(
    void* context, int first, int rows, float* layout);
extern "C" void fastembedr_massive_umap_multigpu_destroy(void* context);
// Returns 2 when a shard exceeds its available workspace budget.
extern "C" int fastembedr_cuda_umap_wide_init(
    int n, const double* offsets, const int* neighbors,
    const int* weight_bits, int iterations, unsigned int seed,
    const int* devices, int cards, double device_limit, float* output,
    int* shard_rows, double* shard_edges, double* clocks,
    void** retained = nullptr, double layout_scale = 0.0);
extern "C" int fastembedr_cuda_umap_wide_graph_build(
    const int* indices, const int* bits, int n, int k, int index_base,
    double mean, const int* devices, int cards, double limit,
    void** graph, double* edges, double* routed, double* times,
    float* max_weight);
extern "C" int fastembedr_cuda_umap_wide_graph_counts(
    void* graph, float minimum, double limit, double* scheduled);
extern "C" double fastembedr_cuda_umap_wide_graph_allocated_bytes(
    const void* graph);
extern "C" void fastembedr_cuda_umap_wide_graph_destroy(void* graph);
extern "C" int fastembedr_cuda_umap_wide_graph_step(
    void* context, void* graph, int epoch);
extern "C" const char* fastembedr_cuda_embedding_last_error();

#endif
