#ifndef FASTEMBEDR_MASSIVE_UMAP_METAL_H
#define FASTEMBEDR_MASSIVE_UMAP_METAL_H

#include <cstdint>

extern "C" {
void* fastembedr_massive_umap_metal_create(const float* init,
    int vertices, int dimensions, int epochs, int negatives,
    double learning_rate, double a, double b, double repulsion,
    int seed);
std::uint64_t fastembedr_massive_umap_metal_step(void* context,
    const int* heads, const int* tails, const int* negative_counts,
    int count, int epoch);
void fastembedr_massive_umap_metal_download(void* context,
    float* output);
void fastembedr_massive_umap_metal_destroy(void* context);
}

#endif
