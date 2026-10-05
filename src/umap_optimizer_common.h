#ifndef FASTEMBEDR_UMAP_OPTIMIZER_COMMON_H
#define FASTEMBEDR_UMAP_OPTIMIZER_COMMON_H

#include <cstddef>
#include <utility>

std::pair<double, double> fastembedr_umap_curve(double min_dist);
double fastembedr_umap_pow(double value, double exponent);
int fastembedr_umap_negative_vertex(int n, int seed, int epoch,
                                   std::size_t edge, int sample);

#endif
