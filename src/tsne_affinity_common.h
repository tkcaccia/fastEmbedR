#ifndef FASTEMBEDR_TSNE_AFFINITY_COMMON_H
#define FASTEMBEDR_TSNE_AFFINITY_COMMON_H

#include <algorithm>
#include <cfloat>
#include <cmath>

inline void tsne_row_probabilities_float(const float* distances,
                                         int k, double perplexity,
                                         float* row_p) {
  std::fill(row_p, row_p + k, 0.0f);
  float min_d2 = FLT_MAX;
  float max_d2 = 0.0f;
  for (int j = 0; j < k; ++j) {
    const float d2 = distances[j] * distances[j];
    min_d2 = std::min(min_d2, d2);
    max_d2 = std::max(max_d2, d2);
  }
  if (max_d2 - min_d2 <=
      FLT_EPSILON * std::max(1.0f, max_d2)) {
    std::fill(row_p, row_p + k, 1.0f / static_cast<float>(k));
    return;
  }

  float beta = 1.0f;
  float min_beta = -FLT_MAX;
  float max_beta = FLT_MAX;
  float sum_p = FLT_MIN;
  for (int iter = 0; iter < 200; ++iter) {
    sum_p = FLT_MIN;
    for (int j = 0; j < k; ++j) {
      const float d2 = distances[j] * distances[j] - min_d2;
      row_p[j] = std::exp(-beta * d2);
      sum_p += row_p[j];
    }
    float entropy = 0.0f;
    for (int j = 0; j < k; ++j) {
      const float d2 = distances[j] * distances[j] - min_d2;
      entropy += beta * d2 * row_p[j];
    }
    entropy = entropy / sum_p + std::log(sum_p);
    const float diff = entropy -
      static_cast<float>(std::log(perplexity));
    if (std::abs(diff) < 1e-5f) break;
    if (diff > 0.0f) {
      min_beta = beta;
      beta = (max_beta == FLT_MAX || max_beta == -FLT_MAX) ?
        beta * 2.0f : (beta + max_beta) * 0.5f;
    } else {
      max_beta = beta;
      beta = (min_beta == -FLT_MAX || min_beta == FLT_MAX) ?
        beta * 0.5f : (beta + min_beta) * 0.5f;
    }
    if (!std::isfinite(beta)) break;
  }

  if (!std::isfinite(sum_p) || sum_p <= FLT_MIN) {
    int tied = 0;
    for (int j = 0; j < k; ++j) {
      const float d2 = distances[j] * distances[j];
      if (std::abs(d2 - min_d2) <=
          FLT_EPSILON * std::max(1.0f, min_d2)) ++tied;
    }
    const float mass = 1.0f / static_cast<float>(std::max(1, tied));
    for (int j = 0; j < k; ++j) {
      const float d2 = distances[j] * distances[j];
      row_p[j] = std::abs(d2 - min_d2) <=
        FLT_EPSILON * std::max(1.0f, min_d2) ? mass : 0.0f;
    }
    return;
  }
  const float inverse = 1.0f / sum_p;
  for (int j = 0; j < k; ++j) row_p[j] *= inverse;
}

#endif
