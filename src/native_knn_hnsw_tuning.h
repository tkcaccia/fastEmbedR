/*
 * SPDX-FileCopyrightText: 2026 Stefano Cacciatore
 * SPDX-License-Identifier: MIT
 *
 * The recall-0.99 policy is adapted from faissR 0.99.48, commit
 * 09f4c88fe8af431053a35a945db809a1da22033e. The source policy was
 * calibrated for FAISS HNSW. fastEmbedR records that provenance without
 * treating it as a per-call recall measurement for its native HNSW.
 */

#ifndef FASTEMBEDR_NATIVE_KNN_HNSW_TUNING_H
#define FASTEMBEDR_NATIVE_KNN_HNSW_TUNING_H

#include "native_knn_common.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace fastembedr {

enum class HnswShape {
  Small,
  MediumLow,
  LargeLow,
  LargeHigh,
  Other
};

struct HnswTuning {
  int m;
  int ef_construction;
  int ef_search;
  const char* shape;
  int k_bucket;
  bool reference_target_met;
  std::string rule;
};

inline HnswShape hnsw_shape(int n, int p) {
  if (n < 50000) return HnswShape::Small;
  if (n < 500000 && p <= 64) return HnswShape::MediumLow;
  if (n >= 500000 && p <= 64) return HnswShape::LargeLow;
  if (p >= 256) return HnswShape::LargeHigh;
  return HnswShape::Other;
}

inline int hnsw_k_bucket(int k) {
  if (k <= 15) return 15;
  if (k <= 30) return 30;
  if (k <= 50) return 50;
  return 100;
}

inline HnswTuning hnsw_euclidean_99(HnswShape shape, int bucket) {
  if (shape == HnswShape::LargeHigh) {
    return {24, 160, 120, "large_high_dim", bucket, true, ""};
  }
  if (shape == HnswShape::LargeLow) {
    if (bucket == 100) {
      return {32, 240, 220, "large_low_dim", bucket, true, ""};
    }
    return {24, 160, 120, "large_low_dim", bucket, true, ""};
  }
  if (shape == HnswShape::MediumLow) {
    if (bucket == 100) {
      return {12, 60, 100, "medium_low_dim", bucket, true, ""};
    }
    return {12, 80, 60, "medium_low_dim", bucket, true, ""};
  }
  if (shape == HnswShape::Small) {
    if (bucket == 15) {
      return {12, 60, 45, "small_n", bucket, true, ""};
    }
    const int search = bucket == 100 ? 100 : 60;
    return {12, 80, search, "small_n", bucket, true, ""};
  }
  return {24, 160, 120, "other", bucket, false, ""};
}

inline HnswTuning hnsw_cosine_99(HnswShape shape, int bucket) {
  if (shape == HnswShape::LargeHigh) {
    return {32, 240, 220, "large_high_dim", bucket, true, ""};
  }
  if (shape == HnswShape::LargeLow) {
    if (bucket == 100) {
      return {32, 240, 220, "large_low_dim", bucket, true, ""};
    }
    return {24, 160, 120, "large_low_dim", bucket, true, ""};
  }
  if (shape == HnswShape::MediumLow) {
    if (bucket == 50) {
      return {16, 100, 80, "medium_low_dim", bucket, true, ""};
    }
    const int search = bucket == 100 ? 100 : 60;
    return {12, 80, search, "medium_low_dim", bucket, true, ""};
  }
  if (shape == HnswShape::Small) {
    if (bucket == 15) {
      return {24, 160, 120, "small_n", bucket, false, ""};
    }
    if (bucket == 100) {
      return {16, 100, 100, "small_n", bucket, true, ""};
    }
    return {12, 80, 60, "small_n", bucket, true, ""};
  }
  return {32, 240, 220, "other", bucket, false, ""};
}

inline HnswTuning hnsw_correlation_99(HnswShape shape, int bucket) {
  if (shape == HnswShape::LargeHigh) {
    return {32, 240, 220, "large_high_dim", bucket, true, ""};
  }
  if (shape == HnswShape::LargeLow) {
    if (bucket == 100) {
      return {32, 240, 220, "large_low_dim", bucket, true, ""};
    }
    return {24, 160, 120, "large_low_dim", bucket, true, ""};
  }
  if (shape == HnswShape::MediumLow) {
    const int search = bucket == 100 ? 100 : 80;
    return {16, 100, search, "medium_low_dim", bucket, true, ""};
  }
  if (shape == HnswShape::Small) {
    if (bucket == 15) {
      return {48, 320, 400, "small_n", bucket, false, ""};
    }
    if (bucket == 30) {
      return {16, 100, 80, "small_n", bucket, true, ""};
    }
    const int search = bucket == 100 ? 100 : 60;
    return {12, 80, search, "small_n", bucket, true, ""};
  }
  return {32, 240, 220, "other", bucket, false, ""};
}

inline HnswTuning tune_native_hnsw(int n, int p, int k,
                                   KnnMetric metric,
                                   double target_recall) {
  if (!std::isfinite(target_recall) ||
      std::abs(target_recall - 0.99) > 1e-12) {
    Rcpp::stop("Native CPU HNSW supports target_recall = 0.99.");
  }
  const HnswShape shape = hnsw_shape(n, p);
  const int bucket = hnsw_k_bucket(k);
  HnswTuning tuning = metric == KnnMetric::Euclidean ?
    hnsw_euclidean_99(shape, bucket) :
    (metric == KnnMetric::Cosine ?
      hnsw_cosine_99(shape, bucket) :
      hnsw_correlation_99(shape, bucket));
  tuning.m = std::min(tuning.m, n);
  tuning.ef_construction = std::max(tuning.m, tuning.ef_construction);
  tuning.ef_search = std::max(k, tuning.ef_search);
  const char* metric_name = metric == KnnMetric::Euclidean ? "euclidean" :
    (metric == KnnMetric::Cosine ? "cosine" : "correlation");
  tuning.rule = std::string("faissr_cpu_hnsw_") + metric_name + "_" +
    tuning.shape + "_k" + std::to_string(bucket) + "_recall99";
  return tuning;
}

}  // namespace fastembedr

#endif
