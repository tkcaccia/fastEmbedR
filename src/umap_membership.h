// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#ifndef FASTEMBEDR_UMAP_MEMBERSHIP_H
#define FASTEMBEDR_UMAP_MEMBERSHIP_H

#include <algorithm>
#include <cmath>
#include <limits>

struct UmapMembershipScale {
  float rho;
  float sigma;
};

inline UmapMembershipScale umap_membership_scale(
    const float* distances, int k, double global_mean) {
  double rho = std::numeric_limits<double>::infinity();
  double row_sum = 0.0;
  int row_count = 0;
  for (int j = 0; j < k; ++j) {
    const float d = distances[j];
    if (!std::isfinite(d)) continue;
    if (d >= 0.0f) {
      row_sum += static_cast<double>(d);
      ++row_count;
    }
    if (d > 0.0f && static_cast<double>(d) < rho) {
      rho = static_cast<double>(d);
    }
  }
  if (!std::isfinite(rho)) rho = 0.0;

  const double target = std::log2(static_cast<double>(k));
  const double maximum = std::numeric_limits<double>::max();
  double sigma = 1.0;
  double best = sigma;
  double best_diff = maximum;
  double lo = 0.0;
  double hi = maximum;
  for (int iteration = 0; iteration < 64; ++iteration) {
    double sum = 0.0;
    const double safe_sigma = std::max(sigma, 1.0e-12);
    for (int j = 0; j < k; ++j) {
      const float raw = distances[j];
      if (!std::isfinite(raw)) continue;
      const double d = static_cast<double>(raw) - rho;
      sum += d <= 0.0 ? 1.0 :
        std::exp(-d / safe_sigma);
    }
    const double diff = std::abs(sum - target);
    if (diff < best_diff) {
      best_diff = diff;
      best = sigma;
    }
    if (sum > target) {
      hi = sigma;
      sigma = 0.5 * (lo + hi);
    } else {
      lo = sigma;
      sigma = hi == maximum ? sigma * 2.0 : 0.5 * (lo + hi);
    }
    if (diff < 1.0e-5) break;
  }
  const double row_mean = row_count > 0 ?
    row_sum / static_cast<double>(row_count) : global_mean;
  const double floor = 1.0e-3 * (rho > 0.0 ? row_mean : global_mean);
  return {static_cast<float>(rho),
          static_cast<float>(std::max(std::max(best, floor), 1.0e-12))};
}

#endif
