// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#ifndef FASTEMBEDR_NATIVE_KNN_CUDA_PILOT_H
#define FASTEMBEDR_NATIVE_KNN_CUDA_PILOT_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace fastembedr {

inline double query_pilot_distance_recall(
    const std::vector<float>& reference,
    const std::vector<float>& observed,
    int rows, int k) {
  double matches = 0.0;
  for (int row = 0; row < rows; ++row) {
    const std::size_t offset = static_cast<std::size_t>(row) * k;
    float cutoff = 0.0f;
    for (int column = 0; column < k; ++column) {
      const float value = reference[offset + column];
      if (!std::isfinite(value)) return 0.0;
      cutoff = std::max(cutoff, value);
    }
    const double tolerance = 1e-5 * std::max(1.0, double(cutoff));
    for (int column = 0; column < k; ++column) {
      const float value = observed[offset + column];
      if (std::isfinite(value) && value >= -tolerance &&
          value <= double(cutoff) + tolerance) {
        matches += 1.0;
      }
    }
  }
  return matches / (static_cast<double>(rows) * k);
}

}  // namespace fastembedr

#endif
