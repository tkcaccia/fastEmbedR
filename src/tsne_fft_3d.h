/*
 * SPDX-FileCopyrightText: 2026 Stefano Cacciatore
 * SPDX-License-Identifier: MIT
 *
 * Three-dimensional FFT repulsion through the t-SNE kernel potential.
 */

struct TsneFft3dWorkspace {
  int grid_size = 0;
  int fft_size = 0;
  int threads = 0;
  double sum_q = 0.0;
  double fft_elapsed_sec = 0.0;
  double correction_elapsed_sec = 0.0;
  std::vector<float> mass;
  std::vector<float> potential;
  std::vector<std::complex<float>> mass_fft;
  std::vector<std::complex<float>> kernel_fft;
  std::vector<std::complex<float>> line_scratch;
  std::vector<double> partial_sum;
  std::vector<int> bin_head;
  std::vector<int> bin_next;
  std::array<float, 343> self_kernel;
  FftPlanT<float> plan;

  void ensure(int grid, int n_threads) {
    if (grid_size == grid && threads == n_threads) return;
    grid_size = grid;
    fft_size = 2 * grid;
    threads = n_threads;
    const std::size_t grid_total =
      static_cast<std::size_t>(grid) * grid * grid;
    const std::size_t fft_total =
      static_cast<std::size_t>(fft_size) * fft_size * fft_size;
    mass.resize(grid_total);
    potential.resize(grid_total);
    mass_fft.resize(fft_total);
    kernel_fft.resize(fft_total);
    line_scratch.resize(static_cast<std::size_t>(threads) * fft_size);
    partial_sum.resize(static_cast<std::size_t>(threads));
    plan.ensure(fft_size);
  }
};

void tsne_fft_3d(std::vector<std::complex<float>>& values,
                 bool inverse, TsneFft3dWorkspace& ws) {
  const int size = ws.fft_size;
  const int lines = size * size;
  parallel_for(lines, ws.threads, [&](int begin, int end, int) {
    for (int line = begin; line < end; ++line) {
      fft_1d_t<float>(values.data() +
        static_cast<std::size_t>(line) * size, size, inverse, ws.plan);
    }
  });
  parallel_for(lines, ws.threads, [&](int begin, int end, int thread_id) {
    auto* scratch = ws.line_scratch.data() +
      static_cast<std::size_t>(thread_id) * size;
    for (int line = begin; line < end; ++line) {
      const int z = line / size;
      const int x = line % size;
      for (int y = 0; y < size; ++y) {
        scratch[y] = values[
          (static_cast<std::size_t>(z) * size + y) * size + x
        ];
      }
      fft_1d_t<float>(scratch, size, inverse, ws.plan);
      for (int y = 0; y < size; ++y) {
        values[(static_cast<std::size_t>(z) * size + y) * size + x] =
          scratch[y];
      }
    }
  });
  parallel_for(lines, ws.threads, [&](int begin, int end, int thread_id) {
    auto* scratch = ws.line_scratch.data() +
      static_cast<std::size_t>(thread_id) * size;
    for (int line = begin; line < end; ++line) {
      const int y = line / size;
      const int x = line % size;
      for (int z = 0; z < size; ++z) {
        scratch[z] = values[
          (static_cast<std::size_t>(z) * size + y) * size + x
        ];
      }
      fft_1d_t<float>(scratch, size, inverse, ws.plan);
      for (int z = 0; z < size; ++z) {
        values[(static_cast<std::size_t>(z) * size + y) * size + x] =
          scratch[z];
      }
    }
  });
}

void tsne_cubic_weights(float t, float* weight, float* derivative) {
  const float t2 = t * t;
  weight[0] = -t * (t - 1.0f) * (t - 2.0f) / 6.0f;
  weight[1] = (t + 1.0f) * (t - 1.0f) * (t - 2.0f) / 2.0f;
  weight[2] = -(t + 1.0f) * t * (t - 2.0f) / 2.0f;
  weight[3] = (t + 1.0f) * t * (t - 1.0f) / 6.0f;
  derivative[0] = (-3.0f * t2 + 6.0f * t - 2.0f) / 6.0f;
  derivative[1] = (3.0f * t2 - 4.0f * t - 1.0f) / 2.0f;
  derivative[2] = (-3.0f * t2 + 2.0f * t + 2.0f) / 2.0f;
  derivative[3] = (3.0f * t2 - 1.0f) / 6.0f;
}

void tsne_grid_point_3d(const float* y, const float* lower,
                        float inv_spacing, int grid, int* start,
                        float weight[3][4], float derivative[3][4]) {
  for (int axis = 0; axis < 3; ++axis) {
    const float coordinate = (y[axis] - lower[axis]) * inv_spacing;
    const int cell = std::max(1, std::min(grid - 3,
      static_cast<int>(std::floor(coordinate))));
    start[axis] = cell - 1;
    tsne_cubic_weights(coordinate - cell, weight[axis],
                       derivative[axis]);
  }
}

int tsne_fft_3d_grid_size(int n) {
  return n < 5000 ? 16 : 64;
}

float tsne_smooth_q(float squared_distance, float cutoff_squared,
                    float q_cutoff) {
  if (squared_distance >= cutoff_squared) {
    return 1.0f / (1.0f + squared_distance);
  }
  const float delta = squared_distance - cutoff_squared;
  return q_cutoff * (1.0f - q_cutoff * delta +
                     q_cutoff * q_cutoff * delta * delta);
}

void tsne_fft_3d_bins(const std::vector<float>& y, int n,
                      const std::array<float, 3>& lower,
                      float cutoff, int bin_count,
                      TsneFft3dWorkspace& ws) {
  ws.bin_head.assign(static_cast<std::size_t>(bin_count) *
                     bin_count * bin_count, -1);
  ws.bin_next.resize(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    const float* point = y.data() + static_cast<std::size_t>(i) * 3;
    int bin[3];
    for (int axis = 0; axis < 3; ++axis) {
      bin[axis] = std::max(0, std::min(bin_count - 1,
        static_cast<int>((point[axis] - lower[axis]) / cutoff)));
    }
    const std::size_t index =
      (static_cast<std::size_t>(bin[2]) * bin_count + bin[1]) *
      bin_count + bin[0];
    ws.bin_next[static_cast<std::size_t>(i)] = ws.bin_head[index];
    ws.bin_head[index] = i;
  }
}

void tsne_fft_3d_self(const float weight[3][4],
                      const float derivative[3][4],
                      const std::array<float, 343>& kernel,
                      double& value, double slope[3]) {
  float auto_weight[3][7] = {};
  float cross_weight[3][7] = {};
  for (int axis = 0; axis < 3; ++axis) {
    for (int gather = 0; gather < 4; ++gather) {
      for (int scatter = 0; scatter < 4; ++scatter) {
        const int delta = gather - scatter + 3;
        auto_weight[axis][delta] +=
          weight[axis][gather] * weight[axis][scatter];
        cross_weight[axis][delta] +=
          derivative[axis][gather] * weight[axis][scatter];
      }
    }
  }
  value = 0.0;
  slope[0] = slope[1] = slope[2] = 0.0;
  for (int z = 0; z < 7; ++z) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int x = 0; x < 7; ++x) {
        const float q = kernel[(z * 7 + yy) * 7 + x];
        value += q * auto_weight[0][x] * auto_weight[1][yy] *
          auto_weight[2][z];
        slope[0] += q * cross_weight[0][x] * auto_weight[1][yy] *
          auto_weight[2][z];
        slope[1] += q * auto_weight[0][x] * cross_weight[1][yy] *
          auto_weight[2][z];
        slope[2] += q * auto_weight[0][x] * auto_weight[1][yy] *
          cross_weight[2][z];
      }
    }
  }
}

void tsne_fft_3d_near(const std::vector<float>& y, int i,
                      const std::array<float, 3>& lower,
                      float cutoff, float cutoff_squared,
                      float q_cutoff, int bin_count,
                      const TsneFft3dWorkspace& ws,
                      double& value, double slope[3]) {
  const float* point = y.data() + static_cast<std::size_t>(i) * 3;
  int bin[3];
  for (int axis = 0; axis < 3; ++axis) {
    bin[axis] = std::max(0, std::min(bin_count - 1,
      static_cast<int>((point[axis] - lower[axis]) / cutoff)));
  }
  for (int z = std::max(0, bin[2] - 1);
       z <= std::min(bin_count - 1, bin[2] + 1); ++z) {
    for (int yy = std::max(0, bin[1] - 1);
         yy <= std::min(bin_count - 1, bin[1] + 1); ++yy) {
      for (int x = std::max(0, bin[0] - 1);
           x <= std::min(bin_count - 1, bin[0] + 1); ++x) {
        const std::size_t index =
          (static_cast<std::size_t>(z) * bin_count + yy) *
          bin_count + x;
        for (int j = ws.bin_head[index]; j >= 0;
             j = ws.bin_next[static_cast<std::size_t>(j)]) {
          if (j == i) continue;
          const float* other = y.data() + static_cast<std::size_t>(j) * 3;
          const float dx = point[0] - other[0];
          const float dy = point[1] - other[1];
          const float dz = point[2] - other[2];
          const float distance = dx * dx + dy * dy + dz * dz;
          if (distance >= cutoff_squared) continue;
          const float q = 1.0f / (1.0f + distance);
          const float delta = distance - cutoff_squared;
          const float smooth_derivative = -q_cutoff * q_cutoff +
            2.0f * q_cutoff * q_cutoff * q_cutoff * delta;
          const float correction = 2.0f *
            (-q * q - smooth_derivative);
          value += q - tsne_smooth_q(distance, cutoff_squared, q_cutoff);
          slope[0] += correction * dx;
          slope[1] += correction * dy;
          slope[2] += correction * dz;
        }
      }
    }
  }
}

void compute_gradient_fft_3d_f(const SparseProbabilitiesF& p,
                               const std::vector<float>& y, int n,
                               float exaggeration, int n_threads,
                               TsneFft3dWorkspace& ws,
                               std::vector<float>& grad,
                               int grid_override = 0) {
  const int grid = grid_override > 0 ? grid_override :
    tsne_fft_3d_grid_size(n);
  ws.ensure(grid, n_threads);
  const int size = ws.fft_size;
  const std::size_t total = ws.mass_fft.size();
  std::array<float, 3> lower, upper;
  for (int axis = 0; axis < 3; ++axis) {
    lower[axis] = y[axis];
    upper[axis] = y[axis];
  }
  for (int i = 1; i < n; ++i) {
    const std::size_t base = static_cast<std::size_t>(i) * 3;
    for (int axis = 0; axis < 3; ++axis) {
      lower[axis] = std::min(lower[axis], y[base + axis]);
      upper[axis] = std::max(upper[axis], y[base + axis]);
    }
  }
  float span = 0.0f;
  for (int axis = 0; axis < 3; ++axis) {
    span = std::max(span, upper[axis] - lower[axis]);
  }
  if (!std::isfinite(span)) {
    Rcpp::stop("3D t-SNE coordinates must remain finite.");
  }
  const float half = 0.55f * span + 1.0e-3f;
  const float spacing = 2.0f * half / (grid - 5);
  if (!std::isfinite(spacing) || spacing <= 0.0f) {
    Rcpp::stop("3D t-SNE grid spacing is outside float32 range.");
  }
  const float inv_spacing = 1.0f / spacing;
  const float cutoff = (n < 5000 ? 3.0f : 2.0f) * spacing;
  const float cutoff_squared = cutoff * cutoff;
  const float q_cutoff = 1.0f / (1.0f + cutoff_squared);
  // The omitted third-order Taylor remainder is below 1e-6 here.
  const bool correct_near = cutoff_squared > 0.01f;
  for (int axis = 0; axis < 3; ++axis) {
    lower[axis] = 0.5f * (lower[axis] + upper[axis]) - half -
      2.0f * spacing;
  }
  std::fill(ws.mass.begin(), ws.mass.end(), 0.0f);
  for (int z = 0; z < 7; ++z) {
    for (int yy = 0; yy < 7; ++yy) {
      for (int x = 0; x < 7; ++x) {
        const float dx = (x - 3) * spacing;
        const float dy = (yy - 3) * spacing;
        const float dz = (z - 3) * spacing;
        const float distance = dx * dx + dy * dy + dz * dz;
        ws.self_kernel[(z * 7 + yy) * 7 + x] =
          tsne_smooth_q(distance, cutoff_squared, q_cutoff);
      }
    }
  }
  for (int i = 0; i < n; ++i) {
    int start[3];
    float weight[3][4], derivative[3][4];
    tsne_grid_point_3d(y.data() + static_cast<std::size_t>(i) * 3,
                       lower.data(), inv_spacing, grid, start,
                       weight, derivative);
    for (int z = 0; z < 4; ++z) {
      for (int yy = 0; yy < 4; ++yy) {
        const std::size_t row =
          (static_cast<std::size_t>(start[2] + z) * grid +
           start[1] + yy) * grid + start[0];
        const float yz = weight[2][z] * weight[1][yy];
        for (int x = 0; x < 4; ++x) {
          ws.mass[row + x] += yz * weight[0][x];
        }
      }
    }
  }
  parallel_for(size, n_threads, [&](int begin, int end, int) {
    for (int z = begin; z < end; ++z) {
      for (int yy = 0; yy < size; ++yy) {
        const std::size_t row =
          (static_cast<std::size_t>(z) * size + yy) * size;
        for (int x = 0; x < size; ++x) {
          ws.mass_fft[row + x] =
            z < grid && yy < grid && x < grid ?
            ws.mass[(static_cast<std::size_t>(z) * grid + yy) * grid + x] :
            0.0f;
        }
      }
    }
  });
  parallel_for(size, n_threads, [&](int begin, int end, int) {
    for (int z = begin; z < end; ++z) {
      const int dz = z < grid ? z : z - size;
      const float zz = static_cast<float>(dz) * spacing;
      for (int yy = 0; yy < size; ++yy) {
        const int dy = yy < grid ? yy : yy - size;
        const float y_offset = static_cast<float>(dy) * spacing;
        for (int x = 0; x < size; ++x) {
          const int dx = x < grid ? x : x - size;
          const float xx = static_cast<float>(dx) * spacing;
          const float d2 = xx * xx + y_offset * y_offset + zz * zz;
          ws.kernel_fft[(static_cast<std::size_t>(z) * size + yy) *
                        size + x] =
            tsne_smooth_q(d2, cutoff_squared, q_cutoff);
        }
      }
    }
  });
  const auto fft_started = std::chrono::steady_clock::now();
  tsne_fft_3d(ws.mass_fft, false, ws);
  tsne_fft_3d(ws.kernel_fft, false, ws);
  parallel_for(static_cast<int>(total), n_threads,
               [&](int begin, int end, int) {
    for (int pos = begin; pos < end; ++pos) {
      ws.mass_fft[static_cast<std::size_t>(pos)] *=
        ws.kernel_fft[static_cast<std::size_t>(pos)];
    }
  });
  tsne_fft_3d(ws.mass_fft, true, ws);
  parallel_for(grid, n_threads, [&](int begin, int end, int) {
    for (int z = begin; z < end; ++z) {
      for (int yy = 0; yy < grid; ++yy) {
        for (int x = 0; x < grid; ++x) {
          ws.potential[(static_cast<std::size_t>(z) * grid + yy) *
                       grid + x] = ws.mass_fft[
            (static_cast<std::size_t>(z) * size + yy) * size + x
          ].real();
        }
      }
    }
  });
  const auto fft_finished = std::chrono::steady_clock::now();
  ws.fft_elapsed_sec += std::chrono::duration<double>(
    fft_finished - fft_started).count();
  const int bin_count =
    std::max(1, static_cast<int>(std::ceil(grid * spacing / cutoff)));
  if (correct_near) {
    tsne_fft_3d_bins(y, n, lower, cutoff, bin_count, ws);
  }
  std::fill(ws.partial_sum.begin(), ws.partial_sum.end(), 0.0);
  parallel_for(n, n_threads, [&](int begin, int end, int thread_id) {
    double partial = 0.0;
    for (int i = begin; i < end; ++i) {
      int start[3];
      float weight[3][4], derivative[3][4];
      const std::size_t base = static_cast<std::size_t>(i) * 3;
      tsne_grid_point_3d(y.data() + base, lower.data(),
                         inv_spacing, grid, start, weight, derivative);
      double value = 0.0;
      double slope[3] = {0.0, 0.0, 0.0};
      for (int z = 0; z < 4; ++z) {
        for (int yy = 0; yy < 4; ++yy) {
          const std::size_t row =
            (static_cast<std::size_t>(start[2] + z) * grid +
             start[1] + yy) * grid + start[0];
          for (int x = 0; x < 4; ++x) {
            const double potential = ws.potential[row + x];
            value += potential * weight[0][x] *
              weight[1][yy] * weight[2][z];
            slope[0] += potential * derivative[0][x] *
              weight[1][yy] * weight[2][z];
            slope[1] += potential * weight[0][x] *
              derivative[1][yy] * weight[2][z];
            slope[2] += potential * weight[0][x] *
              weight[1][yy] * derivative[2][z];
          }
        }
      }
      double self_value, self_slope[3];
      tsne_fft_3d_self(weight, derivative, ws.self_kernel,
                       self_value, self_slope);
      value -= self_value;
      double near_value = 0.0, near_slope[3] = {0.0, 0.0, 0.0};
      if (correct_near) {
        tsne_fft_3d_near(y, i, lower, cutoff, cutoff_squared,
                         q_cutoff, bin_count, ws, near_value, near_slope);
      }
      value += near_value;
      partial += value;
      for (int axis = 0; axis < 3; ++axis) {
        grad[base + axis] = static_cast<float>(
          0.5 * ((slope[axis] - self_slope[axis]) * inv_spacing +
                 near_slope[axis])
        );
      }
    }
    ws.partial_sum[static_cast<std::size_t>(thread_id)] = partial;
  });
  ws.sum_q =
    std::accumulate(ws.partial_sum.begin(), ws.partial_sum.end(), 0.0);
  if (!std::isfinite(ws.sum_q) || ws.sum_q <= 0.0) {
    Rcpp::stop("3D t-SNE normalization is not positive and finite.");
  }
  const float inv_sum = static_cast<float>(
    1.0 / std::max(ws.sum_q, static_cast<double>(FLT_MIN))
  );
  parallel_for(static_cast<int>(grad.size()), n_threads,
               [&](int begin, int end, int) {
    for (int pos = begin; pos < end; ++pos) {
      grad[static_cast<std::size_t>(pos)] *= inv_sum;
    }
  });
  ws.correction_elapsed_sec += std::chrono::duration<double>(
    std::chrono::steady_clock::now() - fft_finished).count();
  add_sparse_attractive_gradient_f(p, y, n, 3, exaggeration,
                                   n_threads, grad);
}
