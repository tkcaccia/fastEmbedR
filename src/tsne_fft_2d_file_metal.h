/*
 * SPDX-FileCopyrightText: 2026 Stefano Cacciatore
 * SPDX-License-Identifier: MIT
 *
 * File-backed two-dimensional Metal t-SNE using the resident 2D FFT kernels.
 */

struct MetalTsne2dFileWorkspace {
  MetalEmbeddingState& state;
  MetalTsne3dBuffers owned;
  std::uint32_t n, grid, fft, log_fft, grid_cells, fft_cells;
  std::uint32_t block_size = 256u;
  std::uint32_t stats_blocks, sum_blocks;
  id<MTLBuffer> current = nil, next = nil, gains = nil, updates = nil;
  id<MTLBuffer> stats = nil, grid_params = nil, center = nil;
  id<MTLBuffer> mass = nil, mass_x = nil, mass_y = nil;
  id<MTLBuffer> mass_lo = nil, mass_hi = nil;
  id<MTLBuffer> x_lo = nil, x_hi = nil;
  id<MTLBuffer> y_lo = nil, y_hi = nil;
  id<MTLComputePipelineState> clear_fixed = nil;
  id<MTLComputePipelineState> scatter_fixed = nil;
  id<MTLComputePipelineState> convert_fixed = nil;
  id<MTLComputePipelineState> load_fixed = nil;
  id<MTLBuffer> mass_fft = nil, mass_x_fft = nil, mass_y_fft = nil;
  id<MTLBuffer> kernel_q = nil, kernel_q2 = nil;
  id<MTLBuffer> q_grid = nil, q2_grid = nil;
  id<MTLBuffer> xq2_grid = nil, yq2_grid = nil;
  id<MTLBuffer> scratch = nil, twiddles = nil;
  id<MTLBuffer> q_blocks = nil, inv_sum_q = nil;

  MetalTsne2dFileWorkspace(MetalEmbeddingState& device,
                           const std::vector<float>& initial, int rows)
    : state(device), owned(device.device), n(rows),
      grid(metal_tsne_fft_grid_size(rows)), fft(grid * 2u),
      log_fft(log2_power_of_two(fft)),
      grid_cells(grid * grid), fft_cells(fft * fft),
      stats_blocks((n + block_size - 1u) / block_size),
      sum_blocks(stats_blocks) {
    const auto layout_bytes = initial.size() * sizeof(float);
    const auto real_bytes = static_cast<std::size_t>(grid_cells) *
      sizeof(float);
    const auto complex_bytes = static_cast<std::size_t>(fft_cells) *
      sizeof(float) * 2u;
    std::vector<float> ones(initial.size(), 1.0f);
    std::vector<float> zeros(initial.size(), 0.0f);
    current = owned.make(layout_bytes, initial.data());
    next = owned.make(layout_bytes);
    gains = owned.make(layout_bytes, ones.data());
    updates = owned.make(layout_bytes, zeros.data());
    stats = owned.make(static_cast<std::size_t>(stats_blocks) *
                       sizeof(OpenTsneLayoutStats));
    grid_params = owned.make(sizeof(OpenTsneFFTGridParams));
    center = owned.make(sizeof(Center2));
    mass = owned.make(real_bytes);
    mass_x = owned.make(real_bytes);
    mass_y = owned.make(real_bytes);
    mass_lo = owned.make(real_bytes);
    mass_hi = owned.make(real_bytes);
    x_lo = owned.make(real_bytes);
    x_hi = owned.make(real_bytes);
    y_lo = owned.make(real_bytes);
    y_hi = owned.make(real_bytes);
    mass_fft = owned.make(complex_bytes);
    mass_x_fft = owned.make(complex_bytes);
    mass_y_fft = owned.make(complex_bytes);
    kernel_q = owned.make(complex_bytes);
    kernel_q2 = owned.make(complex_bytes);
    q_grid = owned.make(complex_bytes);
    q2_grid = owned.make(complex_bytes);
    xq2_grid = owned.make(complex_bytes);
    yq2_grid = owned.make(complex_bytes);
    scratch = owned.make(complex_bytes);
    const auto table = make_fft_twiddles(fft, log_fft);
    twiddles = owned.make(table.size() * sizeof(float), table.data());
    q_blocks = owned.make(static_cast<std::size_t>(sum_blocks) *
                          sizeof(float));
    inv_sum_q = owned.make(sizeof(float));
    clear_fixed = make_pipeline(state, "opentsne_fft_clear_fixed");
    scatter_fixed = make_pipeline(state, "opentsne_fft_scatter_fixed");
    convert_fixed = make_pipeline(state, "opentsne_fft_convert_fixed");
    load_fixed = make_pipeline(state, "opentsne_fft_load_fixed");
  }

  ~MetalTsne2dFileWorkspace() {
    [load_fixed release];
    [convert_fixed release];
    [scatter_fixed release];
    [clear_fixed release];
  }

  void encode_prepare(id<MTLCommandBuffer> command) {
    const auto scalar = [](const auto& value) {
      return std::make_pair(static_cast<const void*>(&value),
                            sizeof(value));
    };
    metal_tsne_3d_encode(command,
      state.opentsne_fft_layout_stats_blocks_pipeline,
      stats_blocks, {current, stats},
      {scalar(n), scalar(block_size)});
    metal_tsne_3d_encode(command,
      state.opentsne_fft_finalize_layout_stats_pipeline, 1u,
      {stats, grid_params, center},
      {scalar(stats_blocks), scalar(n), scalar(grid), scalar(fft)});
    metal_tsne_3d_encode(command, clear_fixed, grid_cells,
      {mass_lo, mass_hi, x_lo, x_hi, y_lo, y_hi, grid_params});
    metal_tsne_3d_encode(command, scatter_fixed, n,
      {current, mass_lo, mass_hi, x_lo, x_hi, y_lo, y_hi,
       grid_params});
    metal_tsne_3d_encode(command, convert_fixed, grid_cells,
      {mass_lo, mass_hi, x_lo, x_hi, y_lo, y_hi,
       mass, mass_x, mass_y, grid_params});
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:load_fixed];
    NSUInteger slot = 0u;
    for (id<MTLBuffer> buffer : {mass, mass_x, mass_y, mass_fft,
         mass_x_fft, mass_y_fft, kernel_q, kernel_q2, grid_params}) {
      [encoder setBuffer:buffer offset:0 atIndex:slot++];
    }
    [encoder dispatchThreads:MTLSizeMake(fft, fft, 1u)
       threadsPerThreadgroup:metal_threadgroup_2d(
         load_fixed)];
    [encoder endEncoding];
  }

  void encode_fft(id<MTLCommandBuffer> command) {
    for (id<MTLBuffer> buffer : {mass_fft, mass_x_fft, mass_y_fft,
                                 kernel_q, kernel_q2}) {
      encode_fft_2d_metal(state, command, buffer, scratch, twiddles,
                          fft, log_fft, false);
    }
    encode_fft_convolution_metal(state, command, mass_fft, kernel_q,
      q_grid, scratch, twiddles, fft, log_fft, fft_cells);
    encode_fft_convolution_metal(state, command, mass_fft, kernel_q2,
      q2_grid, scratch, twiddles, fft, log_fft, fft_cells);
    encode_fft_convolution_metal(state, command, mass_x_fft, kernel_q2,
      xq2_grid, scratch, twiddles, fft, log_fft, fft_cells);
    encode_fft_convolution_metal(state, command, mass_y_fft, kernel_q2,
      yq2_grid, scratch, twiddles, fft, log_fft, fft_cells);
  }

  void encode_sum_q(id<MTLCommandBuffer> command) {
    const auto scalar = [](const auto& value) {
      return std::make_pair(static_cast<const void*>(&value),
                            sizeof(value));
    };
    metal_tsne_3d_encode(command,
      state.opentsne_fft_sum_q_blocks_pipeline, sum_blocks,
      {current, q_grid, q_blocks, grid_params},
      {scalar(block_size)});
    metal_tsne_3d_encode(command,
      state.opentsne_fft_finalize_sum_q_pipeline, 1u,
      {q_blocks, inv_sum_q}, {scalar(sum_blocks), scalar(n)});
  }

  void encode_update(id<MTLCommandBuffer> command,
                     id<MTLBuffer> row_ptr, id<MTLBuffer> col,
                     id<MTLBuffer> prob, std::uint32_t first,
                     std::uint32_t count, std::uint64_t edge_base,
                     const OpenTsneMetalParams& params) {
    const std::uint32_t index_base = 1u;
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:state.opentsne_fft_epoch_pipeline];
    NSUInteger slot = 0u;
    for (id<MTLBuffer> buffer : {row_ptr, col, prob, current, next,
         gains, updates, q2_grid, xq2_grid, yq2_grid}) {
      [encoder setBuffer:buffer offset:0 atIndex:slot++];
    }
    [encoder setBytes:&params length:sizeof(params) atIndex:10];
    [encoder setBuffer:grid_params offset:0 atIndex:11];
    [encoder setBuffer:inv_sum_q offset:0 atIndex:12];
    [encoder setBytes:&first length:sizeof(first) atIndex:18];
    [encoder setBytes:&edge_base length:sizeof(edge_base) atIndex:19];
    [encoder setBytes:&index_base length:sizeof(index_base) atIndex:20];
    [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
       threadsPerThreadgroup:MTLSizeMake(
         bounded_threads(state.opentsne_fft_epoch_pipeline), 1u, 1u)];
    [encoder endEncoding];
  }

  void encode_center(id<MTLCommandBuffer> command) {
    const auto scalar = [](const auto& value) {
      return std::make_pair(static_cast<const void*>(&value),
                            sizeof(value));
    };
    metal_tsne_3d_encode(command,
      state.opentsne_fft_layout_stats_blocks_pipeline,
      stats_blocks, {next, stats},
      {scalar(n), scalar(block_size)});
    metal_tsne_3d_encode(command,
      state.opentsne_fft_finalize_layout_stats_pipeline, 1u,
      {stats, grid_params, center},
      {scalar(stats_blocks), scalar(n), scalar(grid), scalar(fft)});
    metal_tsne_3d_encode(command, state.opentsne_center_pipeline,
      n, {next, center}, {scalar(n)});
  }
};

struct MetalTsne2dFileTiles {
  std::vector<std::pair<std::uint32_t, std::uint32_t>> ranges;
  id<MTLBuffer> offsets = nil, indices = nil, weights = nil;

  MetalTsne2dFileTiles(MetalTsne3dBuffers& owned,
                       const MetalTsne3dStream& graph,
                       std::uint32_t n) {
    constexpr std::uint32_t row_capacity = 262144u;
    const std::size_t capacity = std::min<std::size_t>(
      graph.edge_capacity, graph.offsets[n]);
    offsets = owned.make((row_capacity + 1u) * sizeof(std::uint64_t));
    indices = owned.make(std::max<std::size_t>(1u, capacity) *
                         sizeof(std::uint32_t));
    weights = owned.make(std::max<std::size_t>(1u, capacity) *
                         sizeof(float));
    for (std::uint32_t begin = 0u; begin < n;) {
      std::uint32_t end = begin;
      const std::uint64_t base = graph.offsets[begin];
      while (end < n && end - begin < row_capacity &&
             graph.offsets[end + 1u] - base <= capacity) ++end;
      if (end == begin) Rcpp::stop(
        "Metal t-SNE graph row exceeds `edge_chunk_capacity`.");
      ranges.emplace_back(begin, end);
      begin = end;
    }
  }

  std::uint64_t load(const MetalTsne3dStream& graph,
                     std::uint32_t begin, std::uint32_t end) {
    const std::uint64_t base = graph.offsets[begin];
    const std::size_t count = graph.offsets[end] - base;
    std::memcpy([offsets contents], graph.offsets + begin,
                (static_cast<std::size_t>(end - begin) + 1u) *
                  sizeof(std::uint64_t));
    if (count > 0u) {
      std::memcpy([indices contents], graph.indices + base,
                  count * sizeof(std::uint32_t));
      std::memcpy([weights contents], graph.weights + base,
                  count * sizeof(float));
    }
    return base;
  }
};

void metal_tsne_2d_wait(id<MTLCommandBuffer> command,
                        const char* stage) {
  [command commit];
  [command waitUntilCompleted];
  if (command.status == MTLCommandBufferStatusError) Rcpp::stop(
    "Metal 2D t-SNE %s failed: %s", stage,
    ns_error_message(command.error).c_str());
}

List run_metal_tsne_2d_file(
    MetalEmbeddingState& state, const std::vector<float>& initial,
    int n, int early_iter, int normal_iter,
    double early_exaggeration, double exaggeration,
    double learning_rate, bool learning_rate_auto,
    double initial_momentum, double final_momentum,
    double min_gain, double max_step_norm,
    id<MTLBuffer> row_ptr, id<MTLBuffer> col,
    id<MTLBuffer> prob, const MetalTsne3dStream* stream,
    const std::string& output_path, int start_iter,
    const std::string& state_path, int checkpoint_every,
    SEXP checkpoint_callback) {
  MetalTsne2dFileWorkspace work(state, initial, n);
  const auto layout_bytes = initial.size() * sizeof(float);
  if (start_iter > 0) {
    if (!std::filesystem::exists(state_path) ||
        std::filesystem::file_size(state_path) != 3 * layout_bytes) {
      Rcpp::stop("Metal 2D t-SNE checkpoint has invalid size.");
    }
    std::ifstream saved(state_path, std::ios::binary);
    for (id<MTLBuffer> buffer : {work.current, work.updates,
                                  work.gains}) {
      saved.read(static_cast<char*>([buffer contents]),
                 static_cast<std::streamsize>(layout_bytes));
    }
    if (!saved) Rcpp::stop("Cannot read Metal 2D t-SNE checkpoint.");
  }
  std::unique_ptr<MetalTsne2dFileTiles> tiles;
  if (stream != nullptr) tiles.reset(new MetalTsne2dFileTiles(
    work.owned, *stream, static_cast<std::uint32_t>(n)));
  const auto started = std::chrono::steady_clock::now();
  for (int iter = start_iter; iter < early_iter + normal_iter; ++iter) {
    id<MTLCommandBuffer> command = [state.queue commandBuffer];
    work.encode_prepare(command);
    work.encode_fft(command);
    work.encode_sum_q(command);
    const bool early = iter < early_iter;
    const double phase_exag = early ? early_exaggeration : exaggeration;
    const OpenTsneMetalParams params = {
      static_cast<std::uint32_t>(n), 5489u,
      static_cast<float>(learning_rate_auto ? n / phase_exag :
                         learning_rate),
      static_cast<float>(phase_exag),
      static_cast<float>(early ? initial_momentum : final_momentum),
      static_cast<float>(min_gain),
      static_cast<float>(std::isfinite(max_step_norm) ?
                         max_step_norm : FLT_MAX), 1.0f
    };
    if (stream == nullptr) {
      work.encode_update(command, row_ptr, col, prob, 0u,
                         n, 0u, params);
    } else {
      metal_tsne_2d_wait(command, "FFT repulsion");
      for (const auto& range : tiles->ranges) {
        const std::uint64_t base = tiles->load(
          *stream, range.first, range.second);
        id<MTLCommandBuffer> attraction = [state.queue commandBuffer];
        work.encode_update(attraction, tiles->offsets,
          tiles->indices, tiles->weights, range.first,
          range.second - range.first, base, params);
        metal_tsne_2d_wait(attraction, "streamed attraction");
      }
      command = [state.queue commandBuffer];
    }
    work.encode_center(command);
    metal_tsne_2d_wait(command, "optimizer update");
    std::swap(work.current, work.next);
    if (checkpoint_every > 0 &&
        ((iter + 1) % checkpoint_every == 0 ||
         iter + 1 == early_iter + normal_iter)) {
      const auto snapshot = output_path + ".iter_" +
        std::to_string(iter + 1) + ".state.f32";
      metal_tsne_3d_write_buffers(snapshot,
        {work.current, work.updates, work.gains}, layout_bytes);
      const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
      Rcpp::Function callback(checkpoint_callback);
      callback(iter + 1, snapshot, elapsed);
    }
    if ((iter & 7) == 0) Rcpp::checkUserInterrupt();
  }
  const float* final = static_cast<const float*>(
    [work.current contents]);
  if (!std::all_of(final, final + initial.size(),
                   [](float value) { return std::isfinite(value); })) {
    Rcpp::stop("Metal 2D t-SNE produced nonfinite coordinates.");
  }
  metal_tsne_3d_write_buffers(output_path, {work.current},
                              layout_bytes);
  const double elapsed = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - started).count();
  return List::create(
    Rcpp::Named("backend_used") = stream == nullptr ?
      "native_metal_fft_graph_mmap" :
      "native_metal_fft_graph_stream",
    Rcpp::Named("repulsion") = "fft_grid_2d_metal",
    Rcpp::Named("fft_grid_size") = static_cast<int>(work.grid),
    Rcpp::Named("iterations") = early_iter + normal_iter,
    Rcpp::Named("resumed_from_iteration") = start_iter,
    Rcpp::Named("n_threads") = 1,
    Rcpp::Named("elapsed_seconds") = elapsed);
}
