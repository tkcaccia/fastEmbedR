// Only incoming records are staged in RAM. Each owner regenerates its
// outgoing records from the supplied neighbors for the current CSR tile.
constexpr int wide_tile_rows = 1 << 18;
constexpr WideRecordCount wide_target_records = 8LL << 20;

struct WideHostTile {
  std::unique_ptr<WideEdgeKey[]> keys;
  std::unique_ptr<float[]> weights;
  WideRecordCount size;

  WideRecordCount lower(int row) const {
    return std::lower_bound(keys.get(), keys.get() + size,
      static_cast<WideEdgeKey>(row) << 32) - keys.get();
  }
};

using WideTiles = std::vector<WideHostTile>;
using WideTileRanks = std::vector<WideTiles>;

WideTiles stage_wide_tiles(const WideNeighborSource& source, int first,
    int rows, int device, double limit) {
  const auto batch = std::min(rows, wide_tile_rows);
  WideBuildRecords records(device,
    static_cast<WideRecordCount>(batch) * source.k);
  records.allocate(0, limit);
  WideTiles staged;
  staged.reserve((rows - 1) / batch + 1);
  for (int offset = 0; offset < rows;) {
    const int active = std::min(batch, rows - offset);
    records.items = static_cast<WideRecordCount>(active) * source.k;
    pack_wide_records(records, source, first + offset, active, true, limit);
    records.sort(0, limit);
    auto keys = std::unique_ptr<WideEdgeKey[]>(new WideEdgeKey[records.items]);
    wide_build_check(cudaMemcpy(keys.get(), records.keys.Current(),
      records.items * sizeof(WideEdgeKey), cudaMemcpyDeviceToHost),
      "wide graph stage keys to host");
    const auto last = std::lower_bound(keys.get(), keys.get() + records.items,
      ULLONG_MAX) - keys.get();
    auto weights = std::unique_ptr<float[]>(new float[last]);
    if (last) {
      wide_build_check(cudaMemcpy(weights.get(), records.weights.Current(),
        last * sizeof(float), cudaMemcpyDeviceToHost),
        "wide graph stage weights to host");
    }
    staged.push_back({std::move(keys), std::move(weights), last});
    offset += active;
  }
  return staged;
}


WideRecordCount wide_tile_count(const WideTileRanks& staged,
    int first, int end) {
  WideRecordCount count = 0;
  for (const auto& source : staged) {
    for (const auto& tile : source) count += tile.lower(end) - tile.lower(first);
  }
  return count;
}

void ingest_wide_tiles(WideBuildRecords& records, const WideTileRanks& staged,
    int first, int end, int owner, WideRecordCount output, double& routed) {
  for (std::size_t rank = 0; rank < staged.size(); ++rank) {
    for (const auto& tile : staged[rank]) {
      const auto begin = tile.lower(first);
      const auto length = tile.lower(end) - begin;
      if (!length) continue;
      wide_build_check(cudaMemcpy(records.keys.Current() + output,
        tile.keys.get() + begin, length * sizeof(WideEdgeKey),
        cudaMemcpyHostToDevice), "wide graph owner keys");
      wide_build_check(cudaMemcpy(records.weights.Current() + output,
        tile.weights.get() + begin, length * sizeof(float),
        cudaMemcpyHostToDevice), "wide graph owner weights");
      if (rank != static_cast<std::size_t>(owner)) routed += length;
      output += length;
    }
  }
  if (output != records.items) {
    throw std::runtime_error("CUDA fuzzy graph staged record count differs.");
  }
}

float build_staged_owner(WideDiffusionShard& shard,
    const WideNeighborSource& source, const WideTileRanks& staged, int owner,
    double limit, double& routed) {
  const int end = shard.first + shard.rows;
  const auto records_in = [&](int first, int rows) {
    return static_cast<WideRecordCount>(rows) * source.k +
      wide_tile_count(staged, first, first + rows);
  };
  const auto raw = records_in(shard.first, shard.rows);
  shard.edges = raw;
  shard.graph_bytes = wide_graph_bytes(shard.rows, shard.edges);
  wide_build_require(0, shard.graph_bytes, limit);
  if (allocate_wide_graph(shard)) {
    throw std::runtime_error(embedding_last_error);
  }
  shard.edges = 0;
  std::size_t free_bytes = 0, total_bytes = 0;
  wide_build_check(cudaMemGetInfo(&free_bytes, &total_bytes),
    "wide graph owner workspace budget");
  const double available = std::min(
    std::max(0.0, limit - shard.graph_bytes), 0.9 * free_bytes);
  routed = 0;
  float maximum = 0;
  for (int first = shard.first; first < end;) {
    int rows = std::min(wide_tile_rows, end - first);
    auto count = records_in(first, rows);
    const auto work_bytes = [&](int active, WideRecordCount items) {
      const auto upload = static_cast<std::size_t>(
        wide_knn_upload_rows(active, source.k)) * source.k;
      return wide_build_record_bytes(items) +
        2 * align_bytes(upload * sizeof(int));
    };
    while (rows > 1 && (count > wide_target_records ||
        work_bytes(rows, count) > 0.9 * available)) {
      rows /= 2;
      count = records_in(first, rows);
    }
    const auto outgoing = static_cast<WideRecordCount>(rows) * source.k;
    WideBuildRecords records(shard.device, count);
    records.allocate(shard.graph_bytes, limit);
    pack_wide_records(records, source, first, rows, false,
      limit - shard.graph_bytes);
    ingest_wide_tiles(records, staged, first, first + rows, owner, outgoing,
      routed);
    records.sort(shard.graph_bytes, limit);
    const auto weight = wide_build_dispatch(count, [&](auto items) {
      return compact_wide_records_typed(shard, records, items, limit,
        first, rows);
    });
    maximum = std::max(maximum, weight);
    first += rows;
  }
  return maximum;
}
