// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT
// CPU graph options; included inside graph_clustering.cpp's private namespace.

std::vector<int> cluster_partition(const Graph& graph,
    const std::string& method, double resolution, int iterations,
    int runs, std::uint64_t seed) {
  std::vector<int> best(static_cast<std::size_t>(graph.n));
  std::iota(best.begin(), best.end(), 0);
  double quality = modularity_score(graph, best, resolution);
  for (int run = 0; run < runs; ++run) {
    Rcpp::checkUserInterrupt();
    const std::uint64_t run_seed = seed +
      static_cast<std::uint64_t>(run) * 0x9e3779b97f4a7c15ULL;
    std::vector<int> labels = method == "louvain" ?
      louvain_partition(graph, iterations, resolution, run_seed) :
      leiden_partition(graph, iterations, iterations, resolution, run_seed);
    const double score = modularity_score(graph, labels, resolution);
    if (run == 0 || score > quality + 1e-12) {
      best = std::move(labels);
      quality = score;
    }
  }
  return best;
}

Rcpp::List graph_controls(SEXP value, const std::set<std::string>& allowed) {
  if (TYPEOF(value) != VECSXP) Rcpp::stop("Graph options must be lists.");
  Rcpp::List controls(value);
  if (controls.size() == 0) return controls;
  if (Rf_isNull(Rf_getAttrib(value, R_NamesSymbol))) {
    Rcpp::stop("Graph option controls must be named.");
  }
  Rcpp::CharacterVector names = controls.names();
  if (names.size() != controls.size()) {
    Rcpp::stop("Graph option controls must be named.");
  }
  std::set<std::string> seen;
  for (R_xlen_t i = 0; i < names.size(); ++i) {
    if (names[i] == NA_STRING) Rcpp::stop("Missing graph control name.");
    const std::string name = Rcpp::as<std::string>(names[i]);
    if (!allowed.count(name) || !seen.insert(name).second) {
      Rcpp::stop("Unknown or duplicated graph control: " + name);
    }
  }
  return controls;
}

double graph_control_number(const Rcpp::List& controls, const char* name,
    double default_value, bool integer = false) {
  if (!controls.containsElementNamed(name)) return default_value;
  SEXP value = controls[name];
  if (!Rf_isNumeric(value) || Rf_xlength(value) != 1) {
    Rcpp::stop(std::string(name) + " must be one positive number.");
  }
  const double number = Rcpp::as<double>(value);
  if (!std::isfinite(number) || number <= 0 ||
      (integer && (number != std::floor(number) || number > INT_MAX))) {
    Rcpp::stop(std::string(name) + " must be positive and finite" +
      (integer ? " and integer." : "."));
  }
  return number;
}

Graph consensus_graph(const Graph& original,
    const Rcpp::NumericVector& resolutions, const Rcpp::NumericVector& seeds,
    const std::string& method, int iterations, Rcpp::List& metadata) {
  std::vector<double> agreement(original.edges.size(), 0.0);
  for (double resolution : resolutions) for (double seed : seeds) {
    const auto labels = cluster_partition(original, method, resolution,
      iterations, 1, static_cast<std::uint64_t>(
        static_cast<std::int64_t>(seed)));
    for (std::size_t i = 0; i < original.edges.size(); ++i) {
      const Edge& edge = original.edges[i];
      agreement[i] += labels[edge.u] == labels[edge.v];
    }
  }
  const double count = static_cast<double>(resolutions.size()) * seeds.size();
  std::vector<Edge> edges;
  for (std::size_t i = 0; i < original.edges.size(); ++i) {
    Edge edge = original.edges[i];
    const double fraction = agreement[i] / count;
    edge.weight *= fraction * fraction;
    if (edge.weight > 0) edges.push_back(edge);
  }
  metadata = Rcpp::List::create(Rcpp::Named("method") = method,
    Rcpp::Named("resolutions") = resolutions, Rcpp::Named("seeds") = seeds,
    Rcpp::Named("partitions") = count, Rcpp::Named("agreement_power") = 2,
    Rcpp::Named("implementation") = "native_sparse_edge_consensus",
    Rcpp::Named("weighted_edges") = static_cast<double>(edges.size()));
  return graph_from_edges(original.n, std::move(edges));
}

// Exact double-precision 2D search, separate from the float32 data KNN route.
class LayoutRadii {
  const Rcpp::NumericMatrix& xy;
  std::vector<int> ids;

  void build(int begin, int end, int axis) {
    if (begin == end) return;
    const int mid = begin + (end - begin) / 2;
    std::nth_element(ids.begin() + begin, ids.begin() + mid,
      ids.begin() + end, [&](int a, int b) {
        return xy(a, axis) < xy(b, axis) ||
          (xy(a, axis) == xy(b, axis) && a < b);
      });
    build(begin, mid, 1 - axis);
    build(mid + 1, end, 1 - axis);
  }

  void search(int u, int begin, int end, int axis, int k,
      std::priority_queue<double>& nearest) const {
    if (begin == end) return;
    const int mid = begin + (end - begin) / 2;
    const int v = ids[mid];
    const double dx = xy(u, 0) - xy(v, 0);
    const double dy = xy(u, 1) - xy(v, 1);
    const double distance = dx * dx + dy * dy;
    if (v != u && (static_cast<int>(nearest.size()) < k ||
        distance < nearest.top())) {
      nearest.push(distance);
      if (static_cast<int>(nearest.size()) > k) nearest.pop();
    }
    const double delta = xy(u, axis) - xy(v, axis);
    if (delta < 0) search(u, begin, mid, 1 - axis, k, nearest);
    else search(u, mid + 1, end, 1 - axis, k, nearest);
    if (static_cast<int>(nearest.size()) < k ||
        delta * delta <= nearest.top()) {
      if (delta < 0) search(u, mid + 1, end, 1 - axis, k, nearest);
      else search(u, begin, mid, 1 - axis, k, nearest);
    }
  }

 public:
  explicit LayoutRadii(const Rcpp::NumericMatrix& layout) : xy(layout),
      ids(static_cast<std::size_t>(layout.nrow())) {
    std::iota(ids.begin(), ids.end(), 0);
    build(0, xy.nrow(), 0);
  }

  std::vector<double> radii(int k) const {
    std::vector<double> radius(ids.size());
    for (int u = 0; u < xy.nrow(); ++u) {
      if (u % 1024 == 0) Rcpp::checkUserInterrupt();
      std::priority_queue<double> nearest;
      search(u, 0, xy.nrow(), 0, k, nearest);
      radius[u] = std::sqrt(nearest.top());
      if (!(radius[u] > 0) || !std::isfinite(radius[u])) {
        Rcpp::stop("Local projection radii must be positive and finite.");
      }
    }
    return radius;
  }
};

Graph projection_graph(const Graph& original, SEXP option,
    Rcpp::List& metadata) {
  const Rcpp::List controls = graph_controls(option,
    {"layout", "k", "stretch", "cutoff", "layout_input_type"});
  if (!controls.containsElementNamed("layout")) {
    Rcpp::stop("projection requires an n-by-2 layout.");
  }
  const Rcpp::NumericMatrix xy =
    Rcpp::as<Rcpp::NumericMatrix>(controls["layout"]);
  if (xy.nrow() != original.n || xy.ncol() != 2) {
    Rcpp::stop("projection$layout must be an n-by-2 matrix.");
  }
  for (double value : xy) {
    if (!std::isfinite(value)) Rcpp::stop("Projection must be finite.");
  }
  const int k = graph_control_number(controls, "k", 30, true);
  const double stretch = graph_control_number(controls, "stretch", 4);
  const double cutoff = graph_control_number(controls, "cutoff", 0.001);
  if (k >= original.n || cutoff >= 1) {
    Rcpp::stop("Projection k must be < n and cutoff must be < 1.");
  }
  const auto radius = LayoutRadii(xy).radii(k);
  std::vector<Edge> edges;
  for (Edge edge : original.edges) {
    const double dx = xy(edge.u, 0) - xy(edge.v, 0);
    const double dy = xy(edge.u, 1) - xy(edge.v, 1);
    const double ratio = std::sqrt(dx * dx + dy * dy) /
      std::sqrt(radius[edge.u] * radius[edge.v]);
    if (!std::isfinite(ratio)) Rcpp::stop("Rescale overflowing layout.");
    const double excess = std::max(ratio / stretch - 1.0, 0.0);
    const double penalty = std::exp(-excess * excess);
    if (penalty >= cutoff) {
      edge.weight *= penalty;
      edges.push_back(edge);
    }
  }
  metadata = Rcpp::List::create(Rcpp::Named("k") = k,
    Rcpp::Named("stretch") = stretch, Rcpp::Named("cutoff") = cutoff,
    Rcpp::Named("distance_precision") = "double",
    Rcpp::Named("layout_input_type") =
      (controls.containsElementNamed("layout_input_type") ?
        Rcpp::as<std::string>(controls["layout_input_type"]) : "double"),
    Rcpp::Named("implementation") = "native_exact_2d_edge_penalty");
  return graph_from_edges(original.n, std::move(edges));
}

std::vector<int> resolve_graph_groups(const std::vector<int>& raw,
    SEXP option, Rcpp::List& metadata, Rcpp::IntegerVector& reassigned,
    Rcpp::IntegerVector& unassigned) {
  const Rcpp::List controls = graph_controls(option,
    {"indices", "distances", "min_size"});
  if (!controls.containsElementNamed("indices") ||
      !controls.containsElementNamed("distances")) {
    Rcpp::stop("Small groups require KNN indices and distances.");
  }
  const Rcpp::IntegerMatrix ids = controls["indices"];
  const Rcpp::NumericMatrix distances = controls["distances"];
  const int n = raw.size();
  const int min_size = graph_control_number(controls, "min_size", 20, true);
  if (ids.nrow() != n || ids.ncol() < 1 ||
      distances.nrow() != n || distances.ncol() != ids.ncol()) {
    Rcpp::stop("Small-group KNN dimensions must match graph vertices.");
  }
  const bool one_based = std::find(ids.begin(), ids.end(), 0) == ids.end();
  const int offset = one_based ? 1 : 0;
  for (R_xlen_t i = 0; i < ids.size(); ++i) {
    const int v = ids[i];
    if (v == NA_INTEGER || v < offset || v >= n + offset ||
        !std::isfinite(distances[i]) || distances[i] < 0) {
      Rcpp::stop("Invalid small-group KNN indices or distances.");
    }
  }
  std::vector<int> counts(static_cast<std::size_t>(community_count(raw)), 0);
  for (int label : raw) ++counts[label];
  std::vector<int> labels = raw;
  std::vector<int> moved, unknown;
  bool has_core = false;
  for (int count : counts) has_core = has_core || count >= min_size;
  if (!has_core) Rcpp::stop("No core community: reduce min_size.");
  for (int u = 0; u < n; ++u) {
    if (counts[raw[u]] >= min_size) continue;
    int best = -1;
    double distance = std::numeric_limits<double>::infinity();
    for (int col = 0; col < ids.ncol(); ++col) {
      const int v = ids(u, col) - offset;
      if (counts[raw[v]] >= min_size && distances(u, col) < distance) {
        best = v;
        distance = distances(u, col);
      }
    }
    labels[u] = best < 0 ? -1 : raw[best];
    if (best < 0) unknown.push_back(u + 1);
    else moved.push_back(u + 1);
  }
  std::fill(counts.begin(), counts.end(), 0);
  for (int label : labels) if (label >= 0) ++counts[label];
  std::vector<int> order;
  for (std::size_t i = 0; i < counts.size(); ++i) {
    if (counts[i]) order.push_back(i);
  }
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return counts[a] > counts[b];
  });
  std::vector<int> compact(counts.size(), -1);
  for (std::size_t i = 0; i < order.size(); ++i) compact[order[i]] = i;
  for (int& label : labels) if (label >= 0) label = compact[label];
  reassigned = Rcpp::wrap(moved);
  unassigned = Rcpp::wrap(unknown);
  metadata = Rcpp::List::create(Rcpp::Named("min_size") = min_size,
    Rcpp::Named("k") = ids.ncol(),
    Rcpp::Named("reassigned_tiles") = moved.size(),
    Rcpp::Named("unassigned_tiles") = unknown.size(),
    Rcpp::Named("implementation") = "native_nearest_core_assignment");
  return labels;
}

Rcpp::IntegerVector graph_membership_r(const std::vector<int>& labels) {
  Rcpp::IntegerVector result(labels.size());
  for (std::size_t i = 0; i < labels.size(); ++i) result[i] = labels[i] + 1;
  return result;
}

Rcpp::List graph_edges_r(const Graph& graph) {
  Rcpp::IntegerVector from(graph.edges.size()), to(graph.edges.size());
  Rcpp::NumericVector weight(graph.edges.size());
  for (std::size_t i = 0; i < graph.edges.size(); ++i) {
    from[i] = graph.edges[i].u + 1;
    to[i] = graph.edges[i].v + 1;
    weight[i] = graph.edges[i].weight;
  }
  return Rcpp::List::create(Rcpp::Named("from") = from,
    Rcpp::Named("to") = to, Rcpp::Named("weight") = weight,
    Rcpp::Named("n_vertices") = graph.n,
    Rcpp::Named("n_edges") = static_cast<double>(graph.edges.size()));
}
