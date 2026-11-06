"""Bounded two-to-four GPU local moves on a disk-backed symmetric CSR graph.

The graph and partition remain in host storage. Each GPU only receives one
bounded row/edge batch. This worker fits one local-moving level for the R
multilevel Louvain or hybrid Leiden controller; it is not cuGraph.
"""

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import time

import numpy as np


KERNEL = r'''
extern "C" __global__ void propose(
    const int* ptr, const int* target, const int* neighbor,
    const float* weight,
    const double* neighbor_volume, const int* current,
    const float* degree, const double* current_volume,
    int* output, int rows, int first, int color, int seed,
    double total, double resolution) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  int old = current[row];
  output[row] = old;
  unsigned int hash = (unsigned int)(first + row) * 1103515245u +
                      (unsigned int)seed * 12345u;
  if ((hash & 7u) != (unsigned int)color || ptr[row] == ptr[row+1])
    return;
  double old_weight = 0.0;
  for (int e = ptr[row]; e < ptr[row+1]; ++e)
    if (neighbor[e] == old && target[e] != first + row)
      old_weight += weight[e];
  double best = old_weight - resolution * degree[row] *
                (current_volume[row] - degree[row]) / total;
  int winner = old;
  for (int e = ptr[row]; e < ptr[row+1]; ++e) {
    int candidate = neighbor[e];
    if (candidate == old) continue;
    bool seen = false;
    for (int prior = ptr[row]; prior < e; ++prior)
      if (neighbor[prior] == candidate) {seen = true; break;}
    if (seen) continue;
    double linked = 0.0;
    for (int other = ptr[row]; other < ptr[row+1]; ++other)
      if (neighbor[other] == candidate) linked += weight[other];
    double score = linked - resolution * degree[row] *
                   neighbor_volume[e] / total;
    if (score > best + 1e-7 ||
        (winner != old && score >= best - 1e-7 &&
         candidate < winner)) {
      best = score;
      winner = candidate;
    }
  }
  output[row] = winner;
}
'''


def validate_graph(paths, vertices, edge_budget):
    offsets = np.memmap(paths[0], dtype="<u8", mode="r")
    if vertices < 2 or vertices >= 2**31 or len(offsets) != vertices + 1:
        raise ValueError("CSR vertex count is invalid")
    if offsets[0] != 0:
        raise ValueError("CSR offsets are invalid")
    for first in range(0, vertices, 65536):
        last = min(vertices, first + 65536)
        degree = np.diff(offsets[first:last + 1])
        if np.any(offsets[first + 1:last + 1] <
                  offsets[first:last]):
            raise ValueError("CSR offsets are invalid")
        if np.any(degree > edge_budget):
            raise ValueError("A CSR row exceeds the GPU edge budget")
    edges = int(offsets[-1])
    if edges < 1 or edges >= 2**63:
        raise ValueError("CSR edge count is invalid")
    if os.path.getsize(paths[1]) != edges * 4 or \
            os.path.getsize(paths[2]) not in (edges * 4, edges * 8):
        raise ValueError("CSR edge files have inconsistent sizes")
    return offsets, edges


def weight_dtype(paths, edges):
    return "<f8" if os.path.getsize(paths[2]) == edges * 8 else "<f4"


def bounded_end(offsets, first, end, chunk_rows, edge_budget):
    last = min(end, first + chunk_rows)
    edge_end = int(offsets[first]) + edge_budget
    last = min(last, int(np.searchsorted(offsets, edge_end,
                                        side="right") - 1))
    if last <= first:
        raise ValueError("A CSR row exceeds the GPU edge budget")
    return last


def graph_degree(paths, offsets, vertices, chunk_rows,
                 edge_budget):
    weights = np.memmap(paths[2], dtype=weight_dtype(
        paths, int(offsets[-1])), mode="r")
    degree = np.zeros(vertices, dtype="<f8")
    first = 0
    while first < vertices:
        last = bounded_end(offsets, first, vertices,
                           chunk_rows, edge_budget)
        begin, end = int(offsets[first]), int(offsets[last])
        values = weights[begin:end]
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError("CSR has invalid weights")
        prefix = np.empty(end - begin + 1, dtype=np.float64)
        prefix[0] = 0
        np.cumsum(values, dtype=np.float64, out=prefix[1:])
        row_offsets = np.asarray(offsets[first:last + 1] - begin,
                                 dtype=np.int64)
        degree[first:last] = np.diff(prefix[row_offsets])
        first = last
    total = float(np.sum(degree, dtype=np.float64))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("CSR graph volume is invalid")
    return degree, total


def modularity(paths, offsets, labels, degree, total,
               resolution, chunk_rows, edge_budget):
    vertices = len(labels)
    target = np.memmap(paths[1], dtype="<u4", mode="r")
    weight = np.memmap(paths[2], dtype=weight_dtype(
        paths, int(offsets[-1])), mode="r")
    volume = np.bincount(labels, weights=degree,
                         minlength=vertices).astype(np.float64)
    internal = 0.0
    first = 0
    while first < vertices:
        last = bounded_end(offsets, first, vertices,
                           chunk_rows, edge_budget)
        begin, end = int(offsets[first]), int(offsets[last])
        ids = target[begin:end]
        if np.any(ids == 0) or np.any(ids > vertices):
            raise ValueError("CSR has out-of-range neighbor IDs")
        counts = np.asarray(np.diff(offsets[first:last + 1]),
                            dtype=np.int64)
        source = np.repeat(np.arange(first, last, dtype=np.int32),
                           counts)
        internal += np.sum(weight[begin:end][
            labels[ids - 1] == labels[source]], dtype=np.float64)
        first = last
    score = internal / total - resolution * np.sum(
        (volume / total)**2, dtype=np.float64)
    return float(score), volume


def worker(device, paths, offsets_path, vertices, labels_path,
           proposals_path, volumes_path, degree_path, first, end,
           chunk_rows, edge_budget, color, seed, total, resolution,
           pool_limit, result_queue):
    try:
        import cupy as cp

        cp.cuda.Device(device).use()
        bus_id = cp.cuda.runtime.deviceGetPCIBusId(device)
        if isinstance(bus_id, bytes):
            bus_id = bus_id.decode("ascii")
        if pool_limit:
            cp.get_default_memory_pool().set_limit(size=pool_limit)
        kernel = cp.RawKernel(KERNEL, "propose")
        offsets = np.memmap(offsets_path, dtype="<u8", mode="r")
        target = np.memmap(paths[1], dtype="<u4", mode="r")
        weights = np.memmap(paths[2], dtype=weight_dtype(
            paths, int(offsets[-1])), mode="r")
        labels = np.memmap(labels_path, dtype="<i4", mode="r",
                           shape=(vertices,))
        proposals = np.memmap(proposals_path, dtype="<i4", mode="r+",
                              shape=(vertices,))
        volumes = np.memmap(volumes_path, dtype="<f8", mode="r",
                            shape=(vertices,))
        degree = np.memmap(degree_path, dtype="<f8", mode="r",
                           shape=(vertices,))
        processed = 0
        while first < end:
            last = bounded_end(offsets, first, end, chunk_rows,
                               edge_budget)
            begin, finish = int(offsets[first]), int(offsets[last])
            ids = np.asarray(target[begin:finish], dtype=np.int64)
            if np.any(ids == 0) or np.any(ids > vertices):
                raise ValueError("CSR has out-of-range neighbor IDs")
            neighbor = np.asarray(labels[ids - 1], dtype=np.int32)
            neighbor_volume = np.asarray(volumes[neighbor],
                                         dtype=np.float64)
            ptr = np.asarray(offsets[first:last + 1] - begin,
                             dtype=np.int32)
            out = cp.empty(last - first, dtype=cp.int32)
            kernel(((last-first+127)//128,), (128,), (
                cp.asarray(ptr),
                cp.asarray(np.asarray(ids - 1, dtype=np.int32)),
                cp.asarray(neighbor),
                cp.asarray(np.asarray(weights[begin:finish],
                                      dtype=np.float32)),
                cp.asarray(neighbor_volume),
                cp.asarray(labels[first:last]),
                cp.asarray(np.asarray(degree[first:last],
                                      dtype=np.float32)),
                cp.asarray(volumes[labels[first:last]]), out,
                np.int32(last-first), np.int32(first),
                np.int32(color), np.int32(seed),
                np.float64(total), np.float64(resolution)))
            proposals[first:last] = cp.asnumpy(out)
            processed += last - first
            first = last
        proposals.flush()
        cp.cuda.runtime.deviceSynchronize()
        result_queue.put({"device": device, "rows": processed,
                          "pci_bus_id": bus_id,
                          "peak_pool_bytes": cp.get_default_memory_pool()
                          .total_bytes()})
    except Exception as error:
        result_queue.put({"device": device, "error": repr(error)})


def run_phase(paths, vertices, state, devices, chunk_rows,
              edge_budget, color, seed, total, resolution,
              pool_limit=0):
    context = mp.get_context("spawn")
    queue = context.Queue()
    jobs = []
    for device in range(devices):
        first = vertices * device // devices
        end = vertices * (device + 1) // devices
        job = context.Process(target=worker, args=(
            device, paths, paths[0], vertices, state["labels"],
            state["proposals"], state["volumes"],
            state["degree"], first, end, chunk_rows, edge_budget,
            color, seed, total, resolution, pool_limit, queue))
        job.start()
        jobs.append(job)
    reports = []
    deadline = time.monotonic() + 3600
    try:
        while len(reports) < len(jobs):
            try:
                reports.append(queue.get(timeout=2))
            except Empty:
                if any(job.exitcode not in (None, 0) for job in jobs):
                    raise RuntimeError("A GPU worker exited without a report")
                if time.monotonic() > deadline:
                    raise TimeoutError("A GPU phase exceeded one hour")
            if any("error" in report for report in reports):
                raise RuntimeError("GPU phase failed: " + repr(reports))
        for job in jobs:
            job.join(timeout=60)
    finally:
        for job in jobs:
            if job.is_alive():
                job.terminate()
            job.join(timeout=10)
    if (any(job.exitcode != 0 for job in jobs) or
            any("error" in report for report in reports) or
            sum(report["rows"] for report in reports) != vertices):
        raise RuntimeError("GPU phase failed: " + repr(reports))
    if pool_limit and any(report["peak_pool_bytes"] > pool_limit
                          for report in reports):
        raise MemoryError("GPU memory pool exceeded its test limit")
    if len(set(report["pci_bus_id"] for report in reports)) != devices:
        raise RuntimeError("GPU workers did not use distinct devices")
    return reports


def fit(args):
    paths = args.paths
    offsets, edges = validate_graph(paths, args.vertices,
                                     args.edge_budget)
    if args.devices < 2 or args.devices > 4:
        raise ValueError("Two to four GPUs are required")
    work = Path(args.output).parent
    prefix = Path(args.output).name
    state = {name: str(work / (prefix + "." + name + ".part"))
             for name in ("labels", "proposals", "volumes", "degree")}
    if any(Path(path).exists() for path in state.values()):
        raise FileExistsError("Temporary clustering state already exists")
    labels = np.memmap(state["labels"], dtype="<i4", mode="w+",
                       shape=(args.vertices,))
    proposals = np.memmap(state["proposals"], dtype="<i4",
                          mode="w+", shape=(args.vertices,))
    volumes = np.memmap(state["volumes"], dtype="<f8", mode="w+",
                         shape=(args.vertices,))
    degree, total = graph_degree(paths, offsets, args.vertices,
                                 args.chunk_rows, args.edge_budget)
    mapped_degree = np.memmap(state["degree"], dtype="<f8",
                              mode="w+", shape=(args.vertices,))
    mapped_degree[:] = degree
    initial_path = getattr(args, "initial", "")
    if initial_path:
        if os.path.getsize(initial_path) != 4 * args.vertices:
            raise ValueError("Initial partition has the wrong size")
        initial = np.memmap(initial_path, dtype="<u4", mode="r")
        if np.any(initial < 1) or np.any(initial > args.initial_count):
            raise ValueError("Initial partition has invalid labels")
        labels[:] = initial.astype(np.int32) - 1
    else:
        labels[:] = np.arange(args.vertices, dtype=np.int32)
    score, value = modularity(paths, offsets, labels, degree, total,
                              args.resolution, args.chunk_rows,
                              args.edge_budget)
    initial_score = score
    volumes[:] = value
    reports = []
    moves = 0
    for pass_id in range(args.passes):
        accepted = 0
        for color in range(8):
            proposals[:] = labels
            proposals.flush()
            report = run_phase(paths, args.vertices, state,
                               args.devices, args.chunk_rows,
                               args.edge_budget, color, args.seed,
                               total, args.resolution,
                               getattr(args, "gpu_pool_bytes", 0))
            reports.extend(report)
            proposed = int(np.count_nonzero(proposals != labels))
            if not proposed:
                continue
            candidate, value = modularity(paths, offsets, proposals,
                degree, total, args.resolution, args.chunk_rows,
                args.edge_budget)
            if candidate + 1e-9 < score:
                continue
            labels[:] = proposals
            volumes[:] = value
            score = candidate
            accepted += proposed
        moves += accepted
        if not accepted:
            break
    unique, compact = np.unique(labels, return_inverse=True)
    result = np.memmap(args.output + ".part", dtype="<u4",
                       mode="w+", shape=(args.vertices,))
    result[:] = compact.astype(np.uint32) + 1
    result.flush()
    metadata = {"backend": "cuda_streamed_multigpu_local",
                "method": "louvain_local_move", "vertices": args.vertices,
                "directed_edges": edges, "communities": len(unique),
                "modularity": score, "moves": moves,
                "modularity_initial": initial_score,
                "initial_partition": bool(initial_path),
                "gpu_devices_used": sorted(set(r["device"] for r in reports)),
                "gpu_pci_bus_ids": [next(r["pci_bus_id"] for r in reports
                    if r["device"] == device)
                    for device in range(args.devices)],
                "peak_gpu_pool_bytes": {str(device): max(
                    r["peak_pool_bytes"] for r in reports
                    if r["device"] == device)
                    for device in range(args.devices)},
                "edge_budget_per_gpu": args.edge_budget,
                "gpu_pool_limit_bytes": getattr(args,
                    "gpu_pool_bytes", 0),
                "graph_file_bytes": sum(os.path.getsize(p) for p in paths),
                "gpu_batches": len(reports),
                "passes_requested": args.passes}
    Path(args.metadata).write_text(json.dumps(metadata, indent=2))
    for path in state.values():
        os.unlink(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs=3, required=True)
    parser.add_argument("--vertices", type=int, required=True)
    parser.add_argument("--devices", type=int, required=True)
    parser.add_argument("--chunk-rows", type=int, required=True)
    parser.add_argument("--edge-budget", type=int, required=True)
    parser.add_argument("--gpu-pool-bytes", type=int, default=0)
    parser.add_argument("--passes", type=int, required=True)
    parser.add_argument("--resolution", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--initial", default="")
    parser.add_argument("--initial-count", type=int, default=0)
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()
    if min(args.vertices, args.chunk_rows, args.edge_budget,
           args.passes) < 1 or args.seed < 0 or args.resolution <= 0 or \
            args.gpu_pool_bytes < 0 or \
            (bool(args.initial) != (args.initial_count > 0)):
        parser.error("Invalid clustering controls")
    return args


if __name__ == "__main__":
    fit(parse_args())
