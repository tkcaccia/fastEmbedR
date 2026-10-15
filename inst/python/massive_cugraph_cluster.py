"""Optional cuGraph worker for one distributed full-graph community fit."""

import argparse
import csv
import json
import os
import re
import subprocess
import threading
import time
from functools import partial
from pathlib import Path

import numpy as np


def csr_partition(bounds, paths):
    import cudf

    first, last = bounds
    offsets = np.memmap(paths[0], dtype="<u8", mode="r")
    begin, end = int(offsets[first]), int(offsets[last])
    count = end - begin
    if count == 0:
        return cudf.DataFrame({
            "src": np.empty(0, dtype=np.int32),
            "dst": np.empty(0, dtype=np.int32),
            "weight": np.empty(0, dtype=np.float32)})
    degrees = np.diff(np.asarray(offsets[first:last + 1],
                                 dtype=np.int64))
    if np.any(degrees < 0):
        raise ValueError("CSR offsets decrease within a partition")
    source = np.repeat(np.arange(first, last, dtype=np.int32),
                       degrees)
    target = np.memmap(paths[1], dtype="<u4", mode="r",
                       offset=begin * 4, shape=(count,))
    weights = np.memmap(paths[2], dtype="<f4", mode="r",
                        offset=begin * 4, shape=(count,))
    if np.any(target == 0) or np.any(target > len(offsets) - 1):
        raise ValueError("CSR contains an out-of-range neighbor ID")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("CSR contains invalid edge weights")
    keep = source < target - 1
    return cudf.DataFrame({
        "src": source[keep],
        "dst": (target[keep] - 1).astype(np.int32),
        "weight": np.asarray(weights[keep], dtype=np.float32)})


def verify_graph(paths, vertices, chunk_rows):
    offsets = np.memmap(paths[0], dtype="<u8", mode="r")
    if len(offsets) != vertices + 1 or offsets[0] != 0:
        raise ValueError("CSR offsets do not cover every vertex")
    for first in range(0, vertices, chunk_rows):
        last = min(first + chunk_rows, vertices)
        if np.any(offsets[first + 1:last + 1] <
                  offsets[first:last]):
            raise ValueError("CSR offsets are not monotone")
    if int(offsets[-1]) * 4 != os.path.getsize(paths[1]) or \
            os.path.getsize(paths[1]) != os.path.getsize(paths[2]):
        raise ValueError("CSR edge files have inconsistent sizes")
    if offsets[-1] == 0:
        raise ValueError("cuGraph needs at least one graph edge")


def normalized_pci_bus_id(value):
    if isinstance(value, bytes):
        value = value.decode("ascii")
    match = re.fullmatch(
        r"([0-9a-f]{4,8}):([0-9a-f]{2}):([0-9a-f]{2})\.([0-7])",
        value.strip().lower())
    if match is None:
        raise RuntimeError("Invalid GPU PCI bus ID")
    domain, bus, device, function = match.groups()
    return f"{int(domain, 16):08x}:{bus}:{device}.{function}"


def gpu_identity(expected):
    from cupy.cuda import runtime

    if runtime.getDeviceCount() != len(expected):
        raise RuntimeError("CUDA visibility differs from requested GPUs")
    result = subprocess.run([
        "nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,"
        "memory.free,memory.total",
        "--format=csv,noheader,nounits"], check=True,
        capture_output=True, text=True)
    physical = {}
    seen_uuids = set()
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) != 5:
            raise RuntimeError("Invalid GPU identity from nvidia-smi")
        bus = normalized_pci_bus_id(row[2])
        uuid = row[1].strip()
        free, total = int(row[3]), int(row[4])
        if (bus in physical or uuid.lower() in seen_uuids or
                not uuid or free < 0 or total <= 0 or free > total):
            raise RuntimeError("Invalid GPU identity from nvidia-smi")
        seen_uuids.add(uuid.lower())
        physical[bus] = (
            int(row[0]), uuid, free, total)
    mapping = []
    for logical, uuid in enumerate(expected):
        key = normalized_pci_bus_id(
            runtime.deviceGetPCIBusId(logical))
        if key not in physical or physical[key][1].lower() != uuid.lower():
            raise RuntimeError("CUDA and nvidia-smi GPU identity differ")
        mapping.append({"logical_id": logical,
                        "physical_id": physical[key][0],
                        "pci_bus_id": key,
                        "uuid": uuid,
                        "free_bytes": physical[key][2] * 1024**2,
                        "total_bytes": physical[key][3] * 1024**2})
    return mapping


def gpu_used_bytes(expected):
    result = subprocess.run([
        "nvidia-smi", "--query-gpu=uuid,memory.used",
        "--format=csv,noheader,nounits"], check=True,
        capture_output=True, text=True, timeout=10)
    used = {}
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) != 2:
            raise RuntimeError("Invalid GPU memory sample")
        uuid = row[0].strip().lower()
        value = int(row[1])
        if not uuid or uuid in used or value < 0:
            raise RuntimeError("Invalid GPU memory sample")
        used[uuid] = value * 1024**2
    if not all(uuid.lower() in used for uuid in expected):
        raise RuntimeError("GPU memory sample omitted a requested UUID")
    return {uuid: used[uuid.lower()] for uuid in expected}


def process_tree_rss(worker_identities=()):
    import psutil

    parent = psutil.Process()
    processes = {item.pid: item for item in
                 [parent] + parent.children(recursive=True)}
    tracked = set()
    for pid, started in worker_identities:
        try:
            process = psutil.Process(pid)
            if process.create_time() != started:
                raise RuntimeError("cuGraph worker PID was reused")
            processes[pid] = process
            tracked.add(pid)
            for child in process.children(recursive=True):
                processes[child.pid] = child
        except psutil.NoSuchProcess as error:
            raise RuntimeError("cuGraph worker process exited") from error
        except psutil.AccessDenied as error:
            raise RuntimeError("Cannot sample cuGraph worker RAM") from error
    total = 0
    for process in processes.values():
        try:
            total += process.memory_info().rss
        except psutil.NoSuchProcess as error:
            if process.pid == parent.pid or process.pid in tracked:
                raise RuntimeError("cuGraph worker process exited") from error
        except psutil.AccessDenied as error:
            raise RuntimeError("Cannot sample cuGraph worker RAM") from error
    return total


def worker_device_list(identities):
    return ",".join(item["uuid"] for item in identities)


def worker_cuda_identity():
    from cupy.cuda import runtime

    return {"pid": os.getpid(),
            "pci_bus_id": normalized_pci_bus_id(
                runtime.deviceGetPCIBusId(0))}


def verify_distributed_graph(graph, client, devices):
    workers = len(client.scheduler_info()["workers"])
    if workers != devices or not graph.is_multi_gpu():
        raise RuntimeError("cuGraph did not build a graph on all GPUs")
    return workers


class GpuMemorySampler:
    def __init__(self, identities):
        self.uuids = [item["uuid"] for item in identities]
        self.bus_uuids = {item["pci_bus_id"]: item["uuid"]
                          for item in identities}
        self.total = {item["uuid"]: item["total_bytes"]
                      for item in identities}
        self.baseline = {item["uuid"]: item["total_bytes"] -
                         item["free_bytes"] for item in identities}
        self.peak = self.baseline.copy()
        self.rss_baseline = None
        self.rss_peak = 0
        self.samples = 0
        self.worker_identities = ()
        self.worker_devices = ()
        self.tracked_samples = 0
        self.errors = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)

    def _sample(self):
        try:
            observed = gpu_used_bytes(self.uuids)
            for uuid in self.uuids:
                if observed[uuid] > self.total[uuid]:
                    raise ValueError("GPU memory sample exceeds capacity")
                self.peak[uuid] = max(self.peak[uuid], observed[uuid])
            identities = self.worker_identities
            rss = process_tree_rss(identities)
            if rss <= 0:
                raise ValueError("Invalid process-tree RSS sample")
            if self.rss_baseline is None:
                self.rss_baseline = rss
            self.rss_peak = max(self.rss_peak, rss)
            self.samples += 1
            if identities:
                self.tracked_samples += 1
        except (ImportError, OSError, ValueError, RuntimeError,
                subprocess.SubprocessError) as error:
            self.errors.append(str(error))

    def _poll(self):
        while not self.stop_event.wait(0.25):
            self._sample()

    def start(self):
        self._sample()
        if self.errors:
            raise RuntimeError(self.errors[0])
        self.thread.start()

    def track_workers(self, client):
        import psutil

        reported = client.run(worker_cuda_identity)
        if len(reported) != len(self.uuids):
            raise RuntimeError("cuGraph worker GPU map is incomplete")
        pids = tuple(item.get("pid") for item in reported.values())
        buses = tuple(item.get("pci_bus_id") for item in reported.values())
        if (any(not isinstance(pid, int) or pid <= 0 or
                pid == os.getpid() for pid in pids) or
                len(set(pids)) != len(pids) or
                len(set(buses)) != len(buses) or
                set(buses) != set(self.bus_uuids)):
            raise RuntimeError("cuGraph worker GPU map is incomplete")
        try:
            identities = tuple((pid, psutil.Process(pid).create_time())
                               for pid in pids)
        except (psutil.NoSuchProcess, psutil.AccessDenied) as error:
            raise RuntimeError("Cannot identify cuGraph workers") from error
        self.worker_identities = identities
        self.worker_devices = tuple({
            "address": address, "pid": item["pid"],
            "pci_bus_id": item["pci_bus_id"],
            "uuid": self.bus_uuids[item["pci_bus_id"]]}
            for address, item in reported.items())
        self._sample()
        if self.errors:
            raise RuntimeError(self.errors[0])

    def finish(self):
        self.stop_event.set()
        self.thread.join()
        self._sample()
        if (self.errors or self.samples < 2 or
                self.tracked_samples < 2 or self.rss_baseline is None):
            raise RuntimeError("GPU memory sampling was incomplete: " +
                               "; ".join(self.errors))
        devices = [{"uuid": uuid,
                    "baseline_used_bytes": self.baseline[uuid],
                    "peak_used_bytes": self.peak[uuid],
                    "peak_delta_bytes": max(0, self.peak[uuid] -
                                            self.baseline[uuid])}
                   for uuid in self.uuids]
        return {"scope": "whole_physical_gpu", "samples": self.samples,
                "interval_seconds": 0.25, "devices": devices,
                "workers": self.worker_devices,
                "host_process_tree": {
                    "scope": "launcher_and_workers",
                    "samples": self.samples,
                    "tracked_samples": self.tracked_samples,
                    "baseline_rss_bytes": self.rss_baseline,
                    "peak_rss_bytes": self.rss_peak}}


def write_labels(parts, output, vertices, ram_limit, offsets_path):
    if vertices * 48 > 0.7 * ram_limit:
        raise MemoryError("Community-label audit exceeds RAM budget")
    labels = np.memmap(output, dtype="<u4", mode="w+",
                       shape=(vertices,))
    labels[:] = 0
    for delayed in parts.to_delayed():
        frame = delayed.compute().to_pandas()
        ids = frame["vertex"].to_numpy(dtype=np.int64)
        group = frame["partition"].to_numpy(dtype=np.int64)
        if (np.any(ids < 0) or np.any(ids >= vertices) or
                np.any(group < 0) or np.any(group >= vertices) or
                len(np.unique(ids)) != len(ids) or
                np.any(labels[ids] != 0)):
            raise ValueError("Invalid or repeated cuGraph vertex labels")
        labels[ids] = group + 1
    offsets = np.memmap(offsets_path, dtype="<u8", mode="r")
    for first in range(0, vertices, 1_000_000):
        last = min(first + 1_000_000, vertices)
        missing = np.flatnonzero(labels[first:last] == 0)
        if not len(missing):
            continue
        degrees = np.diff(offsets[first:last + 1])
        if np.any(degrees[missing] != 0):
            raise ValueError("cuGraph omitted non-isolated source rows")
        labels[first + missing] = vertices + first + missing + 1
    communities = compact_labels(labels, vertices)
    labels.flush()
    return communities


def compact_labels(labels, vertices):
    popcount = np.array([value.bit_count() for value in range(256)],
                        dtype=np.uint8)
    present = np.zeros((2 * vertices + 8) // 8, dtype=np.uint8)
    for first in range(0, vertices, 1_000_000):
        values = np.unique(labels[first:first + 1_000_000])
        bits = (1 << (values & 7)).astype(np.uint8)
        np.bitwise_or.at(present, values >> 3, bits)
    prefix = np.empty(len(present) + 1, dtype=np.uint32)
    prefix[0] = 0
    for first in range(0, len(present), 1_000_000):
        last = min(first + 1_000_000, len(present))
        np.cumsum(popcount[present[first:last]], dtype=np.uint32,
                  out=prefix[first + 1:last + 1])
        prefix[first + 1:last + 1] += prefix[first]
    for first in range(0, vertices, 1_000_000):
        last = min(first + 1_000_000, vertices)
        values = labels[first:last]
        slots = values >> 3
        below = ((1 << (values & 7)) - 1).astype(np.uint8)
        labels[first:last] = prefix[slots] + \
            popcount[present[slots] & below] + 1
    return int(prefix[-1])


def run(args):
    import cudf
    import cugraph
    import cugraph.dask as dcg
    import dask.dataframe as dd
    from dask_cuda import LocalCUDACluster
    from distributed import Client
    import cugraph.dask.comms.comms as comms

    paths = [Path(path) for path in args.paths]
    verify_graph(paths, args.vertices, args.chunk_rows)
    identities = gpu_identity(args.uuids)
    if any(item["free_bytes"] * 0.7 < args.vram_estimate
           for item in identities):
        raise MemoryError("cuGraph graph exceeds free GPU memory")
    sampler = GpuMemorySampler(identities)
    sampler.start()
    cluster = None
    client = None
    initialized = False
    try:
        cluster = LocalCUDACluster(
            CUDA_VISIBLE_DEVICES=worker_device_list(identities),
            n_workers=len(args.uuids), threads_per_worker=1,
            device_memory_limit=0, enable_cudf_spill=False)
        client = Client(cluster)
        client.wait_for_workers(len(args.uuids))
        sampler.track_workers(client)
        comms.initialize(p2p=True)
        initialized = True
        bounds = [(first, min(first + args.chunk_rows, args.vertices))
                  for first in range(0, args.vertices, args.chunk_rows)]
        meta = cudf.DataFrame({
            "src": np.empty(0, dtype=np.int32),
            "dst": np.empty(0, dtype=np.int32),
            "weight": np.empty(0, dtype=np.float32)})
        frame = dd.from_map(partial(csr_partition, paths=paths),
                            bounds, meta=meta)
        graph = cugraph.Graph(directed=False)
        started = time.monotonic()
        graph.from_dask_cudf_edgelist(
            frame, source="src", destination="dst",
            edge_attr="weight", renumber=False)
        workers = verify_distributed_graph(graph, client,
                                           len(args.uuids))
        built = time.monotonic()
        options = {"resolution": args.resolution}
        if args.method == "leiden":
            options.update(max_iter=args.iterations,
                           random_state=args.seed)
        else:
            options.update(max_level=args.iterations)
        parts, reported = getattr(dcg, args.method)(graph, **options)
        fitted = time.monotonic()
        communities = write_labels(parts, args.output,
                                   args.vertices, args.ram_limit,
                                   paths[0])
        exported = time.monotonic()
        memory = sampler.finish()
    finally:
        sampler.stop_event.set()
        sampler.thread.join()
        if initialized:
            comms.destroy()
        if client is not None:
            client.close()
        if cluster is not None:
            cluster.close()
    metadata = {"method": args.method, "backend": "cugraph_dask",
                "cugraph_version": cugraph.__version__,
                "devices": identities, "rows": args.vertices,
                "distributed_graph": True, "worker_count": workers,
                "communities": communities,
                "iterations_requested": args.iterations,
                "resolution": args.resolution,
                "seed_requested": args.seed,
                "modularity_reported": float(reported),
                "graph_build_seconds": built - started,
                "fit_seconds": fitted - built,
                "label_export_seconds": exported - fitted,
                "gpu_memory": memory,
                "device_spilling": False,
                "iteration_semantics": "max_iter" if args.method ==
                "leiden" else "max_level",
                "seed_applied": args.method == "leiden"}
    args.metadata.write_text(json.dumps(metadata) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs=3, required=True)
    parser.add_argument("--vertices", type=int, required=True)
    parser.add_argument("--chunk-rows", type=int, required=True)
    parser.add_argument("--method", choices=("leiden", "louvain"),
                        required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--resolution", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--uuids", nargs="+", required=True)
    parser.add_argument("--vram-estimate", type=int, required=True)
    parser.add_argument("--ram-limit", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
