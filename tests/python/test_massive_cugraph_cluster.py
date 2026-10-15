"""CPU-only contract tests for the optional multi-GPU cuGraph worker."""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).parents[2] / "inst/python/massive_cugraph_cluster.py"
spec = importlib.util.spec_from_file_location("massive_cugraph_cluster", SCRIPT)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class Partition:
    def __init__(self, frame):
        self.frame = frame

    def compute(self):
        return types.SimpleNamespace(to_pandas=lambda: self.frame)


class Parts:
    def __init__(self, frames):
        self.frames = frames

    def to_delayed(self):
        return [Partition(frame) for frame in self.frames]


class CuGraphWorkerTests(unittest.TestCase):
    def test_csr_partition_keeps_each_undirected_edge_once(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in
                     ("offsets", "indices", "weights")]
            np.array([0, 2, 4, 6, 8], dtype="<u8").tofile(paths[0])
            np.array([2, 4, 1, 3, 2, 4, 1, 3],
                     dtype="<u4").tofile(paths[1])
            np.ones(8, dtype="<f4").tofile(paths[2])
            cudf = types.SimpleNamespace(DataFrame=pd.DataFrame)
            with patch.dict(sys.modules, {"cudf": cudf}):
                first = worker.csr_partition((0, 2), paths)
                second = worker.csr_partition((2, 4), paths)
            edges = list(zip(pd.concat([first, second]).src,
                             pd.concat([first, second]).dst))
            self.assertEqual(edges, [(0, 1), (0, 3), (1, 2), (2, 3)])
            worker.verify_graph(paths, 4, 2)
            np.array([0, 2, 4, 4, 8], dtype="<u8").tofile(paths[0])
            worker.verify_graph(paths, 4, 2)
            np.zeros(5, dtype="<u8").tofile(paths[0])
            paths[1].write_bytes(b"")
            paths[2].write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "at least one"):
                worker.verify_graph(paths, 4, 2)

    def test_csr_partition_rejects_invalid_ids_and_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in
                     ("offsets", "indices", "weights")]
            np.array([0, 1, 2], dtype="<u8").tofile(paths[0])
            ids = np.array([2, 1], dtype="<u4")
            weights = np.ones(2, dtype="<f4")
            cudf = types.SimpleNamespace(DataFrame=pd.DataFrame)
            with patch.dict(sys.modules, {"cudf": cudf}):
                for invalid in (0, 3):
                    ids[0] = invalid
                    ids.tofile(paths[1])
                    weights.tofile(paths[2])
                    with self.assertRaisesRegex(ValueError,
                                                "neighbor ID"):
                        worker.csr_partition((0, 2), paths)
                ids[0] = 2
                ids.tofile(paths[1])
                for invalid in (np.nan, -0.5):
                    weights[0] = invalid
                    weights.tofile(paths[2])
                    with self.assertRaisesRegex(ValueError,
                                                "edge weights"):
                        worker.csr_partition((0, 2), paths)

    def test_graph_rejects_decreasing_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in
                     ("offsets", "indices", "weights")]
            np.array([0, 2, 1], dtype="<u8").tofile(paths[0])
            np.array([2], dtype="<u4").tofile(paths[1])
            np.ones(1, dtype="<f4").tofile(paths[2])
            with self.assertRaisesRegex(ValueError, "not monotone"):
                worker.verify_graph(paths, 2, 1)

    def test_label_export_checks_coverage_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "labels.u32"
            offsets = Path(directory) / "offsets.u64"
            np.arange(5, dtype="<u8").tofile(offsets)
            good = Parts([pd.DataFrame({"vertex": [0, 2],
                                        "partition": [0, 0]}),
                          pd.DataFrame({"vertex": [1, 3],
                                        "partition": [1, 1]})])
            self.assertEqual(worker.write_labels(good, output, 4,
                                                 4096, offsets), 2)
            self.assertEqual(np.fromfile(output, dtype="<u4").tolist(),
                             [1, 2, 1, 2])
            missing = Parts([pd.DataFrame({"vertex": [0, 1, 2],
                                           "partition": [0, 1, 0]})])
            with self.assertRaisesRegex(ValueError, "non-isolated"):
                worker.write_labels(missing, output, 4, 4096, offsets)
            repeated = Parts([pd.DataFrame({"vertex": [0, 1],
                                            "partition": [0, 1]}),
                              pd.DataFrame({"vertex": [1, 2, 3],
                                            "partition": [1, 0, 1]})])
            with self.assertRaisesRegex(ValueError, "repeated"):
                worker.write_labels(repeated, output, 4, 4096, offsets)

    def test_label_export_assigns_verified_isolate_a_singleton(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "labels.u32"
            offsets = Path(directory) / "offsets.u64"
            np.array([0, 1, 2, 2, 3, 4], dtype="<u8").tofile(offsets)
            parts = Parts([pd.DataFrame({"vertex": [0, 1, 3, 4],
                                         "partition": [0, 0, 1, 1]})])
            self.assertEqual(worker.write_labels(parts, output, 5,
                                                 4096, offsets), 3)
            self.assertEqual(np.fromfile(output, dtype="<u4").tolist(),
                             [1, 1, 3, 2, 2])

    def test_label_compaction_matches_sorted_unique_ids(self):
        original = np.array([1, 5, 5, 12, 3, 12, 8], dtype="<u4")
        expected = np.searchsorted(np.unique(original), original) + 1
        observed = original.copy()
        self.assertEqual(worker.compact_labels(observed, len(observed)), 5)
        np.testing.assert_array_equal(observed, expected)

    def test_label_compaction_limits_unique_to_one_chunk(self):
        rows = 1_000_003
        original_unique = np.unique

        def bounded_unique(values):
            self.assertLessEqual(len(values), 1_000_000)
            return original_unique(values)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.u32"
            labels = np.memmap(path, dtype="<u4", mode="w+",
                               shape=(rows,))
            labels[:] = np.arange(rows, dtype=np.uint32) % 97 + 1
            with patch.object(worker.np, "unique",
                              side_effect=bounded_unique):
                self.assertEqual(worker.compact_labels(labels, rows), 97)
            self.assertEqual(labels[0], 1)
            self.assertEqual(labels[-1], rows % 97)

    def test_gpu_identity_rejects_uuid_mismatch(self):
        runtime = types.SimpleNamespace(
            getDeviceCount=lambda: 2,
            deviceGetPCIBusId=lambda i: [b"0000:20:00.0",
                                              b"0000:40:00.0"][i])
        cuda = types.ModuleType("cupy.cuda")
        cuda.runtime = runtime
        cupy = types.ModuleType("cupy")
        cupy.cuda = cuda
        output = ("3, GPU-A, 0000:20:00.0, 1024, 2048\n"
                  "7, GPU-B, 0000:40:00.0, 512, 2048\n")
        with patch.dict(sys.modules, {"cupy": cupy,
                                      "cupy.cuda": cuda}), patch.object(
                worker.subprocess, "run", return_value=
                types.SimpleNamespace(stdout=output)):
            identities = worker.gpu_identity(["GPU-A", "GPU-B"])
            self.assertEqual([item["physical_id"] for item in identities],
                             [3, 7])
            self.assertEqual([item["pci_bus_id"] for item in identities],
                             ["00000000:20:00.0", "00000000:40:00.0"])
            self.assertEqual([item["free_bytes"] for item in identities],
                             [1024**3, 512 * 1024**2])
            with patch.dict(worker.os.environ, {}, clear=True):
                self.assertEqual(worker.worker_device_list(identities),
                                 "GPU-A,GPU-B")
                self.assertNotIn("CUDA_VISIBLE_DEVICES",
                                 worker.os.environ)
            with self.assertRaisesRegex(RuntimeError, "identity differ"):
                worker.gpu_identity(["GPU-B", "GPU-A"])
            duplicate = output.replace("GPU-B", "GPU-A")
            with patch.object(worker.subprocess, "run", return_value=
                              types.SimpleNamespace(stdout=duplicate)):
                with self.assertRaisesRegex(RuntimeError,
                                            "Invalid GPU identity"):
                    worker.gpu_identity(["GPU-A", "GPU-B"])
            impossible = output.replace("1024, 2048", "3072, 2048")
            with patch.object(worker.subprocess, "run", return_value=
                              types.SimpleNamespace(stdout=impossible)):
                with self.assertRaisesRegex(RuntimeError,
                                            "Invalid GPU identity"):
                    worker.gpu_identity(["GPU-A", "GPU-B"])

    def test_distributed_graph_requires_all_workers(self):
        client = types.SimpleNamespace(scheduler_info=lambda: {
            "workers": {"first": {}, "second": {}}})
        graph = types.SimpleNamespace(is_multi_gpu=lambda: True)
        self.assertEqual(worker.verify_distributed_graph(graph,
                                                         client, 2), 2)
        with self.assertRaisesRegex(RuntimeError, "all GPUs"):
            worker.verify_distributed_graph(graph, client, 3)
        graph.is_multi_gpu = lambda: False
        with self.assertRaisesRegex(RuntimeError, "all GPUs"):
            worker.verify_distributed_graph(graph, client, 2)

    def test_gpu_identity_keeps_distinct_pci_domains(self):
        runtime = types.SimpleNamespace(
            getDeviceCount=lambda: 2,
            deviceGetPCIBusId=lambda i: [b"0000:20:00.0",
                                          b"0001:20:00.0"][i])
        cuda = types.ModuleType("cupy.cuda")
        cuda.runtime = runtime
        cupy = types.ModuleType("cupy")
        cupy.cuda = cuda
        output = ("3, GPU-A, 00000000:20:00.0, 1024, 2048\n"
                  "7, GPU-B, 00000001:20:00.0, 512, 2048\n")
        with patch.dict(sys.modules, {"cupy": cupy,
                                      "cupy.cuda": cuda}), patch.object(
                worker.subprocess, "run", return_value=
                types.SimpleNamespace(stdout=output)):
            identities = worker.gpu_identity(["GPU-A", "GPU-B"])
        self.assertEqual([item["pci_bus_id"] for item in identities],
                         ["00000000:20:00.0", "00000001:20:00.0"])
        self.assertEqual(len(set(item["pci_bus_id"] for item in
                                 identities)), 2)

    def test_worker_cuda_identity_uses_first_visible_gpu(self):
        runtime = types.SimpleNamespace(
            deviceGetPCIBusId=lambda index: b"0000:40:00.0")
        cuda = types.ModuleType("cupy.cuda")
        cuda.runtime = runtime
        cupy = types.ModuleType("cupy")
        cupy.cuda = cuda
        with patch.dict(sys.modules, {"cupy": cupy,
                                      "cupy.cuda": cuda}), patch.object(
                worker.os, "getpid", return_value=42):
            self.assertEqual(worker.worker_cuda_identity(), {
                "pid": 42, "pci_bus_id": "00000000:40:00.0"})

    def test_process_tree_rss_includes_live_workers(self):
        child = types.SimpleNamespace(pid=2, memory_info=lambda:
                                      types.SimpleNamespace(rss=300))
        parent = types.SimpleNamespace(pid=1, children=lambda **_:
                                       [child], memory_info=lambda:
                                       types.SimpleNamespace(rss=700))
        psutil = types.SimpleNamespace(Process=lambda: parent,
                                       NoSuchProcess=RuntimeError,
                                       AccessDenied=PermissionError)
        with patch.dict(sys.modules, {"psutil": psutil}):
            self.assertEqual(worker.process_tree_rss(), 1000)
            child.memory_info = lambda: (_ for _ in ()).throw(
                PermissionError())
            with self.assertRaisesRegex(RuntimeError,
                                        "Cannot sample cuGraph worker RAM"):
                worker.process_tree_rss()

    def test_process_tree_rss_tracks_reparented_worker_once(self):
        child = types.SimpleNamespace(
            pid=2, create_time=lambda: 9.0,
            children=lambda **_: [], memory_info=lambda:
            types.SimpleNamespace(rss=300))
        parent = types.SimpleNamespace(
            pid=1, children=lambda **_: [child], memory_info=lambda:
            types.SimpleNamespace(rss=700))
        distant = types.SimpleNamespace(
            pid=3, create_time=lambda: 11.0,
            children=lambda **_: [], memory_info=lambda:
            types.SimpleNamespace(rss=500))
        lookup = {1: parent, 2: child, 3: distant}
        psutil = types.SimpleNamespace(
            Process=lambda pid=None: lookup[pid or 1],
            NoSuchProcess=KeyError, AccessDenied=PermissionError)
        with patch.dict(sys.modules, {"psutil": psutil}):
            self.assertEqual(worker.process_tree_rss(((2, 9.0),)), 1000)
            self.assertEqual(worker.process_tree_rss(((3, 11.0),)), 1500)
            with self.assertRaisesRegex(RuntimeError, "PID was reused"):
                worker.process_tree_rss(((3, 12.0),))
            with self.assertRaisesRegex(RuntimeError, "process exited"):
                worker.process_tree_rss(((4, 11.0),))

    def test_memory_sampler_rejects_missing_uuid_and_records_peaks(self):
        identities = [{"uuid": "GPU-A", "total_bytes": 1000,
                       "free_bytes": 900, "pci_bus_id": "20:00.0"},
                      {"uuid": "GPU-B", "total_bytes": 1000,
                       "free_bytes": 800, "pci_bus_id": "40:00.0"}]
        samples = iter([{"GPU-A": 120, "GPU-B": 210},
                        {"GPU-A": 140, "GPU-B": 230},
                        {"GPU-A": 150, "GPU-B": 240}])
        rss = iter([1000, 1200, 1300])
        psutil = types.SimpleNamespace(
            Process=lambda pid: types.SimpleNamespace(
                create_time=lambda: float(pid)))
        client = types.SimpleNamespace(run=lambda _: {
            "a": {"pid": 2, "pci_bus_id": "20:00.0"},
            "b": {"pid": 3, "pci_bus_id": "40:00.0"}})
        with patch.object(worker, "gpu_used_bytes",
                          side_effect=lambda _: next(samples)), patch.object(
                worker, "process_tree_rss",
                side_effect=lambda _: next(rss)), patch.dict(
                sys.modules, {"psutil": psutil}), patch.object(
                worker.os, "getpid", return_value=1):
            sampler = worker.GpuMemorySampler(identities)
            sampler.start()
            sampler.track_workers(client)
            result = sampler.finish()
        self.assertEqual(result["samples"], 3)
        self.assertEqual(result["host_process_tree"]["tracked_samples"], 2)
        self.assertEqual(result["host_process_tree"]["scope"],
                         "launcher_and_workers")
        self.assertEqual([item["uuid"] for item in result["workers"]],
                         ["GPU-A", "GPU-B"])
        self.assertEqual([item["peak_used_bytes"] for item in
                          result["devices"]], [150, 240])
        self.assertEqual(result["host_process_tree"]["peak_rss_bytes"],
                         1300)
        with patch.object(worker.subprocess, "run", return_value=
                          types.SimpleNamespace(stdout="GPU-A, 7\n")):
            with self.assertRaisesRegex(RuntimeError, "omitted"):
                worker.gpu_used_bytes(["GPU-A", "GPU-B"])
        for output in ("GPU-A, 7\nGPU-A, 8\nGPU-B, 9\n",
                       "GPU-A, -1\nGPU-B, 9\n"):
            with patch.object(worker.subprocess, "run", return_value=
                              types.SimpleNamespace(stdout=output)):
                with self.assertRaisesRegex(RuntimeError,
                                            "Invalid GPU memory sample"):
                    worker.gpu_used_bytes(["GPU-A", "GPU-B"])
        with patch.object(worker, "gpu_used_bytes", return_value={
                "GPU-A": 1001, "GPU-B": 200}), patch.object(
                worker, "process_tree_rss", return_value=1000):
            with self.assertRaisesRegex(RuntimeError,
                                        "exceeds capacity"):
                worker.GpuMemorySampler(identities).start()

    def test_memory_sampler_rejects_missing_host_rss(self):
        identities = [{"uuid": "GPU-A", "total_bytes": 1000,
                       "free_bytes": 900, "pci_bus_id": "20:00.0"}]
        with patch.object(worker, "gpu_used_bytes", return_value={
                "GPU-A": 200}), patch.object(
                worker, "process_tree_rss", side_effect=ImportError(
                    "psutil unavailable")):
            with self.assertRaisesRegex(RuntimeError,
                                        "psutil unavailable"):
                worker.GpuMemorySampler(identities).start()

    def test_memory_sampler_rejects_incomplete_worker_pid_map(self):
        identities = [{"uuid": "GPU-A", "total_bytes": 1000,
                       "free_bytes": 900, "pci_bus_id": "20:00.0"},
                      {"uuid": "GPU-B", "total_bytes": 1000,
                       "free_bytes": 900, "pci_bus_id": "40:00.0"}]
        sampler = worker.GpuMemorySampler(identities)
        with patch.object(worker.os, "getpid", return_value=1), \
                patch.dict(sys.modules, {"psutil": types.SimpleNamespace()}):
            good = {"a": {"pid": 2, "pci_bus_id": "20:00.0"},
                    "b": {"pid": 3, "pci_bus_id": "40:00.0"}}
            cases = [dict(list(good.items())[:1]),
                     {"a": good["a"], "b": good["a"]},
                     {"a": {"pid": 1, "pci_bus_id": "20:00.0"},
                      "b": good["b"]},
                     {"a": {"pid": "2", "pci_bus_id": "20:00.0"},
                      "b": good["b"]},
                     {"a": good["a"],
                      "b": {"pid": 3, "pci_bus_id": "20:00.0"}}]
            for mapping in cases:
                client = types.SimpleNamespace(run=lambda _: mapping)
                with self.assertRaisesRegex(RuntimeError,
                                            "GPU map is incomplete"):
                    sampler.track_workers(client)


if __name__ == "__main__":
    unittest.main()
