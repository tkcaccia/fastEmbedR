"""CPU-side invariants for the bounded multi-GPU clustering worker."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = (Path(__file__).parents[2] / "inst/python" /
          "massive_cuda_stream_cluster.py")
SPEC = importlib.util.spec_from_file_location("stream_cluster", SCRIPT)
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)


def two_triangles(tmp_path):
    offset = np.arange(0, 13, 2, dtype="<u8")
    index = np.array([2, 3, 1, 3, 1, 2,
                      5, 6, 4, 6, 4, 5], dtype="<u4")
    weight = np.ones(12, dtype="<f4")
    paths = []
    for name, values in (("offset", offset), ("index", index),
                         ("weight", weight)):
        path = tmp_path / name
        values.tofile(path)
        paths.append(str(path))
    return paths


def test_graph_validation_and_exact_modularity(tmp_path):
    paths = two_triangles(tmp_path)
    offsets, edges = WORKER.validate_graph(paths, 6, 4)
    degree, total = WORKER.graph_degree(paths, offsets, 6, 4, 4)
    assert edges == 12
    assert total == 12
    assert WORKER.bounded_end(offsets, 0, 6, 2, 4) == 2
    labels = np.array([0, 0, 0, 3, 3, 3], dtype=np.int32)
    score, volumes = WORKER.modularity(paths, offsets, labels,
                                        degree, total, 1, 2, 4)
    assert score == pytest.approx(0.5)
    assert volumes[[0, 3]].tolist() == [6, 6]
    with pytest.raises(ValueError, match="edge budget"):
        WORKER.validate_graph(paths, 6, 1)


def test_full_local_fit_keeps_disk_state_and_audits_moves(
        tmp_path, monkeypatch):
    paths = two_triangles(tmp_path)

    def proposed_phase(paths, vertices, state, devices, chunk_rows,
                       edge_budget, color, seed, total, resolution,
                       pool_limit=0):
        labels = np.memmap(state["labels"], dtype="<i4", mode="r",
                           shape=(vertices,))
        proposals = np.memmap(state["proposals"], dtype="<i4",
                              mode="r+", shape=(vertices,))
        if color == 0:
            proposals[:3] = 0
            proposals[3:] = 3
        else:
            proposals[:] = labels
        proposals.flush()
        return [{"device": d, "rows": vertices // devices,
                 "pci_bus_id": f"0000:0{d}:00.0",
                 "peak_pool_bytes": 1024} for d in range(devices)]

    monkeypatch.setattr(WORKER, "run_phase", proposed_phase)
    output = tmp_path / "clusters.u32"
    metadata = tmp_path / "clusters.json"
    args = SimpleNamespace(paths=paths, vertices=6, devices=2,
        chunk_rows=2, edge_budget=4, passes=2, resolution=1.0,
        seed=1, output=str(output), metadata=str(metadata))
    WORKER.fit(args)
    labels = np.fromfile(str(output) + ".part", dtype="<u4")
    result = json.loads(metadata.read_text())
    assert labels.tolist() == [1, 1, 1, 2, 2, 2]
    assert result["modularity"] == pytest.approx(0.5)
    assert result["gpu_devices_used"] == [0, 1]
    assert result["graph_file_bytes"] > 0
    assert all(not (tmp_path / ("clusters.u32." + name + ".part"))
               .exists() for name in ("labels", "proposals",
                                    "volumes", "degree"))

    initial = tmp_path / "initial.u32"
    np.array([1, 1, 1, 2, 2, 2], dtype="<u4").tofile(initial)
    resume = SimpleNamespace(**vars(args))
    resume.initial = str(initial)
    resume.initial_count = 2
    resume.output = str(tmp_path / "from_initial.u32")
    resume.metadata = str(tmp_path / "from_initial.json")
    WORKER.fit(resume)
    report = json.loads(Path(resume.metadata).read_text())
    assert report["initial_partition"] is True
    assert report["modularity_initial"] == pytest.approx(0.5)
