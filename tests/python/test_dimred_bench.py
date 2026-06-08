from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from dimred_bench.benchmark import BenchmarkConfig, run_benchmark
from dimred_bench.datasets import load_dataset
from dimred_bench.metrics import embedding_metrics, procrustes_stability
from dimred_bench.runners import run_implementation


class DatasetTests(unittest.TestCase):
    def test_iris_loads(self) -> None:
        ds = load_dataset("iris", seed=1)
        self.assertEqual(ds.X.shape, (150, 4))
        self.assertEqual(ds.y.shape, (150,))
        self.assertTrue(np.isfinite(ds.X).all())

    def test_digits_subset_loads(self) -> None:
        ds = load_dataset("digits", subset=100, seed=1)
        self.assertEqual(ds.X.shape[0], 100)
        self.assertEqual(ds.y.shape, (100,))


class RunnerMetricTests(unittest.TestCase):
    def test_pca_runner_and_metrics(self) -> None:
        ds = load_dataset("iris", seed=2)
        result = run_implementation("pca", ds.X, seed=2, n_components=2)
        self.assertEqual(result.status, "ok")
        assert result.embedding is not None
        scores = embedding_metrics(ds.X, result.embedding, ds.y, n_neighbors=10)
        self.assertGreater(scores["trustworthiness"], 0.8)
        self.assertIsNotNone(scores["silhouette"])

    def test_procrustes_stability(self) -> None:
        rng = np.random.default_rng(3)
        x = rng.normal(size=(50, 2))
        y = x[:, ::-1]
        score = procrustes_stability(x, y)
        self.assertGreater(score, 0.99)


class BenchmarkTests(unittest.TestCase):
    def test_benchmark_writes_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = BenchmarkConfig(
                datasets=("iris",),
                implementations=("pca",),
                repeats=2,
                output_dir=tmp,
                metric_sample=100,
                pca_input_dims=None,
            )
            df = run_benchmark(config)
            self.assertEqual(len(df), 2)
            self.assertTrue((Path(tmp) / "results.csv").exists())
            self.assertTrue((Path(tmp) / "summary.csv").exists())
            self.assertTrue((df["status"] == "ok").all())


if __name__ == "__main__":
    unittest.main()
