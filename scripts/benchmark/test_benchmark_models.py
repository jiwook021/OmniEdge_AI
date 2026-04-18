"""Unit tests verifying STT removal from benchmark_models.

Validates:
  1. No STT variants remain in the registry
  2. discover_variants never returns STT entries
  3. bench_fn_map has no 'stt' key
  4. rank_vram_fit priority list has no 'stt'
  5. VRAM ranking works correctly without STT
"""

from __future__ import annotations

import sys
import os
import unittest
from unittest.mock import patch

# Ensure project root is on sys.path so `scripts.common` resolves
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
sys.path.insert(0, PROJECT_ROOT)

from scripts.benchmark.benchmark_models import (
    VARIANT_REGISTRY,
    ModelVariant,
    VariantResult,
    discover_variants,
    rank_vram_fit,
)


class TestNoSttInRegistry(unittest.TestCase):
    """Ensure the STT module is completely removed from the variant registry."""

    def test_no_stt_variants_in_registry(self):
        stt_variants = [v for v in VARIANT_REGISTRY if v.module == "stt"]
        self.assertEqual(stt_variants, [],
                         f"Found STT variants that should have been removed: "
                         f"{[v.name for v in stt_variants]}")

    def test_registry_modules_exclude_stt(self):
        modules = {v.module for v in VARIANT_REGISTRY}
        self.assertNotIn("stt", modules)

    def test_registry_still_has_expected_modules(self):
        modules = {v.module for v in VARIANT_REGISTRY}
        for expected in ("llm", "tts", "vad", "seg"):
            self.assertIn(expected, modules,
                          f"Expected module '{expected}' missing from registry")


class TestDiscoverVariantsNoStt(unittest.TestCase):
    """discover_variants must never yield STT entries."""

    def _make_all_exist(self):
        """Patch ModelVariant.exists to always return True."""
        return patch.object(ModelVariant, "exists", return_value=True)

    def test_default_discovery_no_stt(self):
        with self._make_all_exist():
            variants = discover_variants()
        self.assertTrue(len(variants) > 0, "Expected at least one variant")
        stt = [v for v in variants if v.module == "stt"]
        self.assertEqual(stt, [])

    def test_sweep_no_stt(self):
        with self._make_all_exist():
            variants = discover_variants(sweep=True)
        stt = [v for v in variants if v.module == "stt"]
        self.assertEqual(stt, [])

    def test_filter_stt_returns_empty(self):
        """Explicitly requesting module=stt should return nothing."""
        with self._make_all_exist():
            variants = discover_variants(module_filter="stt")
        self.assertEqual(variants, [])

    def test_filter_llm_still_works(self):
        with self._make_all_exist():
            variants = discover_variants(module_filter="llm")
        self.assertTrue(all(v.module == "llm" for v in variants))
        self.assertTrue(len(variants) > 0)


class TestBenchFnMapNoStt(unittest.TestCase):
    """The bench_fn_map inside run_variant_benchmark must not contain 'stt'."""

    def test_no_stt_in_bench_fn_map(self):
        # Read the source and check the bench_fn_map keys.
        # We can't easily call run_variant_benchmark (it imports benchmark_all),
        # so inspect the source code directly.
        import inspect
        from scripts.benchmark import benchmark_models
        source = inspect.getsource(benchmark_models.run_variant_benchmark)
        # The map should not have "stt" as a key
        self.assertNotIn('"stt"', source,
                         "bench_fn_map still references 'stt'")
        self.assertNotIn("'stt'", source,
                         "bench_fn_map still references 'stt'")


class TestRankVramFitNoStt(unittest.TestCase):
    """rank_vram_fit should work correctly without STT."""

    @staticmethod
    def _make_result(module: str, name: str, vram: int,
                     accuracy: float, status: str = "pass") -> VariantResult:
        v = ModelVariant(module, name, "fp16", f"fake/{name}", False, vram)
        r = VariantResult(variant=v)
        r.status = status
        r.accuracy_metric = accuracy
        r.accuracy_label = "Acc"
        r.vram_measured_mb = vram
        return r

    def test_ranking_without_stt(self):
        results = [
            self._make_result("llm", "LLM-A", 3000, 0.9),
            self._make_result("tts", "TTS-A", 100, 0.8),
            self._make_result("vad", "VAD-A", 0, 0.95),
        ]
        ranking = rank_vram_fit(results, budget_mb=12288)
        self.assertEqual(len(ranking), 1)

        rec = ranking[0]
        self.assertTrue(rec["fits"])
        module_names = [m["module"] for m in rec["models"]]
        self.assertNotIn("stt", module_names)
        # All provided modules should appear
        for mod in ("llm", "tts", "vad"):
            self.assertIn(mod, module_names)

    def test_ranking_total_vram_correct(self):
        results = [
            self._make_result("llm", "LLM", 4000, 0.9),
            self._make_result("tts", "TTS", 200, 0.8),
        ]
        ranking = rank_vram_fit(results, budget_mb=5000)
        self.assertEqual(ranking[0]["total_vram_mb"], 4200)
        self.assertTrue(ranking[0]["fits"])
        self.assertEqual(ranking[0]["headroom_mb"], 800)

    def test_ranking_priority_source_no_stt(self):
        """Verify the priority list in rank_vram_fit source has no 'stt'."""
        import inspect
        from scripts.benchmark import benchmark_models
        source = inspect.getsource(benchmark_models.rank_vram_fit)
        # Find the priority = [...] line
        for line in source.splitlines():
            if "priority" in line and "[" in line:
                self.assertNotIn('"stt"', line,
                                 "priority list still contains 'stt'")
                self.assertNotIn("'stt'", line,
                                 "priority list still contains 'stt'")


if __name__ == "__main__":
    unittest.main()
