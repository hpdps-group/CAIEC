"""Pure-Python tests for DCAE session-pipeline batch helpers."""

import importlib.abc
import importlib.machinery
import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest
import torch


CAIEC_E2E = Path(__file__).resolve().parents[2] / "caiec-script" / "caiec_e2e"
PIPELINE_BENCHMARK = CAIEC_E2E / "bench_dcae_pipeline_session_caiec.py"
SOURCE_BENCHMARK = CAIEC_E2E / "bench_dcae_caiec.py"


class _SourceBenchmarkLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.DATASETS = []
        module.QUALITIES = []


def _load_benchmark_module():
    """Import helper definitions using stubs, without source benchmark or TRT."""
    models = types.ModuleType("compressai.models")
    models.DCAE = None
    runtime = types.ModuleType("compressai.runtime")
    runtime.build_runtime = None
    ans_pipeline = types.ModuleType("compressai.runtime.ans_pipeline")
    ans_pipeline.ans_gpu = None
    codecs = types.ModuleType("compressai.runtime.codecs")
    codecs.GpuPackedEntropyCodec = None
    stream_pipeline = types.ModuleType("compressai.runtime.dcae_stream_pipeline")
    stream_pipeline.compress_dcae_pipelined_session = None
    stream_pipeline.create_dcae_decode_sessions = None
    stream_pipeline.create_dcae_encode_sessions = None
    stream_pipeline.decompress_dcae_pipelined_session = None
    module_names = (
        "compressai.models", "compressai.runtime", "compressai.runtime.ans_pipeline",
        "compressai.runtime.codecs", "compressai.runtime.dcae_stream_pipeline",
    )
    originals = {name: sys.modules.get(name) for name in module_names}
    original_spec = importlib.util.spec_from_file_location

    def source_stub(name, location, *args, **kwargs):
        if Path(location) == SOURCE_BENCHMARK:
            return importlib.machinery.ModuleSpec(name, _SourceBenchmarkLoader())
        return original_spec(name, location, *args, **kwargs)

    sys.modules.update({
        "compressai.models": models,
        "compressai.runtime": runtime,
        "compressai.runtime.ans_pipeline": ans_pipeline,
        "compressai.runtime.codecs": codecs,
        "compressai.runtime.dcae_stream_pipeline": stream_pipeline,
    })
    importlib.util.spec_from_file_location = source_stub
    try:
        spec = original_spec("bench_dcae_pipeline_session_caiec_test", PIPELINE_BENCHMARK)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        importlib.util.spec_from_file_location = original_spec
        for name, original in originals.items():
            if original is None:
                del sys.modules[name]
            else:
                sys.modules[name] = original


BENCHMARK = _load_benchmark_module()


def test_padded_batches_pads_only_final_batch():
    batches = BENCHMARK.padded_batches(torch.tensor([[1], [2], [3]], dtype=torch.int16), 2, -1)

    assert [batch[:, 0].tolist() for batch in batches] == [[1, 2], [3, -1]]
    assert batches[1].dtype is torch.int16


@pytest.mark.parametrize("batch_size", [0, -1])
def test_padded_batches_rejects_nonpositive_batch_size(batch_size):
    with pytest.raises(ValueError, match="batch_size must be positive"):
        BENCHMARK.padded_batches(torch.ones(1, 1), batch_size, 0)


def test_select_target_batches_repeats_then_truncates():
    source = [torch.tensor([value]) for value in range(3)]

    selected = BENCHMARK.select_target_batches(source, 8)

    assert [batch.item() for batch in selected] == [0, 1, 2, 0, 1, 2, 0, 1]
    assert math.ceil(8 / len(source)) == 3


@pytest.mark.parametrize("target", [0, -3])
def test_select_target_batches_rejects_nonpositive_target(target):
    with pytest.raises(ValueError, match="target_batches must be positive"):
        BENCHMARK.select_target_batches([torch.tensor([1])], target)


def test_select_target_batches_rejects_empty_source():
    with pytest.raises(ValueError, match="cannot repeat empty batches"):
        BENCHMARK.select_target_batches([], 1)


def test_total_batch_bytes_and_stats():
    batches = [torch.zeros(2, 3, dtype=torch.int16), torch.zeros(1, 3, dtype=torch.int16)]

    assert BENCHMARK.total_batch_bytes(batches) == 18
    assert BENCHMARK.stats([3.0, 1.0, 2.0], "latency_ms") == {
        "latency_ms_mean": 2.0, "latency_ms_min": 1.0, "latency_ms_max": 3.0
    }
