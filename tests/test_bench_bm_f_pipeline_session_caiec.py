"""Pure-Python tests for CAIEC's session-pipeline batch helpers."""

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
PIPELINE_BENCHMARK = CAIEC_E2E / "bench_bm-f_pipeline_session_caiec.py"
SOURCE_BENCHMARK = CAIEC_E2E / "bench_bm-f_caiec.py"


class _SourceBenchmarkLoader(importlib.abc.Loader):
    """Supply only the source-benchmark values needed during module import."""

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.DATASETS = []
        module.QUALITIES = []


def _load_benchmark_module():
    """Load helper functions without CompressAI or the external benchmark."""
    runtime = types.ModuleType("compressai.runtime")
    runtime.build_runtime = None
    ans_pipeline = types.ModuleType("compressai.runtime.ans_pipeline")
    ans_pipeline.PipelinedAnsDecoder = None
    ans_pipeline.PipelinedAnsEncoder = None
    ans_pipeline.ans_gpu = None
    codecs = types.ModuleType("compressai.runtime.codecs")
    codecs.GpuPackedEntropyCodec = None
    runtime_config = types.ModuleType("compressai.runtime.config")
    runtime_config.RuntimeConfig = None
    stream_pipeline = types.ModuleType("compressai.runtime.stream_pipeline")
    stream_pipeline.compress_pipelined_session = None
    stream_pipeline.decompress_pipelined_session = None
    zoo = types.ModuleType("compressai.zoo")
    zoo.bmshj2018_factorized = None
    module_names = (
        "compressai.runtime",
        "compressai.runtime.ans_pipeline",
        "compressai.runtime.codecs",
        "compressai.runtime.config",
        "compressai.runtime.stream_pipeline",
        "compressai.zoo",
    )
    original_modules = {name: sys.modules.get(name) for name in module_names}
    original_spec = importlib.util.spec_from_file_location

    def source_stub(name, location, *args, **kwargs):
        if Path(location) == SOURCE_BENCHMARK:
            return importlib.machinery.ModuleSpec(name, _SourceBenchmarkLoader())
        return original_spec(name, location, *args, **kwargs)

    sys.modules.update({
        "compressai.runtime": runtime,
        "compressai.runtime.ans_pipeline": ans_pipeline,
        "compressai.runtime.codecs": codecs,
        "compressai.runtime.config": runtime_config,
        "compressai.runtime.stream_pipeline": stream_pipeline,
        "compressai.zoo": zoo,
    })
    importlib.util.spec_from_file_location = source_stub
    try:
        spec = original_spec("bench_bm_f_pipeline_session_caiec_test", PIPELINE_BENCHMARK)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        importlib.util.spec_from_file_location = original_spec
        for name, original in original_modules.items():
            if original is None:
                del sys.modules[name]
            else:
                sys.modules[name] = original


BENCHMARK = _load_benchmark_module()


def test_padded_batches_pads_only_the_final_batch():
    blocks = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int16)

    batches = BENCHMARK.padded_batches(blocks, batchsize=2, pad_value=-1)

    assert len(batches) == 2
    assert torch.equal(batches[0], torch.tensor([[1, 2], [3, 4]], dtype=torch.int16))
    assert torch.equal(batches[1], torch.tensor([[5, 6], [-1, -1]], dtype=torch.int16))


def test_select_target_batches_truncates_repeat_cycle_at_eight():
    source = [torch.tensor([index]) for index in range(3)]

    workload = BENCHMARK.select_target_batches(source, target_batches=8)

    assert [batch.item() for batch in workload] == [0, 1, 2, 0, 1, 2, 0, 1]


def test_final_workload_repeat_cycle_and_bytes_are_exact():
    source_batches = BENCHMARK.padded_batches(torch.arange(5, dtype=torch.int32).reshape(5, 1), 2, -1)
    target_batches = 8

    workload = BENCHMARK.select_target_batches(source_batches, target_batches)

    assert math.ceil(target_batches / len(source_batches)) == 3
    assert [batch[:, 0].tolist() for batch in workload] == [
        [0, 1], [2, 3], [4, -1], [0, 1], [2, 3], [4, -1], [0, 1], [2, 3]
    ]
    assert BENCHMARK.total_batch_bytes(workload) == 8 * 2 * torch.tensor([], dtype=torch.int32).element_size()


@pytest.mark.parametrize("target_batches", [0, -1])
def test_select_target_batches_rejects_invalid_target(target_batches):
    with pytest.raises(ValueError, match="target_batches must be positive"):
        BENCHMARK.select_target_batches([torch.tensor([1])], target_batches)


def test_select_target_batches_rejects_empty_source():
    with pytest.raises(ValueError, match="cannot repeat an empty batch list"):
        BENCHMARK.select_target_batches([], 1)
