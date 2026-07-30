"""Pure-Python coverage for the standalone entropy-P selection helpers."""

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


CAIEC_E2E = Path(__file__).resolve().parents[2] / "caiec-script" / "caiec_e2e"
sys.path.insert(0, str(CAIEC_E2E))

import entropy_p_config as config


def _load_selector_module():
    """Load helpers without the selector's external benchmark dependencies."""
    runtime = types.ModuleType("compressai.runtime")
    runtime.build_runtime = None
    codecs = types.ModuleType("compressai.runtime.codecs")
    codecs.GpuPackedEntropyCodec = None
    runtime_config = types.ModuleType("compressai.runtime.config")
    runtime_config.RuntimeConfig = None
    zoo = types.ModuleType("compressai.zoo")
    zoo.bmshj2018_factorized = None
    original_modules = {
        name: sys.modules.get(name)
        for name in ("compressai.runtime", "compressai.runtime.codecs", "compressai.runtime.config", "compressai.zoo")
    }
    sys.modules.update({
        "compressai.runtime": runtime,
        "compressai.runtime.codecs": codecs,
        "compressai.runtime.config": runtime_config,
        "compressai.zoo": zoo,
    })
    try:
        spec = importlib.util.spec_from_file_location(
            "select_bm_factorized_entropy_p_test", CAIEC_E2E / "select_bm_factorized_entropy_p.py"
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in original_modules.items():
            if original is None:
                del sys.modules[name]
            else:
                sys.modules[name] = original


SELECTOR = _load_selector_module()


def test_empty_results_write_and_load_json(tmp_path):
    path = tmp_path / "nested" / "entropy-p.json"
    results = config.empty_results({"device": "cpu"})

    assert config.load_results(path) == config.empty_results()
    config.write_results(path, results)

    assert config.load_results(path) == results


def test_selection_success_round_trip():
    results = config.empty_results()
    config.upsert_selection(results, "dataset-a", 4, "selected", 32)

    assert config.selection_is_successful(results, "dataset-a", 4)
    assert config.get_selected_entropy_p(results, "dataset-a", 4) == 32


def test_no_feasible_selection_is_complete_but_cannot_be_used():
    results = config.empty_results()
    config.upsert_selection(results, "dataset-a", 4, "no_feasible_p", None)

    assert config.selection_is_successful(results, "dataset-a", 4)
    with pytest.raises(ValueError, match="No feasible entropy P"):
        config.get_selected_entropy_p(results, "dataset-a", 4)


@pytest.mark.parametrize(
    ("results", "error"),
    [
        (config.empty_results(), "No entropy-P selection"),
        (
            {
                **config.empty_results(),
                "selections": {
                    "dataset-a": {
                        "4": {
                            "ans_variant": "other",
                            "status": "selected",
                            "selected_p": 8,
                        }
                    }
                },
            },
            "Selection variant",
        ),
    ],
)
def test_get_selected_entropy_p_rejects_missing_and_wrong_variant(results, error):
    with pytest.raises(ValueError, match=error):
        config.get_selected_entropy_p(results, "dataset-a", 4)


def test_select_entropy_p_uses_smallest_feasible_non_monotonic_candidate():
    rows = [
        {"P": 2, "ans_variant": "warp_smem", "inference_ms": 11.0, "codec_ms": 10.0, "measurement_error": None},
        {"P": 64, "ans_variant": "warp_smem", "inference_ms": 9.0, "codec_ms": 10.0, "measurement_error": None},
        {"P": 32, "ans_variant": "warp_smem", "inference_ms": 8.0, "codec_ms": 10.0, "measurement_error": None},
        {"P": 128, "ans_variant": "other", "inference_ms": 1.0, "codec_ms": 10.0, "measurement_error": None},
    ]

    assert config.select_entropy_p(rows) == 32


def test_select_entropy_p_ignores_measurement_errors_and_returns_none_when_needed():
    rows = [
        {"P": 64, "ans_variant": "warp_smem", "inference_ms": 1.0, "codec_ms": 10.0, "measurement_error": "RuntimeError: failed"},
        {"P": 2, "ans_variant": "warp_smem", "inference_ms": 11.0, "codec_ms": 10.0, "measurement_error": None},
    ]

    assert config.select_entropy_p(rows) is None


def test_fixed_batch_pads_final_batch():
    blocks = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    batch = SELECTOR.fixed_batch(blocks, batch_size=4, pad_value=-1.0)

    assert torch.equal(
        batch, torch.tensor([[1.0, 2.0], [3.0, 4.0], [-1.0, -1.0], [-1.0, -1.0]])
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1, 4,16", [1, 4, 16]), (" 2 ,, 8 ", [2, 8])],
)
def test_parse_csv_ints(value, expected):
    assert SELECTOR.parse_csv_ints(value) == expected


@pytest.mark.parametrize("value", ["", " , ", "1,nope"])
def test_parse_csv_ints_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        SELECTOR.parse_csv_ints(value)
