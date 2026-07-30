"""Pure-Python tests for DCAE entropy-P result configuration."""

import importlib.util
from pathlib import Path

import pytest


CAIEC_E2E = Path(__file__).resolve().parents[2] / "caiec-script" / "caiec_e2e"
CONFIG_PATH = CAIEC_E2E / "dcae_entropy_p_config.py"


@pytest.fixture(scope="module")
def config():
    spec = importlib.util.spec_from_file_location("dcae_entropy_p_config_test", CONFIG_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_empty_results_has_dcae_model_schema_and_defaults(config):
    results = config.empty_results({"device": "cpu"}, {"profile": "mix"})

    assert results == {
        "model": "dcae",
        "schema_version": 1,
        "environment": {"device": "cpu"},
        "precision_profile": {"profile": "mix"},
        "selections": {},
        "measurements": [],
    }
    assert config.RESULTS_SCHEMA["properties"]["model"]["const"] == "dcae"


def test_write_and_load_results_round_trip(tmp_path, config):
    path = tmp_path / "nested" / "dcae-entropy-p.json"
    results = config.empty_results({"gpu": "test"})

    assert config.load_results(path) == config.empty_results()
    config.write_results(path, results)

    assert config.load_results(path) == results


@pytest.mark.parametrize(
    "invalid",
    [
        {},
        {"model": "other", "schema_version": 1, "environment": {}, "precision_profile": {}, "selections": {}, "measurements": []},
        {"model": "dcae", "schema_version": 2, "environment": {}, "precision_profile": {}, "selections": {}, "measurements": []},
        {"model": "dcae", "schema_version": 1, "environment": [], "precision_profile": {}, "selections": {}, "measurements": []},
    ],
)
def test_load_rejects_non_dcae_or_invalid_schema(tmp_path, config, invalid):
    path = tmp_path / "invalid.json"
    config.write_results(path, invalid)

    with pytest.raises(ValueError, match="Unsupported DCAE entropy-P result schema"):
        config.load_results(path)


def test_selection_records_complete_selected_and_no_feasible(config):
    results = config.empty_results()
    config.upsert_selection(results, "dataset", 1, 16)

    assert config.selection_is_complete(results, "dataset", 1)
    assert results["selections"]["dataset"]["1"] == {
        "ans_variant": "warp_smem", "status": "selected", "selected_p": 16
    }

    config.upsert_selection(results, "dataset", 2, None)
    assert config.selection_is_complete(results, "dataset", 2)
    assert results["selections"]["dataset"]["2"]["status"] == "no_feasible_p"


def test_select_entropy_p_returns_smallest_feasible_candidate(config):
    measurements = [
        {"P": 64, "ans_variant": "warp_smem", "inference_ms": 5.0, "codec_ms": 5.0, "measurement_error": None},
        {"P": 16, "ans_variant": "warp_smem", "inference_ms": 6.0, "codec_ms": 7.0, "measurement_error": None},
        {"P": 8, "ans_variant": "warp_smem", "inference_ms": 8.0, "codec_ms": 7.0, "measurement_error": None},
        {"P": 4, "ans_variant": "other", "inference_ms": 1.0, "codec_ms": 2.0, "measurement_error": None},
    ]

    assert config.select_entropy_p(measurements) == 16


def test_select_entropy_p_returns_none_when_no_candidate_is_feasible(config):
    measurements = [
        {"P": 4, "ans_variant": "warp_smem", "inference_ms": 3.0, "codec_ms": 2.0, "measurement_error": None},
        {"P": 8, "ans_variant": "warp_smem", "inference_ms": 1.0, "codec_ms": 5.0, "measurement_error": "failed"},
    ]

    assert config.select_entropy_p(measurements) is None
