# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from compressai.runtime import ans_pipeline
from compressai.runtime.ans_pipeline import PipelinedAnsDecoder
from compressai.runtime.config import RuntimeConfig
from compressai.runtime.dcae_stream_pipeline import (
    DCAEEncodeSessions,
    _check_engine,
    _shape3,
    _validate_blocks,
    create_dcae_decode_sessions,
)
from compressai.runtime.engines.dcae_engine import DCAEEngine


DTYPE_FIELDS = (
    "h_z_s1_input_dtype",
    "h_z_s2_input_dtype",
    "dt_ca_input_dtypes",
)
SLICE_DTYPE_FIELDS = (
    "dt_ca_input_dtypes",
    "cc_mean_input_dtypes",
    "cc_scale_input_dtypes",
    "lrp_input_dtypes",
)


def _make_engine(num_slices=2, **kwargs):
    net = SimpleNamespace(
        num_slices=num_slices,
        max_support_slices=-1,
        dt=torch.zeros(2, 3),
        M=4,
    )
    runners = {name: object() for name in ("ga", "ha", "h_z_s1", "h_z_s2", "gs")}
    for index in range(num_slices):
        for prefix in ("dt_cross_attention", "cc_mean", "cc_scale", "lrp_transforms"):
            runners[f"{prefix}_{index}"] = object()
    return DCAEEngine(net, SimpleNamespace(), runners, **kwargs)


def test_runtime_config_dcae_defaults_are_none():
    config = RuntimeConfig()
    for field in DTYPE_FIELDS:
        assert getattr(config, field) is None


@pytest.mark.parametrize("field", SLICE_DTYPE_FIELDS)
def test_dcae_engine_normalizes_slice_dtypes(field):
    engine = _make_engine(**{field: [torch.float16, None]})
    assert getattr(engine, field) == [torch.float16, None]

    legacy = _make_engine(**{field: (None,)})
    assert getattr(legacy, field) == [None, None]


@pytest.mark.parametrize("field", SLICE_DTYPE_FIELDS)
@pytest.mark.parametrize(
    ("value", "error"),
    [
        (torch.float16, TypeError),
        ([torch.float16], ValueError),
        ([torch.float16, "fp32"], TypeError),
    ],
)
def test_dcae_engine_rejects_invalid_slice_dtypes(field, value, error):
    with pytest.raises(error, match=field):
        _make_engine(**{field: value})


def test_get_y_shape_uses_metadata_and_fallback():
    engine = _make_engine()
    z_hat = torch.empty(1, 1, 3, 5)

    state = [{"size_hw": torch.Size([7, 9])}, {"size_hw": (7, 9)}]
    assert engine._get_y_shape({"state": state}, z_hat) == (7, 9)
    assert engine._get_y_shape({}, z_hat) == (12, 20)
    assert engine._get_y_shape({"state": None}, z_hat) == (12, 20)


@pytest.mark.parametrize(
    "state",
    [
        {},
        [{"size_hw": (2, 3)}],
        [{"size_hw": (2, 3)}, None],
        [{"size_hw": (2,)}, {"size_hw": (2,)}],
        [{"size_hw": (2, 0)}, {"size_hw": (2, 0)}],
        [{"size_hw": (True, 3)}, {"size_hw": (True, 3)}],
        [{"size_hw": (2, 3)}, {"size_hw": (2, 4)}],
    ],
)
def test_get_y_shape_rejects_bad_metadata(state):
    with pytest.raises(ValueError, match=r"pack\['y'\]\['state'\]"):
        _make_engine()._get_y_shape({"state": state}, torch.empty(1, 1, 2, 2))


def test_decoder_gaussian_conditional_factory_forwards_without_cuda(monkeypatch):
    gc = SimpleNamespace(
        _quantized_cdf=object(),
        _cdf_length=object(),
        _offset=object(),
    )
    sentinel = object()
    captured = {}

    def fake_constructor(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(ans_pipeline, "PipelinedAnsDecoder", fake_constructor)
    result = PipelinedAnsDecoder.from_gaussian_conditional(
        gc, B=2, N=12, K=3, chunk_len=4, HW=6
    )

    assert result is sentinel
    assert captured == {
        "cdfs": gc._quantized_cdf,
        "cdf_sizes": gc._cdf_length,
        "offsets": gc._offset,
        "B": 2,
        "N": 12,
        "K": 3,
        "chunk_len": 4,
        "HW": 6,
        "fast_idx_is_channel": False,
    }


def test_ans_factories_select_explicit_index_mode(monkeypatch):
    model = SimpleNamespace(
        _quantized_cdf=object(), _cdf_length=object(), _offset=object()
    )
    captured = []

    def fake_constructor(**kwargs):
        captured.append(kwargs["fast_idx_is_channel"])
        return object()

    encoder_class = ans_pipeline.PipelinedAnsEncoder
    monkeypatch.setattr(ans_pipeline, "PipelinedAnsEncoder", fake_constructor)
    encoder_class.from_entropy_bottleneck(model, 1, 4)
    encoder_class.from_gaussian_conditional(model, 1, 4)
    assert captured == [True, False]


def test_ans_launch_validation_is_not_assert_based():
    encoder = object.__new__(ans_pipeline.PipelinedAnsEncoder)
    encoder.B, encoder.N = 1, 4
    encoder._cdfs = torch.empty(1)
    with pytest.raises(ValueError, match="CUDA"):
        encoder.launch(torch.zeros(1, 4, dtype=torch.int32), torch.zeros(1, 4, dtype=torch.int32))

    decoder = object.__new__(ans_pipeline.PipelinedAnsDecoder)
    decoder.B, decoder.N, decoder.K = 1, 4, 2
    decoder._cdfs = torch.empty(1)
    with pytest.raises(ValueError, match="CUDA"):
        decoder.launch(
            torch.zeros(8, dtype=torch.uint8),
            torch.zeros(1, 2, dtype=torch.uint32),
            32,
            torch.zeros(1, 4, dtype=torch.int32),
        )


@pytest.mark.parametrize("shape", [(1, 2), (1, 2, 0), "123", (1, -2, 3)])
def test_shape3_rejects_invalid_shapes(shape):
    with pytest.raises(ValueError):
        _shape3("shape", shape)


def test_pipeline_empty_pack_and_block_validation_are_cpu_safe():
    engine = _make_engine()
    engine.codec = SimpleNamespace(
        gaussian_conditional=object(), eb=object(), use_warp=True, _variant="warp_smem"
    )
    decoded = create_dcae_decode_sessions(engine, [])
    assert decoded.z == []
    assert decoded.y == []
    assert decoded.batch_size == 0

    sessions = DCAEEncodeSessions([], [], 1, (1, 1, 1), (2, 1, 1))
    _validate_blocks([], sessions)
    with pytest.raises(ValueError, match="BCHW CUDA tensor"):
        _validate_blocks([torch.empty(1, 1, 1, 1)], DCAEEncodeSessions([object()], [[]], 1, (1, 1, 1), (2, 1, 1)))


@pytest.mark.parametrize(
    "pack",
    [
        None,
        {},
        {"y": {}, "z": {}},
        {"y": {"strings": []}, "z": {"strings": object(), "state": {"size_hw": (1, 1)}}},
        {"y": {"strings": [object(), object()]}, "z": {"strings": object()}},
    ],
)
def test_decode_session_factory_rejects_malformed_packs(pack):
    engine = _make_engine()
    engine.codec = SimpleNamespace(
        gaussian_conditional=object(), eb=object(), use_warp=True, _variant="warp_smem"
    )
    with pytest.raises(ValueError):
        create_dcae_decode_sessions(engine, [pack])


def test_pipeline_engine_validation():
    with pytest.raises(TypeError, match="DCAEEngine"):
        _check_engine(object())

    engine = _make_engine()
    engine.codec = None
    with pytest.raises(ValueError, match="codecs"):
        _check_engine(engine)

    engine.codec = SimpleNamespace(gaussian_conditional=object(), _variant="warp")
    with pytest.raises(ValueError, match="warp_smem"):
        _check_engine(engine)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not hasattr(ans_pipeline.ans_gpu, "ans_create_session"),
    reason="CUDA ans_gpu session extension is unavailable",
)
def test_small_gaussian_conditional_session_roundtrip():
    create_doc = getattr(ans_pipeline.ans_gpu.ans_create_session, "__doc__", "") or ""
    decode_doc = getattr(ans_pipeline.ans_gpu.ans_decode_create_session, "__doc__", "") or ""
    if "fast_idx_is_channel" not in create_doc or "fast_idx_is_channel" not in decode_doc:
        pytest.skip("CUDA ans_gpu extension must be rebuilt for explicit index mode")

    from compressai.entropy_models import GaussianConditional

    gc = GaussianConditional(None)
    gc.update_scale_table(torch.tensor([1.0]))
    symbols = torch.tensor([[-1, 0, 1, 0]], device="cuda", dtype=torch.int32)
    indexes = torch.zeros_like(symbols)
    encoder = ans_pipeline.PipelinedAnsEncoder.from_gaussian_conditional(gc, 1, 4, 2)
    encoder.launch(symbols, indexes)
    tight = encoder.finalize()

    chunk_len = int(tight.chunk_len_cpu.reshape(-1)[0])
    decoder = PipelinedAnsDecoder.from_gaussian_conditional(
        gc, 1, 4, int(tight.max_rounds_u32.shape[1]), chunk_len, 4
    )
    decoder.launch(
        tight.packed,
        tight.max_rounds_u32,
        int(tight.header_bytes_cpu.reshape(-1)[0]),
        indexes,
    )
    assert torch.equal(decoder.finalize(), symbols)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not hasattr(ans_pipeline.ans_gpu, "ans_create_session"),
    reason="CUDA ans_gpu session extension is unavailable",
)
def test_gaussian_conditional_session_interoperates_with_varying_indexes():
    from compressai.entropy_models import GaussianConditional
    from compressai.entropy_models.entropy_models import (
        _gpu_ans_decode_with_indexes_warp,
        _gpu_ans_encode_with_indexes_warp,
    )

    create_doc = getattr(ans_pipeline.ans_gpu.ans_create_session, "__doc__", "") or ""
    decode_doc = getattr(ans_pipeline.ans_gpu.ans_decode_create_session, "__doc__", "") or ""
    if "fast_idx_is_channel" not in create_doc or "fast_idx_is_channel" not in decode_doc:
        pytest.skip("CUDA ans_gpu extension must be rebuilt for explicit index mode")

    gc = GaussianConditional([0.5, 1.0, 2.0, 4.0]).cuda()
    gc.update()
    batch, cdfs, spatial, parallelism = 2, 4, 8, 2
    symbols = torch.randint(
        -4, 5, (batch, cdfs * spatial), device="cuda", dtype=torch.int32
    )
    indexes = torch.randint(0, cdfs, symbols.shape, device="cuda", dtype=torch.int32)

    encoder = ans_pipeline.PipelinedAnsEncoder.from_gaussian_conditional(
        gc, batch, symbols.shape[1], parallelism
    )
    encoder.launch(symbols, indexes)
    session_pack = encoder.finalize()
    decoded = _gpu_ans_decode_with_indexes_warp(
        session_pack, indexes, gc._quantized_cdf, gc._cdf_length, gc._offset
    ).reshape_as(symbols)
    assert torch.equal(decoded, symbols)

    regular_pack = _gpu_ans_encode_with_indexes_warp(
        symbols,
        indexes,
        gc._quantized_cdf,
        gc._cdf_length,
        gc._offset,
        parallelism=parallelism,
    )
    decoder = PipelinedAnsDecoder.from_gaussian_conditional(
        gc,
        batch,
        symbols.shape[1],
        int(regular_pack.max_rounds_u32.shape[1]),
        int(regular_pack.chunk_len_cpu.item()),
        spatial,
    )
    decoder.launch(
        regular_pack.packed,
        regular_pack.max_rounds_u32,
        int(regular_pack.header_bytes_cpu.item()),
        indexes,
    )
    assert torch.equal(decoder.finalize(), symbols)
