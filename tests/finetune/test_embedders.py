"""Phase 1B embedders: ``make_embedder`` (B1), ``OnnxEmbedder`` (B2), ``EngineEmbedder`` (B3).

Strategy: the ONNX embedder runs a real onnxruntime session on a tiny mean-RGB model, so the expected
output is computed independently from the spec's preprocessing (resize, /255, ImageNet mean/std, RGB).
The Engine embedder talks to an ``httpx.MockTransport`` that decodes every uploaded JPEG and answers
with its mean colour, so order, batching and reordering are checked on content, not on call counts alone.
``sparrow_finetune.embedders`` is imported inside each test so collection succeeds before it exists.
"""

from __future__ import annotations

import email
import io
import json
import random

import numpy as np
import pytest
from PIL import Image

from .dinov3_helpers import sha256_file, write_mean_rgb_onnx

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@pytest.fixture
def onnx_model(tmp_path):
    return write_mean_rgb_onnx(tmp_path / "mean_rgb.onnx")


def solid(rgb, size=(40, 30)):
    return Image.new("RGB", size, tuple(rgb))


# --------------------------------------------------------------------------- B1 make_embedder


def test_b1_make_embedder_onnx_spec_builds_onnx_embedder(onnx_model):
    from sparrow_finetune.embedders import OnnxEmbedder, make_embedder

    emb = make_embedder(f"onnx:{onnx_model}")
    assert isinstance(emb, OnnxEmbedder)
    assert emb.model_hash == sha256_file(onnx_model)


def test_b1_make_embedder_engine_spec_builds_engine_embedder():
    from sparrow_finetune.embedders import EngineEmbedder, make_embedder

    emb = make_embedder("engine:http://engine.local:8080#dinov3-vitb16")
    assert isinstance(emb, EngineEmbedder)
    assert emb.model_hash == "engine:dinov3-vitb16"


@pytest.mark.parametrize("spec", ["", "   ", "tensorflow:/x.pb", "onnx", "nonsense"])
def test_b1_make_embedder_rejects_empty_or_unknown_spec_naming_env_var(spec):
    from sparrow_finetune.embedders import make_embedder

    with pytest.raises(ValueError, match="SPARROW_FT_EMBEDDER"):
        make_embedder(spec)


# --------------------------------------------------------------------------- B2 OnnxEmbedder


def test_b2_onnx_embedder_returns_normalised_mean_rgb_float32(onnx_model):
    from sparrow_finetune.embedders import OnnxEmbedder

    colours = [(255, 0, 0), (0, 128, 255), (10, 200, 30), (90, 90, 90), (0, 0, 0)]
    emb = OnnxEmbedder(onnx_model, input_size=32, batch_size=2)
    out = emb.embed([solid(c) for c in colours])

    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.shape == (len(colours), 3)
    assert emb.dim == 3
    expected = ((np.array(colours, dtype=np.float32) / 255.0) - MEAN) / STD
    np.testing.assert_allclose(out, expected, atol=2e-3)


def test_b2_onnx_embedder_is_deterministic_and_converts_to_rgb(onnx_model):
    from sparrow_finetune.embedders import OnnxEmbedder

    emb = OnnxEmbedder(onnx_model, input_size=16)
    images = [solid((200, 20, 20)).convert("L"), solid((5, 6, 7)).convert("RGBA")]
    a = emb.embed(images)
    b = emb.embed(images)
    assert a.shape == (2, 3)
    np.testing.assert_array_equal(a, b)


def test_b2_onnx_embedder_flattens_rank4_output(tmp_path):
    from sparrow_finetune.embedders import OnnxEmbedder

    model = write_mean_rgb_onnx(tmp_path / "pooled.onnx", flatten=False)  # output [N,3,1,1]
    emb = OnnxEmbedder(model, input_size=16)
    out = emb.embed([solid((255, 255, 255)), solid((0, 0, 0))])
    assert out.shape == (2, 3)
    assert emb.dim == 3
    np.testing.assert_allclose(out[0], (1.0 - MEAN) / STD, atol=2e-3)


def test_b2_onnx_embedder_model_hash_is_file_sha256(onnx_model):
    from sparrow_finetune.embedders import OnnxEmbedder

    assert OnnxEmbedder(onnx_model).model_hash == sha256_file(onnx_model)


def test_b2_onnx_embedder_missing_file_raises_file_not_found(tmp_path):
    from sparrow_finetune.embedders import OnnxEmbedder

    with pytest.raises(FileNotFoundError):
        OnnxEmbedder(tmp_path / "absent.onnx")


# --------------------------------------------------------------------------- B3 EngineEmbedder


def parse_multipart(request) -> list[tuple[str, bytes]]:
    """``[(field name, payload bytes)]`` of a multipart/form-data request, in order."""
    body = request.read()
    ctype = request.headers["content-type"].encode()
    msg = email.message_from_bytes(b"Content-Type: " + ctype + b"\r\n\r\n" + body)
    assert msg.is_multipart(), request.headers["content-type"]
    return [
        (part.get_param("name", header="content-disposition"), part.get_payload(decode=True))
        for part in msg.get_payload()
    ]


class FakeEngine:
    """Answers ``/v1/embed/batch`` with each image's mean RGB (0..255) as its embedding."""

    def __init__(self, mode: str = "ok", shuffle: bool = True) -> None:
        self.mode, self.shuffle = mode, shuffle
        self.requests: list[dict] = []

    def __call__(self, request):
        import httpx

        parts = parse_multipart(request)
        self.requests.append(
            {
                "path": request.url.path,
                "params": dict(request.url.params),
                "fields": [name for name, _ in parts],
            }
        )
        if self.mode == "5xx":
            return httpx.Response(503, text="engine overloaded")
        results = []
        for i, (_name, payload) in enumerate(parts):
            with Image.open(io.BytesIO(payload)) as im:
                assert im.format == "JPEG"
                vec = np.asarray(im.convert("RGB"), dtype=np.float64).mean(axis=(0, 1))
            results.append({"index": i, "embedding": [float(v) for v in vec]})
        if self.mode == "wrong_count":
            results = results[:-1]
        elif self.mode == "duplicate":
            results[-1] = {"index": 0, "embedding": results[-1]["embedding"]}
        if self.shuffle:
            random.Random(len(self.requests)).shuffle(results)
        return httpx.Response(200, content=json.dumps({"results": results}).encode(),
                              headers={"content-type": "application/json"})


def engine_embedder(engine: FakeEngine, **kw):
    import httpx

    from sparrow_finetune.embedders import EngineEmbedder

    client = httpx.Client(transport=httpx.MockTransport(engine))
    return EngineEmbedder("http://engine.test", "dinov3-test", client=client, **kw)


def coded_images(n: int) -> list[Image.Image]:
    """Image ``i`` encodes its position as (red, green) = (30*(i//10), 25*(i%10)); robust to JPEG."""
    return [solid((30 * (i // 10), 25 * (i % 10), 50), size=(24, 24)) for i in range(n)]


def decode_positions(out: np.ndarray) -> list[int]:
    return [int(round(r / 30.0)) * 10 + int(round(g / 25.0)) for r, g, _b in out]


def test_b3_engine_embedder_posts_images_fields_with_model_and_store_false():
    engine = FakeEngine()
    emb = engine_embedder(engine)
    out = emb.embed(coded_images(3))

    assert out.shape == (3, 3)
    (req,) = engine.requests
    assert req["path"] == "/v1/embed/batch"
    assert req["params"].get("model") == "dinov3-test"
    assert req["params"].get("store") == "false"
    assert req["fields"] == ["images"] * 3
    assert emb.model_hash == "engine:dinov3-test"


def test_b3_engine_embedder_batches_70_images_and_reorders_by_index():
    engine = FakeEngine(shuffle=True)
    emb = engine_embedder(engine)
    out = emb.embed(coded_images(70))

    sizes = [len(r["fields"]) for r in engine.requests]
    assert sum(sizes) == 70
    assert len(sizes) >= 3 and max(sizes) <= 32, sizes
    assert out.dtype == np.float32 and out.shape == (70, 3)
    assert emb.dim == 3
    assert decode_positions(out) == list(range(70))


def test_b3_engine_embedder_batch_size_is_capped_at_32():
    engine = FakeEngine()
    emb = engine_embedder(engine, batch_size=100)
    emb.embed(coded_images(40))
    assert max(len(r["fields"]) for r in engine.requests) <= 32


def test_b3_engine_embedder_raises_embedder_error_with_status_on_5xx():
    from sparrow_finetune.embedders import EmbedderError

    emb = engine_embedder(FakeEngine(mode="5xx"))
    with pytest.raises(EmbedderError) as info:
        emb.embed(coded_images(2))
    assert getattr(info.value, "status_code", None) == 503 or "503" in str(info.value)


@pytest.mark.parametrize("mode", ["wrong_count", "duplicate"])
def test_b3_engine_embedder_rejects_wrong_count_or_duplicate_index(mode):
    from sparrow_finetune.embedders import EmbedderError

    emb = engine_embedder(FakeEngine(mode=mode))
    with pytest.raises(EmbedderError):
        emb.embed(coded_images(4))
