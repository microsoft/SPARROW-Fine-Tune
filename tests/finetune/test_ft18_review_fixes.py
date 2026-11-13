"""FT-18 regression pins (docs/ideas.md FT-18, backlog-sweep 2 review).

Strategy: each test drives a public path (pipeline run, ``count_near_duplicates``, the login
route, the activate route, the CLI) and asserts what a user or operator would observe.

- Near-duplicate counting must stay within a fixed memory budget however many train hashes
  there are (a subprocess with an address-space limit, so a regression fails instead of
  exhausting the host), and stay exact at the Hamming-distance boundary.
- Stock comparison: the stock model's argmax is over all of its classes (a prediction outside
  the shared labels is wrong); the raw "stock is better" result stays visible for any val size,
  but activation is only blocked at ``stock_min_val_crops`` (default 100; 0 = always) or more.
- Both classifier pipelines score the stock model on the same source-image crops.
- Pretrained timm weights already in the Hugging Face cache work with ``HF_HUB_OFFLINE``.
- The login limiter's client address: ``X-Forwarded-For`` hops with a port, bracketed IPv6 and
  IPv4-mapped IPv6 are normalised; malformed hops still stop the walk; uvicorn must not rewrite
  the peer itself (``proxy_headers=False``).
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from .dinov3_helpers import balanced_items, crop_image, make_context, snapshot_items
from .service_helpers import PASSWORD, add_user
from .test_backlog_pipelines import (
    CONV_BASE,
    STOCK_NAME,
    _conv_run,
    _dino_run,
    _save_timm_weights,
    textured_items,
    write_stock_package,
)
from .test_backlog_service import (
    STOCK_WORSE,
    _deployments,
    _login,
    _model,
    _proxied_client,
)

# --------------------------------------------------------------------------- near-duplicates

_NEAR_DUP_CHILD = textwrap.dedent(
    """
    import os, resource, sys
    import numpy as np
    from sparrow_finetune.pipelines.near_dup import count_near_duplicates

    rng = np.random.default_rng(0)
    top = np.iinfo(np.uint64).max
    train = rng.integers(0, top, size=200_000, dtype=np.uint64, endpoint=True)
    val = rng.integers(0, top, size=2_000, dtype=np.uint64, endpoint=True)
    bits = [63, 0, 31, 47, 5, 58, 12, 40]
    for d in range(8):  # Hamming distance d from one train hash
        mask = np.uint64(sum(1 << b for b in bits[:d]))
        val[d] = train[d * 1000] ^ mask
    val[1024] = train[123_456]                        # first row of the second val chunk
    val[1999] = train[-1] ^ np.uint64(1 << 63)        # last val row vs last train row, distance 1

    page = os.sysconf("SC_PAGE_SIZE")
    with open("/proc/self/statm") as fh:
        vsize = int(fh.read().split()[0]) * page
    budget = int(sys.argv[1])
    resource.setrlimit(resource.RLIMIT_AS, (vsize + budget, vsize + budget))
    print(count_near_duplicates(val, train, 4))
    """
)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_AS + /proc")
def test_ft18_near_dup_memory_is_bounded_for_large_train_sets():
    # 2k val x 200k train hashes. 64 B per pair would need ~13 GB per 1024-row val chunk and
    # even a full uint64 XOR row block is ~1.6 GB; 512 MiB is far more than a tiled count needs.
    proc = subprocess.run(
        [sys.executable, "-c", _NEAR_DUP_CHILD, str(512 * 2**20)],
        capture_output=True,
        text=True,
        timeout=600,
        env={"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "PATH": ""},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    # distances 0..4 of the planted rows, plus the two chunk-boundary rows; 5..7 are not near.
    assert int(proc.stdout.strip()) == 7


def test_ft18_near_dup_counts_exact_hamming_boundaries():
    from sparrow_finetune.pipelines.near_dup import count_near_duplicates

    top = np.uint64(np.iinfo(np.uint64).max)
    train = np.asarray([0, top, 1 << 63], dtype=np.uint64)
    val = np.asarray([0b1111, 0b11111, top ^ np.uint64(1), 0], dtype=np.uint64)
    # val[0]: 4 bits from 0; val[1]: 5 bits from 0; val[2]: 1 bit from all-ones; val[3]: exact.
    assert count_near_duplicates(val, train, 4) == 3
    assert count_near_duplicates(val, train, 0) == 1
    assert count_near_duplicates(np.asarray([top], dtype=np.uint64), train[:1], 63) == 0
    assert count_near_duplicates(np.asarray([top], dtype=np.uint64), train[:1], 64) == 1


# --------------------------------------------------------------------------- stock comparison


def test_ft18_stock_argmax_covers_every_stock_class():
    from sparrow_finetune.pipelines.stock_model import (
        comparison_metrics,
        matched_label_map,
    )

    mapping = matched_label_map(["cat", "dog"], ["Bird", "CAT", "Dog"])
    y = np.asarray([0, 0, 1, 1])
    # Rows 1 and 3: the stock model's top class is Bird, which the job does not train -> wrong.
    stock = np.asarray(
        [[0.1, 0.8, 0.1], [0.9, 0.06, 0.04], [0.1, 0.1, 0.8], [0.9, 0.03, 0.07]]
    )
    out = comparison_metrics(y, y.copy(), stock, mapping)
    assert out["stock_val_macro_f1"] == pytest.approx(2 / 3)
    assert out["stock_finetuned_val_macro_f1"] == 1.0
    assert out["stock_worse"] == 0.0


class _Ctx:
    def __init__(self, params):
        self.params = params
        self.labels = ["cat", "dog"]
        self.events = []

    def emit(self, stage, message, **fields):
        self.events.append((stage, message, fields))


class _Stock:
    id = "stock-v1"
    labels = ["Cat", "Dog"]

    def predict_proba(self, crops):
        y = np.asarray([i % 2 for i in range(len(crops))])
        return np.eye(2)[y]  # always right


def _stock_report(n_crops, min_val):
    from sparrow_finetune.pipelines.stock_model import stock_report

    ctx = _Ctx({"stock_model": "stock", "stock_min_val_crops": min_val})
    y = np.asarray([i % 2 for i in range(n_crops)])
    crops = [Image.new("RGB", (4, 4))] * n_crops
    metrics: dict = {}
    card = stock_report(ctx, _Stock(), {0: 0, 1: 1}, y, 1 - y, crops, metrics)
    return ctx, metrics, card


@pytest.mark.parametrize(
    "n_crops, min_val, enforced",
    [(40, 100, False), (99, 100, False), (100, 100, True), (40, 0, True)],
)
def test_ft18_stock_worse_stays_visible_and_records_enforcement(
    n_crops, min_val, enforced
):
    ctx, metrics, card = _stock_report(n_crops, min_val)
    assert metrics["stock_worse"] == 1.0 and card["stock_worse"] == 1.0
    assert any("stock" in m.lower() for s, m, _ in ctx.events if s == "evaluate")
    assert metrics.get("stock_min_val_crops") == float(min_val)
    assert card.get("min_val_crops") == min_val
    assert card.get("enforced") is enforced


@pytest.mark.parametrize("pipeline", ["dinov3", "conventional"])
def test_ft18_stock_min_val_crops_param(pipeline):
    from .test_backlog_pipelines import _trainer

    trainer = _trainer(pipeline)
    base = dict(CONV_BASE) if pipeline == "conventional" else {}
    assert trainer.validate_params(base).get("stock_min_val_crops") == 100
    zero = trainer.validate_params({**base, "stock_min_val_crops": 0})
    assert zero.get("stock_min_val_crops") == 0
    with pytest.raises(ValueError, match="stock_min_val_crops"):
        trainer.validate_params({**base, "stock_min_val_crops": -1})


def test_ft18_activate_small_val_stock_worse_warns_without_blocking(
    client, trainers, registry, settings
):
    small = _model(
        client,
        trainers,
        registry,
        settings,
        {**STOCK_WORSE, "stock_min_val_crops": 100.0},
        {"stock_model": "speciesnet-4"},
    )
    shown = client.get(f"/v1/models/{small}").json()["stock_comparison"]
    assert shown["worse"] is True and shown["crops"] == 40
    assert shown.get("min_val_crops") == 100 and shown.get("enforced") is False
    r = client.post(
        f"/v1/models/{small}/activate", json={"consumer": "local", "project": "a"}
    )
    assert r.status_code == 200, r.text
    assert _deployments(settings, small) == [("local",)]


@pytest.mark.parametrize(
    "crops, min_val", [(40.0, 0.0), (150.0, 100.0)], ids=["threshold-0", "large-val"]
)
def test_ft18_activate_stock_worse_blocked_at_or_above_minimum(
    client, trainers, registry, settings, crops, min_val
):
    mid = _model(
        client,
        trainers,
        registry,
        settings,
        {**STOCK_WORSE, "stock_compared_crops": crops, "stock_min_val_crops": min_val},
        {"stock_model": "speciesnet-4"},
    )
    r = client.post(
        f"/v1/models/{mid}/activate", json={"consumer": "local", "project": "p"}
    )
    assert r.status_code == 409, r.text
    assert r.json()["detail"]["code"] == "stock_model_better"
    assert r.json()["detail"]["stock_comparison"].get("enforced") is True
    assert _deployments(settings, mid) == []


def test_ft18_both_pipelines_score_stock_on_identical_source_crops(
    tmp_path, monkeypatch
):
    from sparrow_finetune.pipelines.dinov3_linear import _split_key, assign_split
    from sparrow_finetune.pipelines.stock_model import StockModel

    seen: list[list[np.ndarray]] = []
    real = StockModel.predict_proba

    def spy(self, images):
        seen.append([np.asarray(im.convert("RGB")).copy() for im in images])
        return real(self, images)

    monkeypatch.setattr(StockModel, "predict_proba", spy)
    weights = tmp_path / "weights"
    write_stock_package(weights)
    syn = textured_items(duplicate=False)
    params = {"stock_model": STOCK_NAME, "seed": 0, "val_fraction": 0.2}
    conv = _conv_run(tmp_path / "conv", syn, params, weights_dir=weights)
    dino = _dino_run(tmp_path / "dino", syn, params, weights_dir=weights)
    assert len(seen) == 2

    def source_crops(ctx):
        return [
            np.asarray(crop_image(it.image_path, a.get("bbox")))
            for it in ctx.items
            if assign_split(_split_key(it), seed=0, val_fraction=0.2) == "val"
            for a in it.annotations
            if a["label"] in ("cat", "dog")
        ]

    for name, run, got in (("conventional", conv, seen[0]), ("dinov3", dino, seen[1])):
        want = source_crops(run["ctx"])
        assert len(got) == len(want) > 0, name
        assert all(
            g.shape == w.shape and np.array_equal(g, w) for g, w in zip(got, want)
        ), f"{name} scored the stock model on crops that differ from the source crops"


# --------------------------------------------------------------------------- HF cache offline


def _hf_cache_with_timm_weights(hub: Path, name: str) -> dict:
    import timm

    repo = timm.models.get_pretrained_cfg(name).hf_hub_id
    root = hub / f"models--{repo.replace('/', '--')}"
    rev = "0" * 40
    sd = _save_timm_weights(hub / "_tmp", name, "safetensors")
    snap = root / "snapshots" / rev
    snap.mkdir(parents=True)
    (hub / "_tmp" / "timm" / name / "model.safetensors").rename(
        snap / "model.safetensors"
    )
    (root / "refs").mkdir()
    (root / "refs" / "main").write_text(rev)
    return sd


@pytest.fixture
def offline_hf_cache(tmp_path, monkeypatch, no_network):
    from huggingface_hub import constants

    hub = tmp_path / "hf" / "hub"
    hub.mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(hub))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    return hub


@pytest.fixture
def no_network(monkeypatch):
    import socket

    def refuse(*_a, **_k):
        raise OSError("network access is disabled in this test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def test_ft18_offline_pretrained_uses_hugging_face_cache(tmp_path, offline_hf_cache):
    import torch

    from sparrow_finetune.pipelines.conventional_classifier import (
        ConventionalClassifierTrainer,
    )

    sd = _hf_cache_with_timm_weights(offline_hf_cache, "resnet18")
    out = _conv_run(
        tmp_path / "run",
        balanced_items(),
        {"pretrained": True, "epochs": 1, "warmup_epochs": 1},
        weights_dir=None,
    )
    assert out["metrics"]["val_crops"] > 0
    trainer = ConventionalClassifierTrainer()
    params = trainer.validate_params({**CONV_BASE, "pretrained": True})
    model = trainer._create(params, 3, pretrained=True)
    assert torch.equal(model.state_dict()["conv1.weight"], sd["conv1.weight"])


def test_ft18_offline_with_empty_hf_cache_fails_prepare_before_crops(
    tmp_path, offline_hf_cache
):
    from sparrow_finetune.pipelines.conventional_classifier import (
        ConventionalClassifierTrainer,
    )

    (tmp_path / "w").mkdir()
    trainer = ConventionalClassifierTrainer(weights_dir=tmp_path / "w")
    items = snapshot_items(balanced_items(), tmp_path / "images")
    ctx, _ = make_context(
        trainer, items, tmp_path / "work", {**CONV_BASE, "pretrained": True}
    )
    with pytest.raises(ValueError, match="HF_HUB_OFFLINE"):
        trainer.prepare(ctx)
    assert not (tmp_path / "work" / "crops").exists() or not any(
        (tmp_path / "work" / "crops").iterdir()
    )


def test_ft22_offline_missing_weights_reads_hf_cache_once(
    tmp_path, offline_hf_cache, monkeypatch
):
    from sparrow_finetune.pipelines.conventional_classifier import (
        ConventionalClassifierTrainer,
    )

    calls = []
    real = ConventionalClassifierTrainer._hf_cache_weights

    def counting(model):
        calls.append(model)
        return real(model)

    monkeypatch.setattr(
        ConventionalClassifierTrainer, "_hf_cache_weights", staticmethod(counting)
    )
    (tmp_path / "w").mkdir()
    trainer = ConventionalClassifierTrainer(weights_dir=tmp_path / "w")
    items = snapshot_items(balanced_items(), tmp_path / "images")
    ctx, _ = make_context(
        trainer, items, tmp_path / "work", {**CONV_BASE, "pretrained": True}
    )
    with pytest.raises(ValueError, match="the Hugging Face cache"):
        trainer.prepare(ctx)
    assert calls == [CONV_BASE["model"]]


# --------------------------------------------------------------------------- client address


def _same_key(settings, registry, peer, failing_xff, probe_xff) -> bool:
    """10 failed logins as ``failing_xff``; is ``probe_xff`` then locked out (same limiter key)?"""
    s = dataclasses.replace(settings, trusted_proxies="10.0.0.0/8")
    with _proxied_client(s, registry, peer) as c:
        for _ in range(10):
            assert _login(c, "wrong", failing_xff) == 401
        return _login(c, PASSWORD, probe_xff) == 429


@pytest.mark.parametrize(
    "peer, failing, same, other",
    [
        ("10.1.2.3", "203.0.113.7:51234", "203.0.113.7", "198.51.100.9"),
        ("10.1.2.3", "[2001:db8::1]:443", "2001:db8::1", "2001:db8:0:1::1"),  # FT-22: /64 key
        ("10.1.2.3", "::ffff:203.0.113.7", "203.0.113.7", "198.51.100.9"),
        ("10.1.2.3", "203.0.113.7, 10.9.9.9:8080", "203.0.113.7", "198.51.100.9"),
        ("::ffff:10.1.2.3", "203.0.113.7", "203.0.113.7", "198.51.100.9"),
    ],
    ids=[
        "ipv4-port",
        "bracketed-ipv6-port",
        "mapped-hop",
        "trusted-hop-port",
        "mapped-peer",
    ],
)
def test_ft18_forwarded_client_address_forms(
    settings, registry, peer, failing, same, other
):
    add_user(settings)
    assert _same_key(settings, registry, peer, failing, same)
    assert not _same_key(settings, registry, peer, failing, other)


def test_ft18_malformed_hop_and_untrusted_peer_still_ignore_spoofed_entries(
    settings, registry
):
    add_user(settings)
    # A malformed hop stops the walk: the address left of it is never the limiter key.
    assert not _same_key(
        settings, registry, "10.1.2.3", "203.0.113.7, bogus", "203.0.113.7"
    )
    # An untrusted peer's forwarded header (with ports) is ignored: every attempt is the peer.
    assert _same_key(
        settings, registry, "172.16.0.5", "203.0.113.7:1", "198.51.100.9:2"
    )


def test_ft18_cli_api_does_not_let_uvicorn_trust_forwarded_headers(
    settings, monkeypatch
):
    import uvicorn

    from sparrow_finetune import cli

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append(kw))
    monkeypatch.setenv("SPARROW_FT_DATABASE_URL", settings.database_url)
    monkeypatch.setenv("SPARROW_FT_DATA_DIR", str(settings.data_dir))
    assert cli.main(["api"]) == 0
    (kwargs,) = calls
    # uvicorn defaults proxy_headers to True and would rewrite the peer from X-Forwarded-For
    # before the app's trusted-proxy check runs.
    assert kwargs.get("proxy_headers") is False
