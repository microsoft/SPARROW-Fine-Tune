from dataclasses import replace

import pytest

from sparrow_finetune.workers import run_once

from .dinov3_helpers import write_mean_rgb_onnx
from .service_helpers import PASSWORD, add_user
from .test_backlog_service_ui import _browser, _login
from .test_dinov3_end_to_end import builtin_registry
from .test_local_base_models import standalone_server as progress_server  # noqa: F401
from .test_release_first_run import syn, upload_dataset


@pytest.fixture
def settings(settings, tmp_path):
    encoder = write_mean_rgb_onnx(tmp_path / "encoder.onnx")
    return replace(settings, embedder=f"onnx:{encoder}")


@pytest.fixture
def registry(settings):
    return builtin_registry(settings)


def test_browser_progress_then_cached_snapshot(request, settings, registry, client, tmp_path):
    pw = pytest.importorskip("playwright.sync_api")
    server = request.getfixturevalue("progress_server")
    items = syn("progress-")
    upload_dataset(client, items)
    add_user(settings, "alice", PASSWORD)
    problems = []
    with pw.sync_playwright() as playwright:
        browser, page = _browser(playwright, problems)
        try:
            page.on("console", lambda m: problems.append(m.text) if m.type == "error" else None)
            page.on("requestfailed", lambda r: problems.append(f"{r.url}: {r.failure}"))
            _login(page, server)
            pw.expect(page.locator("#feature-prep-items .feature-missing")).to_have_count(len(items))
            choice = page.locator("#prepare-features")
            button = page.locator("#create-snapshot")
            status = page.locator("#feature-prep-status")
            choice.check()
            button.click()
            pw.expect(status).to_contain_text("queued")
            pw.expect(page.locator("#start-training")).to_be_disabled()
            page.screenshot(path=str(tmp_path / "queued.png"))
            assert run_once(settings, "progress-ui", {"cpu"}, registry=registry)
            pw.expect(status).to_contain_text("DINO features ready")
            pw.expect(page.locator("#start-training")).to_be_enabled()
            pw.expect(page.locator("#feature-prep-items .feature-valid")).to_have_count(len(items))
            choice.check()
            button.click()
            pw.expect(status).to_contain_text("No new data")
            pw.expect(status).to_contain_text("nothing was computed")
            page.screenshot(path=str(tmp_path / "reused.png"))
            assert not run_once(settings, "progress-idle", {"cpu"}, registry=registry)
        finally:
            browser.close()
    assert problems == []


def test_cache_list_exposes_crops_after_first_page(request, settings, client):
    pw = pytest.importorskip("playwright.sync_api")
    server = request.getfixturevalue("progress_server")
    upload_dataset(client, syn("large-"))
    add_user(settings, "alice", PASSWORD)
    with pw.sync_playwright() as playwright:
        browser, page = _browser(playwright, [])
        try:
            items = [{"image": f"image-{i}.jpg", "crop": [0, 0, 10, 10], "status": "missing"}
                     for i in range(201)]
            page.route("**/feature-prep", lambda r: r.fulfill(json={
                "items": items, "counts": {"total": 201, "missing": 201}, "to_compute": 201,
            }))
            _login(page, server)
            page.get_by_role("button", name="Show more crops (1 remaining)").click()
            pw.expect(page.locator("#feature-prep-items li").last).to_contain_text("image-200.jpg")
            pw.expect(page.locator("#feature-prep-items li").last).to_contain_text("To compute")
        finally:
            browser.close()
