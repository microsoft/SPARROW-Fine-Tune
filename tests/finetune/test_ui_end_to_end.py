"""Phase 2B acceptance test U6: the browser UI end to end.

A real uvicorn server (Postgres, the fast fake trainer, a worker thread) is driven
with headless Chromium: sign in -> workspace and dataset -> upload 4 images ->
import annotations -> confirm one and exclude one in the review grid -> snapshot ->
train -> the Jobs screen shows the job succeeded -> activate the model locally ->
the Active badge is visible. Any ``pageerror`` or console error fails the test.
Screenshots of each screen go to ``tmp_path`` (or ``$SPARROW_FT_SCREENSHOT_DIR``).

Spec: docs/review/implement-phase2-service/spec.md, section 2B (U1-U6).
Needs the ``e2e`` extra (``uv run --extra dev --extra e2e pytest ...``) and a
Playwright Chromium build; the test is skipped when Playwright is not installed.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import threading
import time
from pathlib import Path

import pytest

from .helpers import png_bytes
from .service_helpers import PASSWORD, add_user, db

sync_api = pytest.importorskip("playwright.sync_api")

LABELS = {"cat1.png": "cat", "dog1.png": "dog", "cat2.png": "cat", "dog2.png": "dog"}


@pytest.fixture
def server(settings, registry):
    import uvicorn

    from sparrow_finetune.api import create_app
    from sparrow_finetune.workers import run_once

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(create_app(settings, registry=registry), log_level="warning")
    )
    web = threading.Thread(
        target=lambda: asyncio.run(server.serve(sockets=[sock])), daemon=True
    )
    web.start()

    stop = threading.Event()
    worker_errors: list = []

    def work() -> None:
        while not stop.is_set():
            try:
                if not run_once(settings, "w-ui", {"cpu"}, registry=registry):
                    stop.wait(0.2)
            except Exception as exc:  # surfaced by the test
                worker_errors.append(exc)
                return

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "uvicorn did not start"
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}", worker_errors
    finally:
        stop.set()
        server.should_exit = True
        worker.join(10)
        web.join(10)


def test_u6_ui_end_to_end(server, settings, tmp_path):
    base, worker_errors = server
    add_user(settings, "alice", PASSWORD)
    shots = Path(os.environ.get("SPARROW_FT_SCREENSHOT_DIR") or tmp_path / "shots")
    shots.mkdir(parents=True, exist_ok=True)

    files = []
    for i, name in enumerate(LABELS):
        path = tmp_path / name
        path.write_bytes(png_bytes(seed=i + 1, size=(48, 40)))
        files.append(str(path))
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "items": [
                    {"image": n, "annotations": [{"label": lb}]}
                    for n, lb in LABELS.items()
                ]
            }
        )
    )

    problems: list[str] = []
    with sync_api.sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        page.set_default_timeout(20_000)
        page.on("pageerror", lambda exc: problems.append(f"pageerror: {exc}"))
        page.on(
            "console",
            lambda msg: (
                problems.append(f"console {msg.type}: {msg.text}")
                if msg.type == "error"
                else None
            ),
        )
        expect = sync_api.expect
        try:
            # U1: sign in
            page.goto(base + "/")
            expect(page.locator("#view-login")).to_be_visible()
            page.screenshot(path=str(shots / "01-login.png"))
            page.fill("#login-username", "alice")
            page.fill("#login-password", PASSWORD)
            page.click("#login-submit")
            expect(page.locator("#view-data")).to_be_visible()
            expect(page.locator("#user-name")).to_have_text("alice")
            expect(page.locator("#logout-btn")).to_be_visible()

            # U2: workspace, dataset, upload, annotations
            page.click("#show-new-workspace")
            page.fill("#new-workspace", "Field cameras")
            page.click("#create-workspace")
            expect(page.locator("#topbar-workspace")).to_have_text("Field cameras")
            page.click("#show-new-dataset")
            page.fill("#new-dataset", "Night set")
            page.select_option("#new-dataset-task", "classification")
            page.check("#new-dataset-whole-image")  # image-level labels only, no boxes
            page.click("#create-dataset")
            expect(page.locator("#topbar-dataset")).to_contain_text("Night set")
            # FT-44: assist on a whole-image dataset writes image labels, so no control says boxes.
            expect(page.locator("#assist-enable-text")).to_have_text("Label images with a detector")
            expect(page.locator("#assist-btn")).to_have_text("Label images…")
            expect(page.locator("#assist-overwrite-text")).to_have_text("Replace existing labels")
            expect(page.locator("#assist-help")).to_contain_text("Labels arrive as pending")
            # Choosing files starts the upload (M2: the button opens this picker).
            page.set_input_files("#upload-files", files)
            expect(page.locator("#toast")).to_contain_text("Uploaded 4 image(s)")
            page.select_option("#annotation-format", "sparrow")
            page.set_input_files("#annotations-file", str(annotations))
            cards = page.locator("#example-grid .example-card")
            expect(cards).to_have_count(4)
            expect(page.locator("#count-pending")).to_have_text("4")
            # thumbnails really load (the content route works through the session cookie)
            page.wait_for_function(
                "() => [...document.querySelectorAll('#example-grid img')]"
                ".every((i) => i.complete && i.naturalWidth > 0)"
            )

            card = page.locator('.example-card[data-image="cat1.png"]')
            card.locator('[data-example-action="confirm"]').click()
            expect(card).to_have_attribute("data-status", "confirmed")
            card = page.locator('.example-card[data-image="dog2.png"]')
            card.locator('[data-example-action="exclude"]').click()
            expect(card).to_have_attribute("data-status", "excluded")
            expect(page.locator("#count-confirmed")).to_have_text("1")
            expect(page.locator("#count-excluded")).to_have_text("1")
            expect(page.locator("#count-pending")).to_have_text("2")
            page.screenshot(path=str(shots / "02-prepare-data.png"), full_page=True)

            page.click("#create-snapshot")
            expect(page.locator("#snapshot-info")).to_contain_text("3 item(s)")
            page.select_option("#pipeline-select", "fake_cls")
            expect(page.locator("#start-training")).to_be_enabled()

            # U3: train, follow the job. Hold the request so the loading panel is observable.
            held = []
            page.route(
                "**/v1/jobs",
                lambda route: (
                    held.append(route)
                    if route.request.method == "POST"
                    else route.continue_()
                ),
            )
            page.click("#start-training")
            busy = page.locator("#busy-overlay")
            expect(busy).to_be_visible()
            expect(busy).to_contain_text("Starting training")
            expect(page.locator("#start-training")).to_be_disabled()
            expect(page.locator("#view-data")).to_be_visible()
            assert len(held) == 1
            held[0].continue_()
            page.unroute("**/v1/jobs")
            expect(page.locator("#view-jobs")).to_be_visible()
            expect(busy).to_be_hidden()
            expect(page.locator("#job-status")).to_have_attribute(
                "data-state", "succeeded", timeout=60_000
            )
            expect(page.locator("#event-list li").first).to_be_visible()
            expect(page.locator("#retry-job")).to_be_hidden()
            # the measurements chart really drew the reported point (series colour #0e4a59)
            drawn = page.evaluate(
                """() => { const c = document.querySelector('#live-curve');
                  const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
                  let n = 0; for (let i = 0; i < d.length; i += 4)
                    if (d[i] === 14 && d[i+1] === 74 && d[i+2] === 89 && d[i+3] > 200) n++;
                  return n; }"""
            )
            assert drawn > 10, "the training chart drew no measurement"
            page.screenshot(path=str(shots / "03-jobs.png"), full_page=True)

            # U4: models, activate locally
            page.click('.sidebar-nav [data-page="models"]')
            expect(page.locator("#model-list [data-model-id]")).to_have_count(1)
            expect(page.locator("#model-detail")).to_be_visible()
            expect(page.locator("#activate-studio")).to_be_hidden()
            page.click("#activate-local")
            badge = page.locator("#model-active-badge")
            expect(badge).to_be_visible()
            expect(badge).to_have_text("Active: local/default")
            page.screenshot(path=str(shots / "04-models.png"), full_page=True)

            # U5: no error banner was ever left showing
            expect(page.locator("#error-banner")).to_be_hidden()
        except Exception:
            page.screenshot(path=str(shots / "failure.png"), full_page=True)
            raise
        finally:
            browser.close()

    assert not worker_errors, worker_errors
    assert problems == [], problems

    with db(settings) as conn:
        snap = conn.execute("SELECT item_count FROM snapshots").fetchone()
        dep = conn.execute(
            "SELECT consumer, project, activated_by FROM deployments WHERE deactivated_at IS NULL"
        ).fetchone()
        actions = [
            r["action"]
            for r in conn.execute("SELECT action FROM audit_events ORDER BY id")
        ]
    assert snap["item_count"] == 3, (
        "the excluded example must be left out of the snapshot"
    )
    assert dep == {"consumer": "local", "project": "default", "activated_by": "alice"}
    assert (
        actions[0] == "login"
        and "job.submit" in actions
        and actions[-1] == "model.activate"
    )
