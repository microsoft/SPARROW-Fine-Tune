"""Browser pins for FT-16 (stock-model gate in the model page) and FT-17 (upload replace).

Drives the real page in Chromium against the uvicorn ``server`` fixture.
"""

from __future__ import annotations

import json
import time

import psycopg
import pytest
from psycopg.types.json import Jsonb

from .helpers import build_snapshot, png_bytes, queued_job
from .service_helpers import PASSWORD, add_user
from . import test_ui_end_to_end as _e2e

server = _e2e.server  # the live-server fixture, shared with the UI end-to-end tests

sync_api = pytest.importorskip("playwright.sync_api")

WORSE = {
    "stock_val_macro_f1": 0.81,
    "stock_finetuned_val_macro_f1": 0.62,
    "stock_compared_crops": 40.0,
    "stock_worse": 1.0,
}
BETTER = {
    "stock_val_macro_f1": 0.55,
    "stock_finetuned_val_macro_f1": 0.9,
    "stock_compared_crops": 40.0,
    "stock_worse": 0.0,
}


def _sql(settings, query, params=()):
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        cur = conn.execute(query, params)
        return cur.fetchall() if cur.description else None


def _trained_model(client, trainers, settings, metrics) -> int:
    """Queue a job (the server's worker runs it), then stamp the stock comparison."""
    snap = build_snapshot(client)["snapshot"]
    job_id = queued_job(client, trainers.basic.id, snap["id"])
    deadline = time.monotonic() + 60
    while True:
        rows = _sql(
            settings, "SELECT id FROM model_versions WHERE job_id = %s", (job_id,)
        )
        if rows:
            break
        assert time.monotonic() < deadline, "worker produced no model"
        time.sleep(0.2)
    model_id = rows[0][0]
    _set_metrics(settings, model_id, metrics)
    _sql(
        settings,
        "UPDATE model_versions SET metadata = metadata || %s WHERE id = %s",
        (Jsonb({"stock_model": "speciesnet-4"}), model_id),
    )
    return model_id


def _set_metrics(settings, model_id, metrics):
    _sql(
        settings,
        "UPDATE model_versions SET metrics = metrics || %s WHERE id = %s",
        (Jsonb(metrics), model_id),
    )


def _deployments(settings, model_id) -> int:
    return len(
        _sql(settings, "SELECT id FROM deployments WHERE model_id = %s", (model_id,))
    )


def _browser(pw, problems):
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    page.set_default_timeout(20_000)
    page.on("pageerror", lambda exc: problems.append(f"pageerror: {exc}"))
    return browser, page


def _login(page, base):
    page.goto(base + "/")
    page.fill("#login-username", "alice")
    page.fill("#login-password", PASSWORD)
    page.click("#login-submit")
    sync_api.expect(page.locator("#view-data")).to_be_visible()


@pytest.fixture
def narrow_browser(tmp_path):
    events = []
    problems = []
    with sync_api.sync_playwright() as pw:
        browser, page = _browser(pw, problems)
        page.set_viewport_size({"width": 390, "height": 844})
        page.on("console", lambda m: events.append({"kind": f"console.{m.type}", "text": m.text}))
        page.on("pageerror", lambda e: events.append({"kind": "pageerror", "text": str(e)}))
        page.on("requestfailed", lambda r: events.append({
            "kind": "requestfailed", "url": r.url, "text": r.failure,
        }))
        try:
            yield page
        finally:
            page.screenshot(path=str(tmp_path / "narrow.png"), full_page=True)
            browser.close()
            (tmp_path / "browser-events.json").write_text(json.dumps(events, indent=2))
    assert not problems, problems
    unexpected = [
        e for e in events
        if e["kind"] in ("console.error", "pageerror")
        or (e["kind"] == "requestfailed" and e["text"] != "net::ERR_ABORTED")
    ]
    assert not unexpected, unexpected


def test_narrow_navigation_remains_reachable(server, settings, narrow_browser):
    base, _ = server
    add_user(settings, "alice", PASSWORD)
    page = narrow_browser
    _login(page, base)
    toggle = page.get_by_role("button", name="Toggle navigation")
    sync_api.expect(toggle).to_be_visible(timeout=2000)
    toggle.click()
    jobs = page.locator('.sidebar-nav [data-page="jobs"]')
    assert jobs.bounding_box()["x"] >= 0
    jobs.click()
    sync_api.expect(page.locator("#view-jobs")).to_be_visible()
    sync_api.expect(toggle).to_have_attribute("aria-expanded", "false")
    toggle.click()
    page.locator('.sidebar-nav [data-page="models"]').click()
    sync_api.expect(page.locator("#view-models")).to_be_visible()


def test_benign_resize_observer_notice_shows_no_error_banner(server, settings):
    """Browsers report "ResizeObserver loop completed with undelivered notifications"
    as a window error event without an Error object; extensions resizing the login
    form trigger it. It is not an application failure and must not reach the banner,
    while genuine uncaught errors still must."""
    base, _ = server
    add_user(settings, "alice", PASSWORD)
    problems = []
    with sync_api.sync_playwright() as pw:
        browser, page = _browser(pw, problems)
        try:
            _login(page, base)
            page.evaluate(
                """() => window.dispatchEvent(new ErrorEvent("error", {
                    message: "ResizeObserver loop completed with undelivered notifications.",
                }))"""
            )
            sync_api.expect(page.locator("#error-banner")).to_be_hidden()
            page.evaluate("() => setTimeout(() => { throw new Error('real failure'); })")
            sync_api.expect(page.locator("#error-text")).to_contain_text("real failure")
        finally:
            browser.close()
    assert [p for p in problems if "real failure" not in p] == []


@pytest.mark.parametrize("width", [390, 820, 821, 1000, 1081, 1400])
def test_narrow_models_actions_fit_viewport(
    server, client, trainers, settings, narrow_browser, width,
):
    base, _ = server
    _trained_model(client, trainers, settings, WORSE)
    add_user(settings, "alice", PASSWORD)
    page = narrow_browser
    page.set_viewport_size({"width": 1400, "height": 1000})
    _login(page, base)
    page.locator('.sidebar-nav [data-page="models"]').click()
    sync_api.expect(page.locator("#model-detail")).to_be_visible()
    page.set_viewport_size({"width": width, "height": 844})
    layout = page.evaluate("""() => ({
        viewport: innerWidth,
        width: document.documentElement.scrollWidth,
        actions: ['activate-local', 'model-download'].map(id => {
            const r = document.getElementById(id).getBoundingClientRect();
            return {id, left: r.left, right: r.right, width: r.width};
        }),
        content: Array.from(document.querySelectorAll(
            '#model-detail .metrics-grid > div, #model-record dd, #compare-select, #model-detail .data-table'
        )).map(node => {
            const r = node.getBoundingClientRect();
            return {element: node.id || node.tagName, left: r.left, right: r.right, width: r.width};
        })
    })""")
    assert layout["width"] <= layout["viewport"], layout
    assert layout["content"], layout
    for action in layout["actions"] + layout["content"]:
        assert action["left"] >= 0 and action["right"] <= layout["viewport"], layout
        assert action["width"] > 0, layout


def test_ft16_ui_shows_stock_comparison_and_confirms_activation(
    server, client, trainers, settings, tmp_path
):
    base, worker_errors = server
    model_id = _trained_model(client, trainers, settings, WORSE)
    add_user(settings, "alice", PASSWORD)
    problems: list[str] = []
    dialogs: list[str] = []
    expect = sync_api.expect
    with sync_api.sync_playwright() as pw:
        browser, page = _browser(pw, problems)
        try:
            _login(page, base)
            page.click('.sidebar-nav [data-page="models"]')
            expect(page.locator("#model-detail")).to_be_visible()
            box = page.locator("#stock-comparison")
            expect(box).to_be_visible()
            expect(box).to_contain_text("speciesnet-4")
            expect(box).to_contain_text("0.81")
            expect(box).to_contain_text("0.62")
            expect(box).to_contain_text("stock model scored higher")
            assert "stock-worse" in (box.get_attribute("class") or "")

            # declined: nothing activated
            page.once("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))
            page.click("#activate-local")
            page.wait_for_timeout(500)
            assert dialogs and "speciesnet-4" in dialogs[0], dialogs
            assert _deployments(settings, model_id) == 0
            expect(page.locator("#model-active-badge")).to_be_hidden()

            # accepted: activated with the override
            page.once("dialog", lambda d: (dialogs.append(d.message), d.accept()))
            page.click("#activate-local")
            expect(page.locator("#model-active-badge")).to_be_visible()
            assert _deployments(settings, model_id) == 1
            assert len(dialogs) == 2
            page.screenshot(path=str(tmp_path / "ft16-models.png"), full_page=True)
        except Exception:
            page.screenshot(path=str(tmp_path / "failure.png"), full_page=True)
            raise
        finally:
            browser.close()
    assert not worker_errors, worker_errors
    assert problems == [], problems


def test_ft16_ui_handles_stale_409_with_confirm(
    server, client, trainers, settings, tmp_path
):
    base, worker_errors = server
    model_id = _trained_model(client, trainers, settings, BETTER)
    add_user(settings, "alice", PASSWORD)
    problems: list[str] = []
    dialogs: list[str] = []
    expect = sync_api.expect
    with sync_api.sync_playwright() as pw:
        browser, page = _browser(pw, problems)
        try:
            _login(page, base)
            page.click('.sidebar-nav [data-page="models"]')
            box = page.locator("#stock-comparison")
            expect(box).to_be_visible()
            assert "stock-worse" not in (box.get_attribute("class") or "")
            # the page is now stale: the evaluation says the stock model is better
            _set_metrics(settings, model_id, WORSE)
            page.once("dialog", lambda d: (dialogs.append(d.message), d.accept()))
            page.click("#activate-local")
            expect(page.locator("#model-active-badge")).to_be_visible()
            assert len(dialogs) == 1 and "speciesnet-4" in dialogs[0], dialogs
            assert _deployments(settings, model_id) == 1
            expect(page.locator("#error-banner")).to_be_hidden()
        except Exception:
            page.screenshot(path=str(tmp_path / "failure.png"), full_page=True)
            raise
        finally:
            browser.close()
    assert not worker_errors, worker_errors
    assert problems == [], problems


def test_ft19_ui_shows_reference_warning(server, client, trainers, settings, tmp_path):
    base, worker_errors = server
    _trained_model(
        client,
        trainers,
        settings,
        {
            "reference_val_macro_f1": 0.7636,
            "val_macro_f1": 0.74,
            "val_crops": 48.0,
            "reference_worse": 1.0,
        },
    )
    add_user(settings, "alice", PASSWORD)
    problems: list[str] = []
    expect = sync_api.expect
    with sync_api.sync_playwright() as pw:
        browser, page = _browser(pw, problems)
        try:
            _login(page, base)
            page.click('.sidebar-nav [data-page="models"]')
            expect(page.locator("#model-detail")).to_be_visible()
            box = page.locator("#reference-comparison")
            expect(box).to_be_visible()
            expect(box).to_contain_text("0.764")
            expect(box).to_contain_text("0.740")
            expect(box).to_contain_text("48")
            expect(box).to_contain_text("logistic-regression reference scored higher")
            page.screenshot(path=str(tmp_path / "ft19-models.png"), full_page=True)
        except Exception:
            page.screenshot(path=str(tmp_path / "failure.png"), full_page=True)
            raise
        finally:
            browser.close()
    assert not worker_errors, worker_errors
    assert problems == [], problems


def test_ft17_ui_upload_offers_replace_on_conflict(server, settings, tmp_path):
    base, worker_errors = server
    add_user(settings, "alice", PASSWORD)
    first = tmp_path / "one" / "a.png"
    second = tmp_path / "two" / "a.png"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(png_bytes(seed=1, size=(48, 40)))
    second.write_bytes(png_bytes(seed=2, size=(64, 32)))
    problems: list[str] = []
    dialogs: list[str] = []
    expect = sync_api.expect
    with sync_api.sync_playwright() as pw:
        browser, page = _browser(pw, problems)
        try:
            _login(page, base)
            page.click("#show-new-workspace")
            page.fill("#new-workspace", "Field cameras")
            page.click("#create-workspace")
            expect(page.locator("#topbar-workspace")).to_have_text("Field cameras")
            page.click("#show-new-dataset")
            page.fill("#new-dataset", "Night set")
            page.select_option("#new-dataset-task", "classification")
            page.click("#create-dataset")
            expect(page.locator("#topbar-dataset")).to_contain_text("Night set")
            page.set_input_files("#upload-files", str(first))
            expect(page.locator("#toast")).to_contain_text("Uploaded 1 image(s)")
            # The hidden toast keeps its text: wait for it to clear, then for a new one.
            expect(page.locator("#toast")).to_be_hidden()

            page.once("dialog", lambda d: (dialogs.append(d.message), d.accept()))
            page.set_input_files("#upload-files", str(second))
            expect(page.locator("#toast")).to_be_visible()
            expect(page.locator("#toast")).to_contain_text("Uploaded 1 image(s)")
            assert len(dialogs) == 1 and "a.png" in dialogs[0], dialogs
            expect(page.locator("#error-banner")).to_be_hidden()
        except Exception:
            page.screenshot(path=str(tmp_path / "failure.png"), full_page=True)
            raise
        finally:
            browser.close()
    rows = _sql(settings, "SELECT width, height FROM dataset_images")
    assert rows == [(64, 32)], rows
    assert not worker_errors, worker_errors
    assert problems == [], problems


def _assert_actions_beside_name(page):
    """The Local project / Activate / Download column stays right of the name."""
    name = page.locator(".model-summary > div").first.bounding_box()
    actions = page.locator(".model-summary .model-actions").bounding_box()
    assert actions["x"] >= name["x"] + name["width"], (name, actions)
    assert actions["y"] < name["y"] + name["height"], (name, actions)


def test_ui_renames_a_model(server, client, trainers, settings):
    base, worker_errors = server
    model_id = _trained_model(client, trainers, settings, {})
    add_user(settings, "alice", PASSWORD)
    problems: list[str] = []
    expect = sync_api.expect
    with sync_api.sync_playwright() as pw:
        browser, page = _browser(pw, problems)
        try:
            _login(page, base)
            page.click('.sidebar-nav [data-page="models"]')
            expect(page.locator("#model-name")).to_contain_text(f"Model #{model_id}")
            expect(page.locator("#rename-model-form")).to_be_hidden()
            page.click("#show-rename-model")
            page.fill("#rename-model", "  Amazon mammals v1 ")
            page.click("#save-rename-model")
            expect(page.locator("#model-name")).to_have_text("Amazon mammals v1")
            expect(page.locator("#rename-model-form")).to_be_hidden()
            expect(page.locator("#model-description")).to_contain_text(f"Model #{model_id}")
            # 2026-10-28: after a rename the actions wrapped below the name.
            _assert_actions_beside_name(page)
            expect(page.locator(f'#model-list [data-model-id="{model_id}"] strong')).to_have_text("Amazon mammals v1")
            expect(page.locator("#save-rename-model")).to_be_enabled()
            page.click("#show-rename-model")
            long_name = "Amazon mammals and birds, dry season 2026, all cameras v12"
            page.fill("#rename-model", long_name)
            page.click("#save-rename-model")
            expect(page.locator("#model-name")).to_have_text(long_name)
            _assert_actions_beside_name(page)
            # A blank name goes back to the default title.
            page.click("#show-rename-model")
            expect(page.locator("#rename-model")).to_have_value(long_name)
            page.fill("#rename-model", "")
            page.click("#save-rename-model")
            expect(page.locator("#model-name")).to_contain_text(f"Model #{model_id}")
            _assert_actions_beside_name(page)
        finally:
            browser.close()
    assert problems == []
    assert worker_errors == []
    assert _sql(settings, "SELECT name FROM model_versions WHERE id = %s", (model_id,)) == [(None,)]


def test_ui_jobs_and_models_follow_the_workspace(server, client, trainers, settings):
    """2026-10-28: Jobs and Models listed every workspace's work, so workspaces separated nothing."""
    base, worker_errors = server
    model_a = _trained_model(client, trainers, settings, {})
    model_b = _trained_model(client, trainers, settings, {})
    (ws_a, job_a), (ws_b, job_b) = [
        _sql(
            settings,
            "SELECT d.workspace_id, m.job_id FROM model_versions m"
            " JOIN snapshots s ON s.id = m.snapshot_id JOIN datasets d ON d.id = s.dataset_id"
            " WHERE m.id = %s",
            (mid,),
        )[0]
        for mid in (model_a, model_b)
    ]
    assert ws_a != ws_b
    names = dict(_sql(settings, "SELECT id, name FROM workspaces"))
    add_user(settings, "alice", PASSWORD)
    problems: list[str] = []
    expect = sync_api.expect
    with sync_api.sync_playwright() as pw:
        browser, page = _browser(pw, problems)
        try:
            _login(page, base)
            page.select_option("#workspace-select", str(ws_a))
            page.click('.sidebar-nav [data-page="jobs"]')
            expect(page.locator("#job-history tr")).to_have_count(1)
            expect(page.locator("#job-history")).to_contain_text(f"#{job_a}")
            expect(page.locator("#job-id")).to_have_text(f"JOB #{job_a}")
            expect(page.locator("#jobs-scope")).to_contain_text(names[ws_a])
            page.click('.sidebar-nav [data-page="models"]')
            expect(page.locator("#model-list [data-model-id]")).to_have_count(1)
            expect(page.locator(f'#model-list [data-model-id="{model_a}"]')).to_be_visible()
            expect(page.locator("#models-scope")).to_contain_text(names[ws_a])

            # Switched from the sidebar while the Models page is open: the list reloads in place.
            page.select_option("#workspace-select", str(ws_b))
            expect(page.locator("#models-scope")).to_contain_text(names[ws_b])
            expect(page.locator("#model-list [data-model-id]")).to_have_count(1)
            expect(page.locator("#model-name")).to_contain_text(f"Model #{model_b}")
            expect(page.locator("#models-scope")).to_contain_text(names[ws_b])
            page.click('.sidebar-nav [data-page="jobs"]')
            expect(page.locator("#job-history tr")).to_have_count(1)
            expect(page.locator("#job-id")).to_have_text(f"JOB #{job_b}")
        finally:
            browser.close()
    assert problems == []
    assert worker_errors == []
