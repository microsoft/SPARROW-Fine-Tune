"""Browser pin: the Jobs page follows a job's state while it is open (queued -> running ->
succeeded) without a reload. Regression for a job that trained with a live curve while
its badge still said "queued"."""

from __future__ import annotations

import pytest

from .service_helpers import PASSWORD, add_user
from . import test_ui_end_to_end as _e2e
from .test_backlog_service_ui import _browser, _login, _sql

server = _e2e.server

sync_api = pytest.importorskip("playwright.sync_api")


def test_job_badge_follows_state_changes(server, settings):
    base, _ = server
    add_user(settings, "alice", PASSWORD)
    (ws_id,) = _sql(
        settings, "INSERT INTO workspaces (name) VALUES ('W') RETURNING id"
    )[0]
    (ds_id,) = _sql(
        settings,
        "INSERT INTO datasets (workspace_id, name, task) VALUES (%s, 'D', 'classification')"
        " RETURNING id",
        (ws_id,),
    )[0]
    (snap_id,) = _sql(
        settings,
        "INSERT INTO snapshots (dataset_id, content_hash, item_count, label_counts)"
        " VALUES (%s, 'x', 0, '{}') RETURNING id",
        (ds_id,),
    )[0]
    # No worker serves this pipeline, so the job stays queued until the test moves it.
    (job_id,) = _sql(
        settings,
        "INSERT INTO jobs (pipeline, snapshot_id, state) VALUES ('zz_unserved', %s, 'queued')"
        " RETURNING id",
        (snap_id,),
    )[0]
    problems: list[str] = []
    expect = sync_api.expect
    with sync_api.sync_playwright() as pw:
        browser, page = _browser(pw, problems)
        try:
            _login(page, base)
            page.click("nav [data-page='jobs']")
            badge = page.locator("#job-status")
            row = page.locator(f"#job-history tr[data-job-id='{job_id}'] .status-badge")
            expect(badge).to_have_text("queued")
            # The test server's worker polls, but does not serve this pipeline.
            expect(page.locator("#job-time")).to_have_text("No worker online")

            _sql(
                settings,
                "UPDATE jobs SET state = 'running', started_at = now() WHERE id = %s",
                (job_id,),
            )
            _sql(
                settings,
                "INSERT INTO job_events (job_id, seq, stage, message, metrics)"
                " VALUES (%s, 1, 'train', 'epoch 1', '{\"epoch\": 1, \"loss\": 0.5}')",
                (job_id,),
            )
            expect(badge).to_have_text("running", timeout=10_000)
            expect(row).to_have_text("running")
            expect(page.locator("#job-time")).to_contain_text("started")
            expect(page.locator("#nav-job-badge")).to_have_text("1")

            _sql(
                settings,
                "UPDATE jobs SET state = 'succeeded', finished_at = now() WHERE id = %s",
                (job_id,),
            )
            expect(badge).to_have_text("succeeded", timeout=10_000)
            expect(row).to_have_text("succeeded")
            expect(page.locator("#nav-job-badge")).to_be_hidden()
        finally:
            browser.close()
    assert problems == [], problems
