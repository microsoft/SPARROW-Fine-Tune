"""Browser pins for deleting datasets and workspaces: the typed name must match, and the
confirmation lists the jobs and models that go with trained data."""

from __future__ import annotations

import pytest

from .service_helpers import PASSWORD, add_user
from . import test_ui_end_to_end as _e2e
from .test_backlog_service_ui import _browser, _login, _sql

server = _e2e.server

sync_api = pytest.importorskip("playwright.sync_api")


def _create(page, workspace, dataset):
    expect = sync_api.expect
    page.click("#show-new-workspace")
    page.fill("#new-workspace", workspace)
    page.click("#create-workspace")
    expect(page.locator("#topbar-workspace")).to_have_text(workspace)
    page.click("#show-new-dataset")
    page.fill("#new-dataset", dataset)
    page.click("#create-dataset")
    expect(page.locator("#dataset-select option:checked")).to_contain_text(dataset)


def _answer_prompts(page, answers):
    def reply(dialog):
        assert dialog.type == "prompt", dialog.type
        dialog.accept(answers.pop(0))

    page.on("dialog", reply)


def test_delete_dataset_and_workspace_need_the_typed_name(server, settings):
    base, _ = server
    add_user(settings, "alice", PASSWORD)
    problems: list[str] = []
    expect = sync_api.expect
    with sync_api.sync_playwright() as pw:
        browser, page = _browser(pw, problems)
        try:
            _login(page, base)
            _create(page, "Field cameras", "Night set")
            _answer_prompts(page, ["night set", "Night set", "Field cameras"])

            page.click("#delete-dataset")
            expect(page.locator("#toast")).to_contain_text("was not deleted")
            expect(page.locator("#dataset-select option:checked")).to_contain_text(
                "Night set"
            )

            page.click("#delete-dataset")
            expect(page.locator("#toast")).to_contain_text(
                'Dataset "Night set" deleted'
            )
            expect(page.locator("#dataset-select")).to_contain_text("(none yet)")

            page.click("#delete-workspace")
            expect(page.locator("#toast")).to_contain_text(
                'Workspace "Field cameras" deleted'
            )
            expect(page.locator("#topbar-workspace")).to_have_text(
                "No workspace selected"
            )
        finally:
            browser.close()
    assert problems == [], problems
    assert (
        _sql(settings, "SELECT id FROM workspaces WHERE name = 'Field cameras'") == []
    )


def test_delete_trained_dataset_lists_jobs_and_models_then_removes_them(server, settings):
    """Policy 2026-10-28: the confirmation names the job and model (and where the model
    is in use); after the typed name, all of it is deleted."""
    base, _ = server
    add_user(settings, "alice", PASSWORD)
    problems: list[str] = []
    prompts: list[str] = []
    expect = sync_api.expect
    with sync_api.sync_playwright() as pw:
        browser, page = _browser(pw, problems)
        try:
            _login(page, base)
            _create(page, "Field cameras", "Trained set")
            (ds_id,) = _sql(
                settings, "SELECT id FROM datasets WHERE name = 'Trained set'"
            )[0]
            (snap_id,) = _sql(
                settings,
                "INSERT INTO snapshots (dataset_id, content_hash, item_count, label_counts)"
                " VALUES (%s, 'x', 0, '{}') RETURNING id",
                (ds_id,),
            )[0]
            (job_id,) = _sql(
                settings,
                "INSERT INTO jobs (pipeline, snapshot_id, state) VALUES ('fake_cls', %s,"
                " 'succeeded') RETURNING id",
                (snap_id,),
            )[0]
            (model_id,) = _sql(
                settings,
                "INSERT INTO model_versions (job_id, pipeline, snapshot_id, format,"
                " object_key, sha256, name) VALUES (%s, 'fake_cls', %s, 'onnx-bundle',"
                " 'models/none', 'x', 'Night owls v1') RETURNING id",
                (job_id, snap_id),
            )[0]
            _sql(
                settings,
                "INSERT INTO deployments (model_id, consumer, project, activated_by)"
                " VALUES (%s, 'studio', '7', 'alice') RETURNING id",
                (model_id,),
            )

            def reply(dialog):
                prompts.append(dialog.message)
                dialog.accept("Trained set")

            page.on("dialog", reply)
            page.click("#delete-dataset")
            expect(page.locator("#dataset-select option", has_text="Trained set")).to_have_count(0)
        finally:
            browser.close()
    assert problems == [], problems
    (text,) = prompts
    assert "1 training job" in text
    assert "Night owls v1 (IN USE: installed in Studio project 7)" in text
    assert "Sparrow Studio keeps the copy" in text
    assert _sql(settings, "SELECT id FROM datasets WHERE id = %s", (ds_id,)) == []
    assert _sql(settings, "SELECT id FROM model_versions WHERE id = %s", (model_id,)) == []
