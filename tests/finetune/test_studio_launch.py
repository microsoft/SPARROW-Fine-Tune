"""Opening the fine-tune app from Sparrow Studio (2026-10-27 request).

SL-1 sign-in can be turned off (``SPARROW_FT_AUTH_DISABLED``): even with user accounts in
     the database, the API answers without a login and the page skips the login form.
SL-2 ``POST /v1/connectors/studio/load-project`` loads every reviewed Studio image of a
     project into a workspace named after the project, with one detection and one
     classification dataset; calling it again reuses them instead of creating copies.
SL-3 ``GET /v1/connectors/studio/import-status`` tells whether a dataset already holds
     everything Studio would import (the "all data imported / up to date" note).
SL-4 the page opened as ``/?studio_project=<id>&load=1`` selects that Studio project,
     loads the data, opens the new workspace and shows the up-to-date note.
"""

from __future__ import annotations

import asyncio
import dataclasses
import socket
import threading
import time

import pytest

from sparrow_finetune.config import Settings

from .service_helpers import PASSWORD, add_user, make_service_client, studio_settings
from .studio_fake import FakeStudio, studio_item


@pytest.fixture
def studio():
    return FakeStudio(project="p1", page_size=2)


@pytest.fixture
def open_settings(settings):
    return studio_settings(settings, auth_disabled=True)


@pytest.fixture
def svc(open_settings, registry, studio):
    with make_service_client(open_settings, registry, studio.transport) as client:
        yield client


# --------------------------------------------------------------------------- SL-1


def test_sl1_auth_disabled_from_env(monkeypatch):
    monkeypatch.setenv("SPARROW_FT_DATABASE_URL", "postgresql://x/y")
    monkeypatch.setenv("SPARROW_FT_AUTH_DISABLED", "1")
    assert Settings.from_env().auth_disabled is True
    monkeypatch.setenv("SPARROW_FT_AUTH_DISABLED", "")
    assert Settings.from_env().auth_disabled is False


def test_sl1_users_exist_but_sign_in_is_off(svc, open_settings):
    add_user(open_settings, "alice", PASSWORD)
    me = svc.get("/v1/auth/me").json()
    assert me["auth_required"] is False and me["method"] == "open"
    r = svc.post("/v1/workspaces", json={"name": "no login"})
    assert r.status_code in (200, 201), r.text


def test_sl1_sign_in_still_required_by_default(settings, registry, studio):
    add_user(settings, "alice", PASSWORD)
    with make_service_client(
        studio_settings(settings), registry, studio.transport
    ) as c:
        assert c.post("/v1/workspaces", json={"name": "x"}).status_code == 401


# --------------------------------------------------------------------------- SL-2


def _load(client, project="p1"):
    r = client.post("/v1/connectors/studio/load-project", json={"project": project})
    assert r.status_code == 200, r.text
    return r.json()


def test_sl2_load_project_creates_named_workspace_with_both_datasets(svc):
    body = _load(svc)
    assert body["workspace"]["name"] == "Project One"
    by_task = {d["task"]: d for d in body["datasets"]}
    assert set(by_task) == {"detection", "classification"}
    # default_items: three reviewed images with boxes (one has two boxes).
    assert by_task["detection"]["imported_images"] == 3
    assert by_task["detection"]["imported_items"] == 3
    assert by_task["classification"]["imported_items"] == 4
    listed = svc.get(f"/v1/workspaces/{body['workspace']['id']}/datasets").json()[
        "datasets"
    ]
    assert sorted(d["task"] for d in listed) == ["classification", "detection"]


def test_sl2_load_project_twice_reuses_workspace_and_datasets(svc):
    first = _load(svc)
    second = _load(svc)
    assert second["workspace"]["id"] == first["workspace"]["id"]
    assert {d["id"] for d in second["datasets"]} == {d["id"] for d in first["datasets"]}
    names = [w["name"] for w in svc.get("/v1/workspaces").json()["workspaces"]]
    assert names.count("Project One") == 1


def test_sl2_workspace_list_reports_linked_studio_project(svc):
    """The UI selects the linked workspace when reopened with ?studio_project= and no load."""
    plain = svc.post("/v1/workspaces", json={"name": "plain"}).json()
    linked = _load(svc)["workspace"]
    by_id = {w["id"]: w for w in svc.get("/v1/workspaces").json()["workspaces"]}
    assert by_id[linked["id"]]["studio_project"] == "p1"
    assert by_id[plain["id"]]["studio_project"] is None


def test_sl2_unknown_project_is_rejected(svc):
    r = svc.post("/v1/connectors/studio/load-project", json={"project": "nope"})
    assert r.status_code in (404, 502), r.text
    assert svc.get("/v1/workspaces").json()["workspaces"] == []


# --------------------------------------------------------------------------- SL-3


def _status(client, dataset_id, reviewed_only=True):
    r = client.get(
        "/v1/connectors/studio/import-status",
        params={
            "project": "p1",
            "dataset_id": dataset_id,
            "reviewed_only": str(reviewed_only).lower(),
        },
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_sl3_status_up_to_date_after_import_and_stale_after_studio_change(svc, studio):
    ws = svc.post("/v1/workspaces", json={"name": "w"}).json()
    ds = svc.post(
        f"/v1/workspaces/{ws['id']}/datasets", json={"name": "d", "task": "detection"}
    ).json()["id"]
    before = _status(svc, ds)
    assert before["up_to_date"] is False and before["last_import_at"] is None

    r = svc.post(
        "/v1/connectors/studio/import", json={"project": "p1", "dataset_id": ds}
    )
    assert r.status_code in (200, 201), r.text
    now = _status(svc, ds)
    assert now["up_to_date"] is True and now["last_import_at"] is not None
    assert now["studio_images"] == 3
    # Another filter is a different import.
    assert _status(svc, ds, reviewed_only=False)["up_to_date"] is False

    studio.items.append(studio_item(106, "f.png", [(9, "cat", [1, 1, 8, 8])]))
    assert _status(svc, ds)["up_to_date"] is False


def test_sl3_status_stale_when_an_imported_image_was_deleted_here(svc, settings):
    body = _load(svc)
    det = next(d for d in body["datasets"] if d["task"] == "detection")
    assert _status(svc, det["id"])["up_to_date"] is True
    from .service_helpers import db

    with db(settings) as conn:
        img = conn.execute(
            "SELECT id FROM dataset_images WHERE dataset_id = %s AND name = 'a.jpg'",
            (det["id"],),
        ).fetchone()["id"]
        conn.execute(
            "DELETE FROM annotations a USING dataset_items it"
            " WHERE a.item_id = it.id AND it.image_id = %s",
            (img,),
        )
        conn.execute("DELETE FROM dataset_items WHERE image_id = %s", (img,))
        conn.execute("DELETE FROM dataset_images WHERE id = %s", (img,))
    assert _status(svc, det["id"])["up_to_date"] is False


def test_sl3_deleting_a_loaded_dataset_still_works(svc):
    body = _load(svc)
    ws = body["workspace"]["id"]
    r = svc.delete(f"/v1/workspaces/{ws}")
    assert r.status_code in (200, 204), r.text


# --------------------------------------------------------------------------- SL-4


sync_api = pytest.importorskip("playwright.sync_api")


@pytest.fixture
def open_server(open_settings, registry, studio):
    import uvicorn

    from sparrow_finetune.api import create_app

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(
                open_settings, registry=registry, studio_transport=studio.transport
            ),
            log_level="warning",
        )
    )
    web = threading.Thread(
        target=lambda: asyncio.run(server.serve(sockets=[sock])), daemon=True
    )
    web.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "uvicorn did not start"
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        web.join(10)


def test_sl4_opened_from_studio_with_load(open_server, open_settings, studio):
    add_user(open_settings, "alice", PASSWORD)  # accounts exist, sign-in is still off
    problems: list[str] = []
    expect = sync_api.expect
    with sync_api.sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        page.set_default_timeout(30_000)
        page.on("pageerror", lambda exc: problems.append(f"pageerror: {exc}"))
        try:
            page.goto(open_server + "/?studio_project=p1&load=1")
            # Both datasets exist after a Studio load: the app asks which one to train.
            chooser = page.locator("#choice-dialog")
            expect(chooser).to_be_visible()
            expect(chooser).to_contain_text("What do you want to train?")
            chooser.get_by_role("button", name="A species classifier").click()
            expect(chooser).to_be_hidden()
            expect(page.locator("#dataset-select option:checked")).to_have_text(
                "Project One (classification)"
            )
            expect(page.locator("#view-data")).to_be_visible()
            expect(page.locator("#login-form")).to_be_hidden()
            expect(page.locator("#studio-project")).to_have_value("p1")
            expect(page.locator("#studio-uptodate")).to_be_visible()
            expect(page.locator("#studio-uptodate")).to_contain_text("up to date")
            ws_label = page.locator("#workspace-select option:checked")
            expect(ws_label).to_contain_text("Project One")
            # The launch parameters are consumed: a reload does not import again.
            assert "load=1" not in page.url

            def image_fetches():
                return [r for r in studio.requests if "/images/" in r.url.path]

            fetched = len(image_fetches())
            page.reload()
            expect(page.locator("#view-data")).to_be_visible()
            expect(
                page.locator("#studio-uptodate")
            ).to_be_visible()  # the status check only
            page.wait_for_timeout(500)
            assert len(image_fetches()) == fetched
            expect(page.locator("#error-banner")).to_be_hidden()
        finally:
            browser.close()
    assert problems == [], problems


def test_sl4_opened_from_studio_without_load_only_selects_project(open_server, studio):
    with sync_api.sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        page.set_default_timeout(20_000)
        try:
            page.goto(open_server + "/?studio_project=p1")
            expect = sync_api.expect
            expect(page.locator("#studio-project")).to_have_value("p1")
            page.wait_for_timeout(300)
            assert studio.export_requests() == []
        finally:
            browser.close()


def test_sl1_settings_flag_defaults_off(settings):
    assert dataclasses.replace(settings).auth_disabled is False
