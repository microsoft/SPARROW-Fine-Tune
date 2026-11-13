"""Browser pins for the M7 manual-test defects (2026-10-21).

M7-1 the Studio project is picked by name from a list; the user never types an id.
M7-2 an error from the Studio panel, far down the page, is shown inside the window.
SP-1 (2026-10-25) the Studio project is chosen at the top, beside Workspace and
     Dataset; Studio import and Studio models stay greyed out until it is, and choosing
     it lists that project's models without a further click.
SP-3 (2026-10-25) the Studio model dropdown is the only base-model picker; the
     training panel shows the chosen model's name, architecture and classes, and
     "Clear selection" (which replaced the model-list Refresh) goes back to ImageNet.
M8-1 "Activate in Studio" also takes the Studio project from a list by name; the
     free-text project field (default "default") is only for local activation.
M8-3 after "Activate in Studio" was clicked the page gave no visible response: the
     button must show it is working, and the model must keep showing Studio's install
     status (pending -> installed/failed) instead of a toast that vanishes.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time

import pytest

from sparrow_finetune.connectors import studio as studio_mod

from .service_helpers import PASSWORD, add_user, make_service_client, studio_settings
from .studio_fake import FakeStudio
from .test_engine_package import FAKE_ENGINE
from .test_studio_connector import trained_model
from .test_backlog_service_ui import _browser, _login
from .test_m2_upload_ui import _dataset

sync_api = pytest.importorskip("playwright.sync_api")


@pytest.fixture
def studio():
    fake = FakeStudio()
    fake.extra_projects = [{"id": "2", "name": "Amazon review test"}]
    return fake


@pytest.fixture
def studio_server(settings, registry, studio):
    import uvicorn

    from sparrow_finetune.api import create_app

    s = studio_settings(settings)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(
        create_app(s, registry=registry, studio_transport=studio.transport),
        log_level="warning",
    ))
    web = threading.Thread(target=lambda: asyncio.run(server.serve(sockets=[sock])), daemon=True)
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


def _in_window(page, selector):
    box = page.locator(selector).bounding_box()
    height = page.viewport_size["height"]
    return box is not None and box["y"] >= 0 and box["y"] + box["height"] <= height


def test_m7_pick_studio_project_by_name_and_import(studio_server, settings, studio):
    add_user(settings, "alice", PASSWORD)
    problems: list[str] = []
    expect = sync_api.expect
    with sync_api.sync_playwright() as pw:
        browser, page = _browser(pw, problems)
        try:
            _login(page, studio_server)
            _dataset(page)
            options = page.locator("#studio-project option")
            expect(options).to_have_text([
                "Choose a Studio project",
                "Amazon review test (id 2)",
                "Project One (id p1)",
            ])
            page.select_option("#studio-project", label="Project One (id p1)")
            page.click("#studio-import-btn")
            expect(page.locator("#toast")).to_contain_text("Imported")
            expect(page.locator("#error-banner")).to_be_hidden()
            exports = studio.export_requests()
            assert exports and all("/projects/p1/" in r.url.path for r in exports)
        finally:
            browser.close()
    assert problems == [], problems


@pytest.mark.parametrize("size", [(1400, 700), (390, 700)])
def test_m7_studio_error_is_visible_in_window(studio_server, settings, size):
    add_user(settings, "alice", PASSWORD)
    problems: list[str] = []
    expect = sync_api.expect
    with sync_api.sync_playwright() as pw:
        browser, page = _browser(pw, problems)
        try:
            page.set_viewport_size({"width": size[0], "height": size[1]})
            _login(page, studio_server)  # no dataset yet
            page.select_option("#studio-project", label="Project One (id p1)")
            page.locator("#studio-import-btn").scroll_into_view_if_needed()
            page.click("#studio-import-btn")
            expect(page.locator("#error-text")).to_have_text("Create or select a dataset first.")
            # Both the message and the button the user just pressed are on screen.
            assert _in_window(page, "#error-banner")
            assert _in_window(page, "#studio-import-btn")
        finally:
            browser.close()
    assert problems == [], problems


def test_sp1_studio_project_is_a_top_level_scope(studio_server, settings, studio):
    from .test_base_models import studio_manifest

    studio.add_model(studio_manifest(b"w"), b"bundle")  # pushed to project p1
    add_user(settings, "alice", PASSWORD)
    problems: list[str] = []
    expect = sync_api.expect
    with sync_api.sync_playwright() as pw:
        browser, page = _browser(pw, problems)
        try:
            _login(page, studio_server)
            _dataset(page)
            panel = page.locator(".scope-picker #studio-scope-panel")
            expect(panel).to_be_visible()
            # Same row as the dataset panel on a wide window.
            ws = page.locator("#dataset-panel").bounding_box()
            assert abs(panel.bounding_box()["y"] - ws["y"]) < 2
            expect(page.locator("#current-studio-project")).to_have_text("None selected")
            for sel in ("#studio-import-btn", "#studio-reviewed-only",
                        "#studio-model-select", "#studio-model-use"):
                expect(page.locator(sel)).to_be_disabled()
            expect(page.locator("#studio-hint")).to_be_visible()
            expect(page.locator("#studio-body")).to_be_hidden()

            page.select_option("#studio-project", label="Project One (id p1)")
            expect(page.locator("#current-studio-project")).to_have_text("Project One")
            for sel in ("#studio-import-btn", "#studio-reviewed-only"):
                expect(page.locator(sel)).to_be_enabled()
            expect(page.locator("#studio-hint")).to_be_hidden()
            expect(page.locator("#studio-body")).to_be_visible()
            expect(page.locator("#studio-import-btn")).to_be_visible()
            expect(page.locator("#studio-model-select")).to_be_visible()
            panel_box = page.locator("#studio-panel").bounding_box()
            clear_box = page.locator("#studio-model-clear").bounding_box()
            assert clear_box["x"] + clear_box["width"] <= panel_box["x"] + panel_box["width"]
            # Models are listed without pressing a button, with a visible count.
            expect(page.locator("#studio-model-select option:not([value=''])")).to_have_count(1)
            expect(page.locator("#studio-models-status")).to_contain_text("1 model")

            page.select_option("#studio-project", value="")
            expect(page.locator("#current-studio-project")).to_have_text("None selected")
            expect(page.locator("#studio-body")).to_be_hidden()
            expect(page.locator("#studio-hint")).to_be_visible()
            expect(page.locator("#studio-import-btn")).to_be_disabled()
            expect(page.locator("#studio-model-select option:not([value=''])")).to_have_count(0)
            expect(page.locator("#studio-model-use")).to_be_disabled()
            expect(page.locator("#error-banner")).to_be_hidden()
        finally:
            browser.close()
    assert problems == [], problems


def _register_classifier(registry):
    """Register a stand-in under the image classifier id, a base-model pipeline."""
    from .fakes import RecordingTrainer

    class ClassifierStandIn(RecordingTrainer):
        id = "conventional_classifier"
        name = "Image classifier (test stand-in)"

    registry.register(ClassifierStandIn())


def test_sp2_pick_one_of_several_studio_models_from_a_list(
    studio_server, settings, studio, registry
):
    from .test_base_models import seed_model

    _register_classifier(registry)

    older = seed_model(studio, name="Mammals v1", created_at="2026-10-01T00:00:00+00:00")
    newer = seed_model(studio, name="Mammals v2", created_at="2026-10-20T00:00:00+00:00")
    seed_model(studio, name="Boxes", pipeline="yolo_detector", created_at="2026-10-21T00:00:00+00:00")
    add_user(settings, "alice", PASSWORD)
    problems: list[str] = []
    expect = sync_api.expect
    with sync_api.sync_playwright() as pw:
        browser, page = _browser(pw, problems)
        try:
            _login(page, studio_server)
            _dataset(page)
            page.select_option("#pipeline-select", "conventional_classifier")
            page.select_option("#studio-project", label="Project One (id p1)")
            # "Reviewed only" qualifies the import, so it follows the import button.
            btn = page.locator("#studio-import-btn").bounding_box()
            box = page.locator("#studio-reviewed-only").bounding_box()
            assert box["x"] > btn["x"] + btn["width"] - 1
            sel = page.locator("#studio-model-select")
            expect(sel.locator("option:not([value=''])")).to_have_count(3)
            # Newest usable model first and preselected; detector models cannot be a base.
            expect(sel).to_have_value(newer)
            expect(sel.locator("option").nth(0)).to_contain_text("Mammals v2")
            expect(sel.locator("option", has_text="Boxes")).to_have_js_property("disabled", True)
            expect(page.locator("#studio-model-details")).to_contain_text("Mammals v2")

            page.select_option("#studio-model-select", older)
            expect(page.locator("#studio-model-details")).to_contain_text("Mammals v1")
            # No second picker and no manual list refresh.
            expect(page.locator("#base-model-select")).to_have_count(0)
            expect(page.locator("#studio-models-refresh")).to_have_count(0)
            info = page.locator("#base-model-info")
            expect(info).to_contain_text("ImageNet")
            clear = page.locator("#studio-model-clear")
            expect(clear).to_be_disabled()
            page.click("#studio-model-use")
            expect(info).to_contain_text("Mammals v1")
            expect(info).to_contain_text("resnet18")
            expect(info).to_contain_text("3 classes")
            expect(info).to_contain_text(f"Studio model {older}")
            expect(clear).to_be_enabled()
            page.click("#studio-model-clear")
            expect(info).to_contain_text("ImageNet")
            expect(info).not_to_contain_text("Mammals")
            expect(clear).to_be_disabled()
            expect(page.locator("#error-banner")).to_be_hidden()
        finally:
            browser.close()
    assert problems == [], problems


def test_sp4_base_model_choice_follows_the_pipeline(studio_server, settings, studio, registry):
    """SP-4 / FT-37: a Studio model is offered as a base only for a job of its own pipeline.

    The DINO pipeline warm-starts too (FT-37), so its picker is not greyed out, but an image
    classifier model cannot seed it: the option is disabled and the earlier choice is dropped.
    """
    from .test_base_models import seed_model

    from .fakes import PlannedOverrideTrainer

    registry.register(PlannedOverrideTrainer())
    _register_classifier(registry)
    seed_model(studio, name="Mammals v1")
    add_user(settings, "alice", PASSWORD)
    problems: list[str] = []
    expect = sync_api.expect
    with sync_api.sync_playwright() as pw:
        browser, page = _browser(pw, problems)
        try:
            _login(page, studio_server)
            _dataset(page)
            page.select_option("#studio-project", label="Project One (id p1)")
            sel = page.locator("#studio-model-select")
            expect(sel.locator("option:not([value=''])")).to_have_count(1)
            note = page.locator("#base-model-note")

            page.select_option("#pipeline-select", "conventional_classifier")
            page.click("#studio-model-use")
            expect(page.locator("#base-model-info")).to_contain_text("Mammals v1")
            expect(note).to_be_hidden()

            page.select_option("#pipeline-select", "dinov3_linear")
            expect(page.locator("#base-model-field")).to_be_visible()
            expect(note).to_be_hidden()
            expect(sel.locator("option:not([value=''])")).to_contain_text("model")
            expect(sel.locator("option:not([value=''])")).to_be_disabled()
            expect(sel).to_be_disabled()
            expect(page.locator("#studio-model-use")).to_be_disabled()
            expect(page.locator("#base-model-info")).not_to_contain_text("Mammals v1")
            # Studio data import does not depend on the pipeline.
            expect(page.locator("#studio-import-btn")).to_be_enabled()

            page.select_option("#pipeline-select", "conventional_classifier")
            expect(sel).to_be_enabled()
            expect(page.locator("#studio-model-use")).to_be_enabled()
            expect(page.locator("#error-banner")).to_be_hidden()
        finally:
            browser.close()
    assert problems == [], problems


def test_sp5_yolo_detector_pipeline_offers_only_detector_models(
    studio_server, settings, studio, registry
):
    """A YOLO detector job starts only from a Studio YOLO model; classifier models are disabled."""
    from .fakes import RecordingTrainer
    from .test_base_models import seed_model

    class DetectorStandIn(RecordingTrainer):
        id = "yolo_detector"
        name = "YOLO detector (test stand-in)"

    _register_classifier(registry)
    registry.register(DetectorStandIn())
    seed_model(studio, name="Mammals v1", created_at="2026-10-01T00:00:00+00:00")
    boxes = seed_model(
        studio, name="Boxes", pipeline="yolo_detector", created_at="2026-09-01T00:00:00+00:00"
    )
    add_user(settings, "alice", PASSWORD)
    problems: list[str] = []
    expect = sync_api.expect
    with sync_api.sync_playwright() as pw:
        browser, page = _browser(pw, problems)
        try:
            _login(page, studio_server)
            _dataset(page)
            page.select_option("#pipeline-select", "conventional_classifier")
            page.select_option("#studio-project", label="Project One (id p1)")
            sel = page.locator("#studio-model-select")
            expect(sel.locator("option:not([value=''])")).to_have_count(2)
            page.click("#studio-model-use")
            info = page.locator("#base-model-info")
            expect(info).to_contain_text("Mammals v1")

            page.select_option("#pipeline-select", "yolo_detector")
            expect(page.locator("#base-model-note")).to_be_hidden()
            expect(sel).to_have_value(boxes)
            expect(sel.locator("option", has_text="Mammals v1")).to_have_js_property("disabled", True)
            # The classifier base does not carry over to a detector job.
            expect(info).to_contain_text("None")
            expect(page.locator("#studio-model-use")).to_be_enabled()
            expect(page.locator("#error-banner")).to_be_hidden()
        finally:
            browser.close()
    assert problems == [], problems


def test_m8_activate_in_studio_picks_project_by_name(
    studio_server, settings, registry, studio, monkeypatch
):
    monkeypatch.setattr(studio_mod, "build_engine_package", lambda *a, **k: FAKE_ENGINE)
    with make_service_client(studio_settings(settings), registry, studio.transport) as client:
        model = trained_model(client, settings, registry)  # before a user turns auth on
    add_user(settings, "alice", PASSWORD)
    problems: list[str] = []
    expect = sync_api.expect
    with sync_api.sync_playwright() as pw:
        browser, page = _browser(pw, problems)
        try:
            _login(page, studio_server)
            page.click('.sidebar-nav [data-page="models"]')
            expect(page.locator("#model-detail")).to_be_visible()
            pick = page.locator("#studio-deploy-project")
            expect(pick).to_be_visible()
            expect(pick.locator("option")).to_have_text([
                "Choose a Studio project",
                "Amazon review test (id 2)",
                "Project One (id p1)",
            ])
            page.click("#activate-studio")  # nothing chosen yet
            expect(page.locator("#error-text")).to_have_text("Choose a Studio project first.")
            assert studio.activations == []
            # With the same project chosen at the top, the pushed model shows up in the
            # model dropdown without a manual refresh.
            page.click('.sidebar-nav [data-page="data"]')
            page.select_option("#studio-project", label="Project One (id p1)")
            expect(page.locator("#studio-models-status")).to_contain_text("0 models")
            page.click('.sidebar-nav [data-page="models"]')
            page.select_option("#studio-deploy-project", label="Project One (id p1)")
            page.click("#activate-studio")
            expect(page.locator("#model-active-badge")).to_have_text(
                "Active: studio/Project One"
            )
            expect(page.locator("#toast")).to_contain_text("Studio is installing it")
            assert studio.activations, "Studio was not asked to activate"
            expect(page.locator("#studio-models-status")).to_contain_text("1 model")
            assert model["id"]
        finally:
            browser.close()
    assert problems == [], problems


def test_ft43_classifier_activation_shows_and_sends_the_detector(
    studio_server, settings, registry, studio, monkeypatch
):
    """FT-43: the page shows which detector the classifier will run on and sends the choice."""
    monkeypatch.setattr(studio_mod, "build_engine_package", lambda *a, **k: FAKE_ENGINE)
    studio.detectors.append({
        "pipeline_id": "11", "detector_key": "ft-p1-9", "name": "Fine-tuned animal detector",
        "fine_tuned": True, "detector_kind": "generic", "usable": True, "reason": None,
        "current": False,
    })
    with make_service_client(studio_settings(settings), registry, studio.transport) as client:
        trained_model(client, settings, registry)
    add_user(settings, "alice", PASSWORD)
    problems: list[str] = []
    expect = sync_api.expect
    with sync_api.sync_playwright() as pw:
        browser, page = _browser(pw, problems)
        try:
            _login(page, studio_server)
            page.click('.sidebar-nav [data-page="models"]')
            expect(page.locator("#model-detail")).to_be_visible()
            expect(page.locator("#studio-detector-label")).to_be_hidden()
            page.select_option("#studio-deploy-project", label="Project One (id p1)")
            sel = page.locator("#studio-detector")
            expect(sel).to_be_visible()
            expect(sel.locator("option")).to_have_text([
                "MegaDetector V6 (current)",
                "Fine-tuned species detector (fine-tuned) — unavailable: "
                "a fine-tuned species detector is used alone",
                "Fine-tuned animal detector (fine-tuned)",
            ])
            expect(sel).to_have_value("1")
            expect(sel.locator('option[value="9"]')).to_have_js_property("disabled", True)
            expect(page.locator("#studio-detector-note")).to_be_hidden()
            sel.select_option("11")
            expect(page.locator("#studio-detector-note")).to_contain_text(
                "Fine-tuned animal detector is a fine-tuned detector"
            )
            page.click("#activate-studio")
            expect(page.locator("#toast")).to_contain_text("Studio is installing it")
            assert studio.activation_bodies == [{"detector_pipeline_id": 11}]
            expect(page.locator("#error-banner")).to_be_hidden()
        finally:
            browser.close()
    assert problems == [], problems


def test_ft43_detector_choice_follows_the_model_not_the_dataset(
    studio_server, settings, registry, studio, monkeypatch
):
    """Only a classifier is paired with a detector: the picker hides for a detector model
    even while a classification dataset is the current one."""
    from sparrow_finetune.workers import run_once

    from .fakes import RecordingTrainer
    from .helpers import build_snapshot, create_dataset, models_for_job, queued_job

    class DetectorStandIn(RecordingTrainer):
        id = "fake_det"
        name = "Fake detector (tests)"
        task = "detection"

    registry.register(DetectorStandIn())
    monkeypatch.setattr(studio_mod, "build_engine_package", lambda *a, **k: FAKE_ENGINE)
    with make_service_client(studio_settings(settings), registry, studio.transport) as client:
        built = build_snapshot(client, task="detection")
        job = queued_job(client, "fake_det", built["snapshot"]["id"])
        assert run_once(settings, "w-det", {"cpu"}, registry=registry) is True
        (detector,) = models_for_job(client, job)
        # The user then works on a classification dataset in the same workspace.
        cls = create_dataset(client, built["workspace"]["id"], task="classification", name="Crops")
    add_user(settings, "alice", PASSWORD)
    problems: list[str] = []
    expect = sync_api.expect
    with sync_api.sync_playwright() as pw:
        browser, page = _browser(pw, problems)
        try:
            _login(page, studio_server)
            page.select_option("#dataset-select", str(cls["id"]))
            expect(page.locator("#topbar-dataset")).to_contain_text("Crops")
            page.click('.sidebar-nav [data-page="models"]')
            expect(page.locator("#model-name")).to_contain_text(f"Model #{detector['id']}")
            page.select_option("#studio-deploy-project", label="Project One (id p1)")
            page.wait_for_timeout(500)
            expect(page.locator("#studio-detector-label")).to_be_hidden()
            page.click("#activate-studio")
            expect(page.locator("#toast")).to_contain_text("Studio is installing it")
            assert studio.activation_bodies in ([None], [{}]), studio.activation_bodies
            expect(page.locator("#error-banner")).to_be_hidden()
        finally:
            browser.close()
    assert problems == [], problems


def test_m8_studio_cannot_run_model_is_reported(
    studio_server, settings, registry, studio, monkeypatch
):
    """M8-2: when no Engine package can be built, the page must not claim success."""

    def no_engine(*a, **k):
        raise studio_mod.EngineConversionError("pipeline has no ONNX export")

    monkeypatch.setattr(studio_mod, "build_engine_package", no_engine)
    with make_service_client(studio_settings(settings), registry, studio.transport) as client:
        trained_model(client, settings, registry)
    add_user(settings, "alice", PASSWORD)
    problems: list[str] = []
    expect = sync_api.expect
    with sync_api.sync_playwright() as pw:
        browser, page = _browser(pw, problems)
        try:
            _login(page, studio_server)
            page.click('.sidebar-nav [data-page="models"]')
            expect(page.locator("#studio-deploy-project")).to_be_visible()
            # With the same project chosen at the top, the pushed model shows up in the
            # model dropdown without a manual refresh.
            page.click('.sidebar-nav [data-page="data"]')
            page.select_option("#studio-project", label="Project One (id p1)")
            expect(page.locator("#studio-models-status")).to_contain_text("0 models")
            page.click('.sidebar-nav [data-page="models"]')
            page.select_option("#studio-deploy-project", label="Project One (id p1)")
            page.click("#activate-studio")
            expect(page.locator("#error-text")).to_contain_text(
                "Studio cannot run it: pipeline has no ONNX export"
            )
            assert _in_window(page, "#error-banner")
            assert studio.activations == []
        finally:
            browser.close()
    assert problems == [], problems


def test_m8_activation_shows_progress_and_install_status(
    studio_server, settings, registry, studio, monkeypatch
):
    monkeypatch.setattr(studio_mod, "build_engine_package", lambda *a, **k: FAKE_ENGINE)
    with make_service_client(studio_settings(settings), registry, studio.transport) as client:
        trained_model(client, settings, registry)
    add_user(settings, "alice", PASSWORD)
    problems: list[str] = []
    expect = sync_api.expect
    gate = threading.Event()
    studio.activate_gate = gate
    with sync_api.sync_playwright() as pw:
        browser, page = _browser(pw, problems)
        try:
            _login(page, studio_server)
            page.click('.sidebar-nav [data-page="models"]')
            expect(page.locator("#studio-deploy-project")).to_be_visible()
            # With the same project chosen at the top, the pushed model shows up in the
            # model dropdown without a manual refresh.
            page.click('.sidebar-nav [data-page="data"]')
            page.select_option("#studio-project", label="Project One (id p1)")
            expect(page.locator("#studio-models-status")).to_contain_text("0 models")
            page.click('.sidebar-nav [data-page="models"]')
            page.select_option("#studio-deploy-project", label="Project One (id p1)")
            page.click("#activate-studio")
            # While Studio is answering, the click visibly did something.
            expect(page.locator("#activate-studio")).to_be_disabled()
            expect(page.locator("#busy-overlay")).to_be_visible()
            expect(page.locator("#busy-text")).to_contain_text("Studio")
            gate.set()
            expect(page.locator("#busy-overlay")).to_be_hidden()
            expect(page.locator("#activate-studio")).to_be_enabled()
            status = page.locator("#studio-install-status")
            expect(status).to_contain_text("Project One")
            expect(status).to_contain_text("waiting for Studio to install")
            (studio_id,) = studio.activations
            studio.installs[studio_id] = {"install_status": "installed", "install_error": None}
            expect(status).to_contain_text("installed", timeout=15000)
            expect(status).not_to_contain_text("waiting")
            # The status survives leaving and coming back, unlike a toast.
            page.click('.sidebar-nav [data-page="data"]')
            page.click('.sidebar-nav [data-page="models"]')
            expect(status).to_contain_text("installed", timeout=15000)
        finally:
            gate.set()
            browser.close()
    assert problems == [], problems


def test_m8_failed_studio_install_is_shown(
    studio_server, settings, registry, studio, monkeypatch
):
    monkeypatch.setattr(studio_mod, "build_engine_package", lambda *a, **k: FAKE_ENGINE)
    with make_service_client(studio_settings(settings), registry, studio.transport) as client:
        trained_model(client, settings, registry)
    add_user(settings, "alice", PASSWORD)
    problems: list[str] = []
    expect = sync_api.expect
    with sync_api.sync_playwright() as pw:
        browser, page = _browser(pw, problems)
        try:
            _login(page, studio_server)
            page.click('.sidebar-nav [data-page="models"]')
            expect(page.locator("#studio-deploy-project")).to_be_visible()
            # With the same project chosen at the top, the pushed model shows up in the
            # model dropdown without a manual refresh.
            page.click('.sidebar-nav [data-page="data"]')
            page.select_option("#studio-project", label="Project One (id p1)")
            expect(page.locator("#studio-models-status")).to_contain_text("0 models")
            page.click('.sidebar-nav [data-page="models"]')
            page.select_option("#studio-deploy-project", label="Project One (id p1)")
            page.click("#activate-studio")
            status = page.locator("#studio-install-status")
            expect(status).to_contain_text("waiting for Studio to install")
            (studio_id,) = studio.activations
            studio.installs[studio_id] = {
                "install_status": "failed",
                "install_error": "engine load probe failed",
            }
            expect(status).to_contain_text("install failed: engine load probe failed", timeout=15000)
        finally:
            browser.close()
    assert problems == [], problems
