"""Browser pins for the workspace/dataset picker (user request 2026-10-21): the current
workspace and dataset are shown prominently in two matching panels, and the name inputs
appear only after the user asks to create one."""

from __future__ import annotations

import pytest

from .service_helpers import PASSWORD, add_user
from . import test_ui_end_to_end as _e2e
from .test_backlog_service_ui import _browser, _login

server = _e2e.server

sync_api = pytest.importorskip("playwright.sync_api")


def test_current_choice_is_prominent_and_create_forms_open_on_demand(server, settings):
    base, _ = server
    add_user(settings, "alice", PASSWORD)
    problems: list[str] = []
    expect = sync_api.expect
    with sync_api.sync_playwright() as pw:
        browser, page = _browser(pw, problems)
        try:
            _login(page, base)
            expect(page.locator("#current-workspace")).to_have_text("No workspace yet")
            expect(page.locator("#new-workspace")).to_be_hidden()
            expect(page.locator("#new-dataset")).to_be_hidden()
            expect(page.locator("#show-new-dataset")).to_be_disabled()

            page.click("#show-new-workspace")
            expect(page.locator("#new-workspace")).to_be_focused()
            page.click("#cancel-new-workspace")
            expect(page.locator("#new-workspace")).to_be_hidden()

            page.click("#show-new-workspace")
            page.fill("#new-workspace", "Field cameras")
            page.press("#new-workspace", "Enter")
            expect(page.locator("#current-workspace")).to_have_text("Field cameras")
            expect(page.locator("#new-workspace")).to_be_hidden()
            expect(page.locator("#current-workspace-meta")).to_have_text("0 datasets")

            page.click("#show-new-dataset")
            page.fill("#new-dataset", "Night set")
            page.select_option("#new-dataset-task", "detection")
            page.click("#create-dataset")
            expect(page.locator("#current-dataset")).to_have_text("Night set")
            expect(page.locator("#current-dataset-meta")).to_have_text(
                "Detection · 0 images"
            )
            expect(page.locator("#new-dataset")).to_be_hidden()
            expect(page.locator("#current-workspace-meta")).to_have_text("1 dataset")

            size = page.locator("#current-dataset").evaluate(
                "e => parseFloat(getComputedStyle(e).fontSize)"
            )
            assert size >= 18, size

            # FT-33: the chosen dataset survives a reload too.
            page.click("#show-new-dataset")
            page.fill("#new-dataset", "Day set")
            page.click("#create-dataset")
            expect(page.locator("#current-dataset")).to_have_text("Day set")
            page.select_option("#dataset-select", label="Night set (detection)")
            expect(page.locator("#current-dataset")).to_have_text("Night set")
            page.reload()
            expect(page.locator("#current-workspace")).to_have_text("Field cameras")
            expect(page.locator("#current-dataset")).to_have_text("Night set")
        finally:
            browser.close()
    assert problems == [], problems


def test_workspace_section_sits_in_the_sidebar_above_prepare_data_and_folds(server, settings):
    """2026-10-28: workspace create/select moved to a foldable sidebar section above Prepare data."""
    base, _ = server
    add_user(settings, "alice", PASSWORD)
    problems: list[str] = []
    expect = sync_api.expect
    with sync_api.sync_playwright() as pw:
        browser, page = _browser(pw, problems)
        try:
            _login(page, base)
            panel = page.locator(".sidebar #workspace-panel")
            expect(panel).to_be_visible()
            assert page.locator(".view #workspace-panel").count() == 0
            prepare = page.locator('.sidebar-nav [data-page="data"]')
            assert panel.bounding_box()["y"] + panel.bounding_box()["height"] <= prepare.bounding_box()["y"] + 1

            page.click("#show-new-workspace")
            page.fill("#new-workspace", "Field cameras")
            page.press("#new-workspace", "Enter")
            expect(page.locator("#current-workspace")).to_have_text("Field cameras")
            page.click("#show-new-workspace")
            page.fill("#new-workspace", "Lab set")
            page.click("#create-workspace")
            expect(page.locator("#current-workspace")).to_have_text("Lab set")
            # Nothing in the sidebar is clipped by its 240px width.
            for sel in ("#workspace-select", "#show-new-workspace", "#delete-workspace"):
                box = page.locator(sel).bounding_box()
                assert box["x"] + box["width"] <= 241, (sel, box)

            # Folding hides the controls but keeps the current workspace name in view.
            page.click("#workspace-panel > summary")
            for sel in ("#workspace-select", "#show-new-workspace", "#delete-workspace"):
                expect(page.locator(sel)).to_be_hidden()
            expect(page.locator("#current-workspace")).to_be_visible()
            expect(page.locator("#current-workspace")).to_have_text("Lab set")
            assert prepare.bounding_box()["y"] < 260
            page.reload()
            expect(page.locator("#current-workspace")).to_have_text("Lab set")  # FT-33: selection remembered
            expect(page.locator("#show-new-workspace")).to_be_hidden()  # fold remembered
            page.click("#workspace-panel > summary")
            expect(page.locator("#show-new-workspace")).to_be_visible()
            # 2026-11-04: the fold state is stored in the same turn as the click. It used to be
            # written from the asynchronous `toggle` event, so a quick reload could lose it.
            click_and_read = (
                "() => { document.querySelector('#workspace-panel > summary').click();"
                " return localStorage.getItem('sparrow-ft.workspace-folded'); }"
            )
            assert page.evaluate(click_and_read) == "1"
            assert page.evaluate(click_and_read) == "0"

            # Usable from any page.
            page.click('.sidebar-nav [data-page="models"]')
            page.select_option("#workspace-select", label="Field cameras")
            expect(page.locator("#models-scope")).to_contain_text("Field cameras")
            expect(page.locator("#topbar-workspace")).to_have_text("Field cameras")
            page.reload()
            expect(page.locator("#current-workspace")).to_have_text("Field cameras")

            # A remembered workspace that no longer exists falls back to the first one.
            page.evaluate("localStorage.setItem('sparrow-ft.workspace', '999999')")
            page.reload()
            expect(page.locator("#current-workspace")).to_have_text("Field cameras")
        finally:
            browser.close()
    assert problems == [], problems
