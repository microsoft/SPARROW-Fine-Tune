"""Browser pins for the M2 manual-test defects (2026-10-21).

M2-1 the sign-in card is centered in the window, not offset by the hidden sidebar.
M2-2 "Upload images" / "Import annotations" open the file picker themselves.
M2-3 uploads show progress and a running count; the dataset shows its image total.
M2-4 uploaded images appear in the review grid before any annotations are imported.
"""

from __future__ import annotations

import json

import pytest

from .helpers import png_bytes
from .service_helpers import PASSWORD, add_user
from . import test_ui_end_to_end as _e2e
from .test_backlog_service_ui import _browser, _login, _sql

server = _e2e.server

sync_api = pytest.importorskip("playwright.sync_api")


def _dataset(page, task="classification"):
    expect = sync_api.expect
    page.click("#show-new-workspace")
    page.fill("#new-workspace", "Field cameras")
    page.click("#create-workspace")
    expect(page.locator("#topbar-workspace")).to_have_text("Field cameras")
    page.click("#show-new-dataset")
    page.fill("#new-dataset", "Night set")
    page.select_option("#new-dataset-task", task)
    page.click("#create-dataset")
    expect(page.locator("#topbar-dataset")).to_contain_text("Night set")


def _images(tmp_path, n):
    paths = []
    for i in range(n):
        p = tmp_path / f"img{i}.png"
        p.write_bytes(png_bytes(seed=i, size=(48, 40)))
        paths.append(str(p))
    return paths


@pytest.mark.parametrize("width", [1400, 1920])
def test_m2_login_card_is_centered(server, settings, width):
    base, _ = server
    add_user(settings, "alice", PASSWORD)
    problems: list[str] = []
    with sync_api.sync_playwright() as pw:
        browser, page = _browser(pw, problems)
        try:
            page.set_viewport_size({"width": width, "height": 900})
            page.goto(base + "/")
            sync_api.expect(page.locator("#login-form")).to_be_visible()
            box = page.locator(".login-card").bounding_box()
            center = box["x"] + box["width"] / 2
            assert abs(center - width / 2) <= 2, (center, width)
        finally:
            browser.close()
    assert problems == [], problems


def test_m2_upload_button_opens_picker_and_shows_progress(server, settings, tmp_path):
    base, worker_errors = server
    add_user(settings, "alice", PASSWORD)
    files = _images(tmp_path, 3)
    problems: list[str] = []
    expect = sync_api.expect
    with sync_api.sync_playwright() as pw:
        browser, page = _browser(pw, problems)
        try:
            _login(page, base)
            _dataset(page)
            expect(page.locator("#dataset-image-count")).to_contain_text("0 images")
            with page.expect_file_chooser() as fc:
                page.click("#upload-btn")
            assert fc.value.is_multiple()
            fc.value.set_files(files)
            expect(page.locator("#upload-status")).to_contain_text("3 / 3 uploaded")
            expect(page.locator("#upload-progress")).to_have_attribute("value", "3")
            expect(page.locator("#dataset-image-count")).to_contain_text("3 images")
            expect(page.locator("#dataset-image-count")).to_contain_text(
                "3 not labelled"
            )

            # M2-4: the uploaded images are in the grid, with thumbnails, before labels.
            cards = page.locator(".example-card")
            expect(cards).to_have_count(3)
            expect(cards.first).to_contain_text("Not labelled")
            page.wait_for_function(
                "() => [...document.querySelectorAll('.example-card img')]"
                ".every((i) => i.complete && i.naturalWidth > 0)"
            )

            annotations = tmp_path / "labels.json"
            annotations.write_text(
                json.dumps(
                    {
                        "items": [
                            {"image": "img0.png", "annotations": [{"label": "deer"}]},
                            {"image": "img1.png", "annotations": [{"label": "fox"}]},
                        ]
                    }
                )
            )
            page.select_option("#annotation-format", "sparrow")
            with page.expect_file_chooser() as fc:
                page.click("#import-annotations-btn")
            fc.value.set_files(str(annotations))
            expect(page.locator("#annotations-status")).to_contain_text(
                "Imported labels for 2 images"
            )
            expect(page.locator("#dataset-image-count")).to_contain_text(
                "1 not labelled"
            )
            expect(page.locator("#count-pending")).to_have_text("2")
            expect(cards).to_have_count(3)

            # The pending filter shows labelled items only; unlabelled has its own filter.
            page.select_option("#example-filter", "pending")
            expect(cards).to_have_count(2)
            page.select_option("#example-filter", "unlabelled")
            expect(cards).to_have_count(1)

            # Labelling an unlabelled card creates a confirmed item.
            card = cards.first
            card.locator("[data-label-input]").fill("badger")
            card.locator('[data-example-action="correct"]').click()
            expect(page.locator("#count-confirmed")).to_have_text("1")
            expect(page.locator("#dataset-image-count")).to_contain_text(
                "0 not labelled"
            )
            expect(page.locator("#error-banner")).to_be_hidden()
        except Exception:
            page.screenshot(path=str(tmp_path / "failure.png"), full_page=True)
            raise
        finally:
            browser.close()
    rows = _sql(
        settings,
        "SELECT i.name, it.review_status, a.label FROM dataset_items it"
        " JOIN dataset_images i ON i.id = it.image_id"
        " JOIN annotations a ON a.item_id = it.id ORDER BY i.name",
    )
    assert rows == [
        ("img0.png", "pending", "deer"),
        ("img1.png", "pending", "fox"),
        ("img2.png", "confirmed", "badger"),
    ], rows
    assert not worker_errors, worker_errors
    assert problems == [], problems


def test_species_labels_on_detection_dataset_offer_a_move(server, settings, tmp_path):
    """Species-only labels uploaded to a detection dataset: the app offers to move those
    images to a (new) classification dataset and imports their labels there; the file's
    entries with boxes still land in the detection dataset."""
    base, worker_errors = server
    add_user(settings, "alice", PASSWORD)
    files = _images(tmp_path, 3)
    problems: list[str] = []
    expect = sync_api.expect
    with sync_api.sync_playwright() as pw:
        browser, page = _browser(pw, problems)
        try:
            _login(page, base)
            _dataset(page, task="detection")
            with page.expect_file_chooser() as fc:
                page.click("#upload-btn")
            fc.value.set_files(files)
            expect(page.locator("#upload-status")).to_contain_text("3 / 3 uploaded")
            labels = tmp_path / "labels.json"
            labels.write_text(
                json.dumps(
                    {
                        "items": [
                            {"image": "img0.png", "annotations": [{"label": "deer"}]},
                            {"image": "img1.png", "annotations": [{"label": "fox"}]},
                            {
                                "image": "img2.png",
                                "annotations": [
                                    {"label": "boar", "bbox": [4, 4, 20, 16]}
                                ],
                            },
                        ]
                    }
                )
            )
            page.select_option("#annotation-format", "sparrow")
            with page.expect_file_chooser() as fc:
                page.click("#import-annotations-btn")
            fc.value.set_files(str(labels))
            chooser = page.locator("#choice-dialog")
            expect(chooser).to_contain_text("These labels have no boxes")
            chooser.get_by_role("button", name="Move the 2 images").click()
            status = page.locator("#annotations-status")
            expect(status).to_contain_text("Moved 2 images to Field cameras (classification)")
            expect(status).to_contain_text("imported labels for 2 images")
            expect(status).to_contain_text("Labels with boxes for 1 image went to Night set")
            expect(page.locator("#dataset-select option:checked")).to_contain_text(
                "classification"
            )
            expect(page.locator("#error-banner")).to_be_hidden()
        except Exception:
            page.screenshot(path=str(tmp_path / "failure.png"), full_page=True)
            raise
        finally:
            browser.close()
    rows = _sql(
        settings,
        "SELECT d.task, i.name, a.label FROM dataset_items it"
        " JOIN dataset_images i ON i.id = it.image_id JOIN datasets d ON d.id = i.dataset_id"
        " JOIN annotations a ON a.item_id = it.id ORDER BY i.name",
    )
    assert rows == [
        ("classification", "img0.png", "deer"),
        ("classification", "img1.png", "fox"),
        ("detection", "img2.png", "boar"),
    ], rows
    assert not worker_errors, worker_errors
    assert problems == [], problems
