"""FT-24 browser pins: box editor, per-class metrics, VOC import/export, assist, automatic batch,
calibrated threshold.

Drives the real page in Chromium against the uvicorn ``server`` fixture. Each test fails on a page
error or a console error, and saves a screenshot to ``$SPARROW_FT_SCREENSHOT_DIR`` (or tmp_path).
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from contextlib import contextmanager
from pathlib import Path

import psycopg
import pytest
from psycopg.types.json import Jsonb

from .fakes import RecordingTrainer
from .helpers import (
    build_snapshot,
    create_dataset,
    create_workspace,
    import_annotations,
    png_bytes,
    upload_image,
)
from .service_helpers import PASSWORD, add_user, login
from .test_backlog_service_ui import _login
from . import test_ui_end_to_end as _e2e

server = _e2e.server

sync_api = pytest.importorskip("playwright.sync_api")
expect = sync_api.expect


class FakeYolo(RecordingTrainer):
    """Stands in for the YOLO detector: takes ``batch`` and reports hardware like the real one."""

    id = "yolo_detector"
    name = "YOLO detector (test stand-in)"
    task = "detection"

    def validate_params(self, params):
        params = super().validate_params(params)
        batch = params.get("batch", 16)
        if batch != "auto" and not (isinstance(batch, int) and 1 <= batch <= 256):
            raise ValueError(
                f"batch must be an integer 1..256 or 'auto', got {batch!r}"
            )
        params["batch"] = batch
        return params

    def train(self, ctx):
        self.calls.append("train")
        auto = ctx.params.get("batch") == "auto"
        batch = 8 if auto else ctx.params.get("batch")
        ctx.emit(
            "train",
            f"training on cpu, batch {batch} ({'auto' if auto else 'set by parameter'})",
        )
        ctx.emit("train", "epoch 1", loss=0.5)


@pytest.fixture
def registry(trainers):
    from sparrow_finetune.pipelines import Registry

    reg = Registry()
    for trainer in trainers.all():
        reg.register(trainer)
    reg.register(FakeYolo())
    return reg


@pytest.fixture
def shots(tmp_path) -> Path:
    path = Path(os.environ.get("SPARROW_FT_SCREENSHOT_DIR") or tmp_path / "shots")
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _page(base, shot: Path):
    """Logged-in page; fails on page errors and console errors; screenshot on exit."""
    problems: list[str] = []
    with sync_api.sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        page.set_default_timeout(20_000)
        page.on("pageerror", lambda exc: problems.append(f"pageerror: {exc}"))
        page.on(
            "console",
            lambda m: m.type == "error" and problems.append(f"console: {m.text}"),
        )
        try:
            _login(page, base)
            yield page
            page.screenshot(path=str(shot), full_page=True)
        finally:
            browser.close()
    assert not problems, problems


def _sql(settings, query, params=()):
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        cur = conn.execute(query, params)
        return cur.fetchall() if cur.description else None


def _insert_model(settings, snapshot_id, pipeline, metrics=None, metadata=None) -> int:
    from sparrow_finetune.storage import LocalObjectStore

    key = f"models/ui-{pipeline}-{snapshot_id}-{len(metrics or {})}.zip"
    sha = LocalObjectStore(settings.objects_dir).put_bytes(key, b"fake bundle")
    job = _sql(
        settings,
        "INSERT INTO jobs (pipeline, snapshot_id, state) VALUES (%s, %s, 'succeeded') RETURNING id",
        (pipeline, snapshot_id),
    )[0][0]
    return _sql(
        settings,
        "INSERT INTO model_versions (job_id, pipeline, snapshot_id, format, object_key, sha256, metrics, metadata)"
        " VALUES (%s, %s, %s, 'onnx-bundle', %s, %s, %s, %s) RETURNING id",
        (
            job,
            pipeline,
            snapshot_id,
            key,
            sha,
            Jsonb(metrics or {}),
            Jsonb(metadata or {}),
        ),
    )[0][0]


def _detection_dataset(client, labelled=True):
    ws = create_workspace(client)
    ds = create_dataset(client, ws["id"], task="detection")
    upload_image(client, ds["id"], "a.png", seed=1)
    upload_image(client, ds["id"], "b.png", seed=2)
    if labelled:
        r = import_annotations(
            client,
            ds["id"],
            [
                {
                    "image": "a.png",
                    "annotations": [
                        {"label": "cat", "bbox": [4, 4, 20, 20]},
                        {"label": "dog", "bbox": [40, 20, 10, 10]},
                    ],
                }
            ],
        )
        assert r.status_code == 200, r.text
    return ds


def _items(client, ds_id):
    """Item rows via the API; logs the test client in (a user exists, so auth is on)."""
    login(client)
    return {
        it["image"]: it
        for it in client.get(f"/v1/datasets/{ds_id}/items").json()["items"]
    }


# ------------------------------------------------------------------ 1. box editor


def test_box_editor_keyboard_edit_saves_patch(server, client, settings, shots):
    base, _ = server
    ds = _detection_dataset(client)
    add_user(settings, "alice", PASSWORD)
    with _page(base, shots / "ft24-ui-box-editor.png") as page:
        card = page.locator('.example-card[data-image="a.png"]')
        card.get_by_role("button", name="Edit boxes").click()
        dialog = page.locator("#box-editor")
        expect(dialog).to_be_visible()
        expect(page.locator("#box-count")).to_have_text("2 boxes · box 1 selected")
        expect(page.locator("#box-label")).to_have_value("cat")

        page.focus("#box-stage")
        page.keyboard.press("ArrowRight")  # x 4 -> 5
        page.keyboard.press("Shift+ArrowDown")  # y 4 -> 14
        page.locator('.edit-box[data-box="1"]').click()
        expect(page.locator("#box-label")).to_have_value("dog")
        page.keyboard.press("Delete")
        expect(page.locator("#box-count")).to_have_text("1 box · box 1 selected")
        page.screenshot(path=str(shots / "ft24-ui-box-editor-dialog.png"))
        page.select_option("#box-label", label="+ New label…")
        page.fill("#box-new-label", "zebra")
        page.keyboard.press("Enter")  # adds the label; must not save the dialog
        expect(dialog).to_be_visible()
        expect(page.locator("#box-label")).to_have_value("zebra")
        expect(page.locator("#box-new-label-row")).to_be_hidden()
        expect(page.locator("#box-label-log")).to_contain_text('Added label "zebra"')
        expect(page.locator('.edit-box[data-box="0"] .edit-box-label')).to_have_text(
            "zebra"
        )
        with page.expect_request(
            lambda r: r.method == "PATCH" and "/items/a.png" in r.url
        ) as req:
            page.keyboard.press("Enter")
        assert req.value.post_data_json == {
            "annotations": [{"label": "zebra", "bbox": [5, 14, 20, 20]}],
            "review_status": "confirmed",
        }
        expect(dialog).to_be_hidden()
        expect(page.locator("#toast")).to_contain_text("Saved 1 box for a.png")
    item = _items(client, ds["id"])["a.png"]
    assert item["review_status"] == "confirmed"
    assert [(a["label"], a["bbox"]) for a in item["annotations"]] == [
        ("zebra", [5, 14, 20, 20])
    ]


def test_box_editor_escape_cancels_without_saving(server, client, settings, shots):
    base, _ = server
    ds = _detection_dataset(client)
    add_user(settings, "alice", PASSWORD)
    sent = []
    with _page(base, shots / "ft24-ui-box-editor-cancel.png") as page:
        page.on(
            "request",
            lambda r: (
                r.method in ("PATCH", "POST")
                and "/items/" in r.url
                and sent.append(r.url)
            ),
        )
        page.locator('.example-card[data-image="a.png"]').get_by_role(
            "button", name="Edit boxes"
        ).click()
        page.focus("#box-stage")
        page.keyboard.press("Delete")
        expect(page.locator("#box-count")).to_contain_text("1 box")
        page.keyboard.press("Escape")
        expect(page.locator("#box-editor")).to_be_hidden()
    assert sent == []
    assert len(_items(client, ds["id"])["a.png"]["annotations"]) == 2


def test_box_editor_draws_on_unlabelled_image(server, client, settings, shots):
    base, _ = server
    ds = _detection_dataset(client)
    add_user(settings, "alice", PASSWORD)
    with _page(base, shots / "ft24-ui-box-editor-draw.png") as page:
        card = page.locator('.example-card[data-image="b.png"]')
        card.get_by_role("button", name="Draw boxes").click()
        expect(page.locator("#box-count")).to_have_text("0 boxes")
        page.select_option("#box-label", label="+ New label…")
        page.fill("#box-new-label", "bird")
        page.click("#box-new-label-add")
        expect(page.locator("#box-label")).to_have_value("bird")
        stage = page.locator("#box-stage").bounding_box()
        page.mouse.move(
            stage["x"] + stage["width"] * 0.25, stage["y"] + stage["height"] * 0.25
        )
        page.mouse.down()
        page.mouse.move(
            stage["x"] + stage["width"] * 0.75,
            stage["y"] + stage["height"] * 0.75,
            steps=5,
        )
        page.mouse.up()
        expect(page.locator("#box-count")).to_have_text("1 box · box 1 selected")
        requests = []
        page.on(
            "request",
            lambda r: (
                "/v1/datasets/" in r.url
                and r.method in ("POST", "PATCH")
                and requests.append(r)
            ),
        )
        page.click("#box-save")
        expect(page.locator("#box-editor")).to_be_hidden()
        expect(page.locator("#toast")).to_contain_text("Saved 1 box for b.png")
        assert [r.method for r in requests] == ["POST", "PATCH"]
        posted = requests[0].post_data_json["items"]
        assert posted[0]["image"] == "b.png"
        (ann,) = posted[0]["annotations"]
        assert ann["label"] == "bird"
        assert ann["bbox"] == pytest.approx([16, 12, 32, 24], abs=1)
        assert requests[1].post_data_json == {"review_status": "confirmed"}
    item = _items(client, ds["id"])["b.png"]
    assert item["review_status"] == "confirmed"
    assert item["annotations"][0]["label"] == "bird"


def test_box_editor_label_is_a_dropdown_and_new_labels_carry_over(
    server, client, settings, shots
):
    base, _ = server
    _detection_dataset(client)
    add_user(settings, "alice", PASSWORD)
    with _page(base, shots / "ft24-ui-box-editor-labels.png") as page:
        page.locator('.example-card[data-image="a.png"]').get_by_role(
            "button", name="Edit boxes"
        ).click()
        options = page.locator("#box-label option")
        expect(options).to_have_text(["cat", "dog", "+ New label…"])
        expect(page.locator("#box-new-label-row")).to_be_hidden()
        page.select_option("#box-label", "dog")
        expect(page.locator('.edit-box[data-box="0"] .edit-box-label')).to_have_text(
            "dog"
        )
        page.select_option("#box-label", label="+ New label…")
        expect(page.locator("#box-new-label-row")).to_be_visible()
        expect(page.locator("#box-new-label")).to_be_focused()
        page.fill("#box-new-label", "okapi")
        page.click("#box-new-label-add")
        expect(options).to_have_text(["cat", "dog", "okapi", "+ New label…"])
        page.keyboard.press("Escape")  # closes without saving
        expect(page.locator("#box-editor")).to_be_hidden()
        page.locator('.example-card[data-image="b.png"]').get_by_role(
            "button", name="Draw boxes"
        ).click()
        expect(options).to_contain_text(["okapi"])
        page.select_option("#box-label", "okapi")
        expect(page.locator("#box-label")).to_have_value("okapi")


# ------------------------------------------------------------------ 2. per-class metrics


def test_per_class_table_and_overall_grid(server, client, settings, shots):
    base, _ = server
    snap = build_snapshot(client, task="detection", labels=("zebra", "lion"))[
        "snapshot"
    ]
    flat = {
        "map50": 0.71,
        "precision.zebra": 0.9,
        "recall.zebra": 0.8,
        "map50.zebra": 0.85,
        "ap50_95.zebra": 0.6,
        "instances.zebra": 12.0,
    }
    _insert_model(settings, snap["id"], "yolo_detector", metrics=flat)
    metrics = dict(flat, **{"precision.lion": 0.5})
    per_class = [
        {
            "label": "zebra",
            "precision": 0.9,
            "recall": 0.8,
            "map50": 0.85,
            "map50_95": 0.6,
            "instances": 12,
        },
        {
            "label": "lion",
            "precision": None,
            "recall": None,
            "map50": None,
            "map50_95": None,
            "instances": 0,
        },
    ]
    newest = _insert_model(
        settings,
        snap["id"],
        "yolo_detector",
        metrics=metrics,
        metadata={"per_class": per_class},
    )
    add_user(settings, "alice", PASSWORD)
    with _page(base, shots / "ft24-ui-per-class.png") as page:
        page.locator('.sidebar-nav [data-page="models"]').click()
        expect(page.locator("#per-class-section")).to_be_visible()
        zebra = page.locator('#per-class-body tr[data-label="zebra"] td')
        expect(zebra).to_have_text(
            ["zebra", "0.9000", "0.8000", "0.8500", "0.6000", "12"]
        )
        lion = page.locator('#per-class-body tr[data-label="lion"] td')
        expect(lion).to_have_text(["lion", "—", "—", "—", "—", "0"])
        grid = page.locator("#metrics-grid").inner_text().lower()
        assert "map50" in grid and "." not in grid.replace("0.71", "")
        assert "precision." not in page.locator("#compare-body").inner_text().lower()
        page.screenshot(path=str(shots / "ft24-ui-per-class-table.png"), full_page=True)
        page.locator(f'.model-list-item:not([data-model-id="{newest}"])').click()
        # Older model: only flat keys; ap50_95 fills the mAP50-95 column.
        expect(page.locator('#per-class-body tr[data-label="zebra"] td')).to_have_text(
            ["zebra", "0.9000", "0.8000", "0.8500", "0.6000", "12"]
        )
        expect(page.locator("#per-class-body tr")).to_have_count(1)
        expect(page.locator("#threshold-info")).to_be_hidden()


# ------------------------------------------------------------------ 3. VOC import / export


VOC_XML = """<annotation><filename>b.png</filename><size><width>64</width><height>48</height></size>
<object><name>owl</name><bndbox><xmin>2</xmin><ymin>3</ymin><xmax>12</xmax><ymax>13</ymax></bndbox></object>
</annotation>"""


def test_voc_export_link_and_import(server, client, settings, shots):
    base, _ = server
    ds = _detection_dataset(client)
    snap = client.post(f"/v1/datasets/{ds['id']}/snapshots").json()
    add_user(settings, "alice", PASSWORD)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("Annotations/b.xml", VOC_XML)
        zf.writestr("readme.txt", "not xml")
    with _page(base, shots / "ft24-ui-voc.png") as page:
        page.select_option("#annotation-format", "voc")
        export = page.locator("#export-annotations")
        expect(export).to_be_visible()
        expect(export).to_have_attribute("href", f"/v1/datasets/{ds['id']}/export/voc")
        resp = page.request.get(base + f"/v1/datasets/{ds['id']}/export/voc")
        assert resp.ok and zipfile.ZipFile(io.BytesIO(resp.body())).namelist()
        expect(page.locator("#snapshot-export")).to_have_attribute(
            "href", f"/v1/snapshots/{snap['id']}/export/voc"
        )
        assert page.request.get(base + f"/v1/snapshots/{snap['id']}/export/voc").ok

        page.check("#annotations-overwrite")
        with page.expect_request(lambda r: "/import/voc" in r.url) as req:
            page.set_input_files(
                "#annotations-file",
                files=[
                    {
                        "name": "voc.zip",
                        "mimeType": "application/zip",
                        "buffer": buf.getvalue(),
                    }
                ],
            )
        assert req.value.url.endswith("/import/voc?overwrite=true")
        assert req.value.headers["content-type"] == "application/zip"
        # Chromium does not expose File bodies; the stored box below proves the zip arrived.
        status = page.locator("#annotations-status")
        expect(status).to_contain_text("Imported 1 box for 1 image from voc.zip.")
        expect(status).to_contain_text("1 entry skipped: readme.txt (not an .xml file)")
        expect(status).to_contain_text("New labels: owl.")
        expect(page.locator('.example-card[data-image="b.png"]')).to_contain_text("owl")
    assert _items(client, ds["id"])["b.png"]["annotations"][0]["bbox"] == [2, 3, 10, 10]


def test_label_format_defaults_to_camtrapdp(server, client, settings, shots):
    base, _ = server
    ds = _detection_dataset(client)
    snap = client.post(f"/v1/datasets/{ds['id']}/snapshots").json()
    add_user(settings, "alice", PASSWORD)
    with _page(base, shots / "ft24-ui-label-format.png") as page:
        fmt = page.locator("#annotation-format")
        expect(fmt).to_have_value("camtrapdp")
        expect(page.locator("#annotation-format-help")).to_contain_text("Camera Trap DP 1.0 package")
        expect(page.locator("#annotation-format-warning")).to_contain_text("camera locations or capture times")
        expect(page.locator("#annotations-file")).to_have_attribute("accept", ".zip,application/zip")
        export = page.locator("#export-annotations")
        expect(export).to_have_text("Export")
        expect(export).to_have_attribute("title", "Download this dataset's labels as Camera Trap DP")
        imp = page.locator("#import-annotations-btn").bounding_box()
        exp_box = export.bounding_box()
        assert abs(exp_box["y"] - imp["y"]) < 2 and exp_box["x"] > imp["x"] + imp["width"]
        sel_box = fmt.bounding_box()
        over = page.locator("#annotations-overwrite-label")
        expect(over).to_have_text("Overwrite labels on confirmed images")
        assert over.bounding_box()["y"] >= sel_box["y"] + sel_box["height"]
        help_box = page.locator("#annotation-format-help").bounding_box()
        warn_box = page.locator("#annotation-format-warning").bounding_box()
        assert warn_box["y"] - (help_box["y"] + help_box["height"]) >= 8
        expect(export).to_have_attribute("href", f"/v1/datasets/{ds['id']}/export/camtrapdp")
        snap_link = page.locator("#snapshot-export")
        expect(snap_link).to_have_text("Download snapshot as Camera Trap DP")
        expect(snap_link).to_have_attribute("href", f"/v1/snapshots/{snap['id']}/export/camtrapdp")
        assert page.request.get(base + f"/v1/datasets/{ds['id']}/export/camtrapdp").ok

        # A .json file with a zip-only format is refused before any request.
        page.set_input_files(
            "#annotations-file",
            files=[{"name": "labels.json", "mimeType": "application/json", "buffer": b"{}"}],
        )
        expect(page.locator("#error-text")).to_contain_text("needs a .zip")

        page.select_option("#annotation-format", "coco")
        expect(page.locator("#annotation-format-warning")).to_be_hidden()
        expect(export).to_have_attribute("href", f"/v1/datasets/{ds['id']}/export/coco")
        expect(snap_link).to_have_text("Download snapshot as MS COCO")
        coco = json.dumps(
            {
                "images": [{"id": 1, "file_name": "a.png", "width": 64, "height": 48}],
                "categories": [{"id": 1, "name": "badger"}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 10, 10]}],
            }
        ).encode()
        with page.expect_request(lambda r: "/import/coco" in r.url) as req:
            page.set_input_files(
                "#annotations-file",
                files=[{"name": "coco.json", "mimeType": "application/json", "buffer": coco}],
            )
        assert req.value.url.endswith("/import/coco?overwrite=false")
        expect(page.locator("#annotations-status")).to_contain_text("Imported 1 box for 1 image from coco.json.")

        page.select_option("#annotation-format", "sparrow")
        expect(export).to_be_hidden()
        expect(snap_link).to_be_hidden()
        expect(page.locator("#annotations-overwrite-label")).to_be_hidden()
    labels = [a["label"] for a in _items(client, ds["id"])["a.png"]["annotations"]]
    assert labels == ["badger"]


def test_voc_controls_hidden_for_classification(server, client, settings, shots):
    base, _ = server
    built = build_snapshot(client)
    snap = built["snapshot"]
    add_user(settings, "alice", PASSWORD)
    with _page(base, shots / "ft24-ui-voc-classification.png") as page:
        expect(page.locator("#example-grid .example-card").first).to_be_visible()
        expect(page.locator('#annotation-format option[value="voc"]')).to_be_disabled()
        # FT-35/FT-38: classification datasets get assisted labelling; the help text follows the
        # whole-image setting (the fixture trains on whole images).
        expect(page.locator("#assist-block")).to_be_visible()
        expect(page.locator("#assist-help")).to_contain_text("trains on whole images")
        r = page.request.patch(
            f"{base}/v1/datasets/{built['dataset']['id']}",
            data={"whole_image": False},
            headers={"X-Sparrow-CSRF": "1"},
        )
        assert r.status == 200, r.text()
        page.reload()
        expect(page.locator("#assist-help")).to_contain_text("People and vehicles are left out")
        expect(page.locator("#snapshot-export")).to_have_attribute(
            "href", f"/v1/snapshots/{snap['id']}/export/camtrapdp"
        )


# ------------------------------------------------------------------ 4. assist


def _summary(upload=None, studio=None):
    zero = {"images": 0, "empty": 0, "with_boxes": 0, "confirmed": 0, "excluded": 0}
    body = {"sources": {"upload": dict(zero, **(upload or {})), "studio": dict(zero, **(studio or {}))}}
    return lambda route: route.fulfill(json=body)


def _enable_assist(page):
    expect(page.locator("#assist-body")).to_be_hidden()
    page.check("#assist-enable")
    expect(page.locator("#assist-body")).to_be_visible()


def _confirm_assist(page):
    page.click("#assist-btn")
    expect(page.locator("#assist-confirm")).to_be_visible()
    page.click("#assist-confirm-btn")


def test_assist_lists_detectors_only_and_chunks_by_500(server, client, settings, shots):
    base, _ = server
    ds = _detection_dataset(client)
    snap = client.post(f"/v1/datasets/{ds['id']}/snapshots").json()
    yolo = _insert_model(settings, snap["id"], "yolo_detector")
    _insert_model(settings, snap["id"], "conventional_classifier")
    add_user(settings, "alice", PASSWORD)
    names = [f"img{i:04d}.png" for i in range(1201)]
    bodies = []

    def images(route):
        route.fulfill(
            json={
                "images": [
                    {
                        "id": i + 1,
                        "name": n,
                        "sha256": "0" * 64,
                        "width": 64,
                        "height": 48,
                    }
                    for i, n in enumerate(names)
                ]
            }
        )

    def content(route):
        route.fulfill(body=png_bytes(0), content_type="image/png")

    def assist(route):
        body = route.request.post_data_json
        bodies.append(body)
        n = len(body["images"])
        route.fulfill(
            json={
                "model_id": body["model_id"],
                "conf": body["conf"],
                "processed": n,
                "items_written": n - 1,
                "boxes": 2 * (n - 1),
                "unknown_labels": ["bird"],
                "skipped": [{"image": body["images"][0], "reason": "has boxes"}],
                "items": [],
            }
        )

    with _page(base, shots / "ft24-ui-assist.png") as page:
        page.route("**/v1/datasets/*/images", images)
        page.route("**/v1/datasets/*/images/*/content", content)
        page.route("**/v1/datasets/*/assist", assist)
        page.route("**/v1/datasets/*/assist/summary", _summary({"images": 1201, "empty": 1201}))
        page.reload()
        expect(page.locator("#assist-block")).to_be_visible()
        _enable_assist(page)
        options = page.locator("#assist-model option")
        # No MegaDetector on this server: listed first but disabled; the trained detector is chosen.
        expect(options).to_have_count(2)
        expect(options.nth(0)).to_be_disabled()
        expect(options.nth(0)).to_contain_text("not installed on this server")
        expect(page.locator("#assist-model")).to_have_value(str(yolo))
        expect(page.locator("#assist-animal-row")).to_be_hidden()
        expect(page.locator("#assist-btn")).to_be_enabled()
        _confirm_assist(page)
        status = page.locator("#assist-status")
        expect(status).to_contain_text(
            "Added 2396 boxes to 1198 images for review; 3 images left unchanged."
        )
        expect(status).to_contain_text(
            "The model's labels bird are not used in this dataset"
        )
    assert [len(b["images"]) for b in bodies] == [500, 500, 201]
    assert [b["images"][0] for b in bodies] == [
        "img0000.png",
        "img0500.png",
        "img1000.png",
    ]
    assert sum((b["images"] for b in bodies), []) == names
    assert all(
        b == dict(b, model_id=yolo, conf=0.25, overwrite=False, sources=["upload"]) for b in bodies
    )


def test_assist_disabled_without_a_detector(server, client, settings, shots):
    base, _ = server
    ds = _detection_dataset(client)
    snap = client.post(f"/v1/datasets/{ds['id']}/snapshots").json()
    _insert_model(settings, snap["id"], "conventional_classifier")
    add_user(settings, "alice", PASSWORD)
    with _page(base, shots / "ft24-ui-assist-none.png") as page:
        page.check("#assist-enable")
        expect(page.locator("#assist-model")).to_contain_text("No detector available")
        expect(page.locator("#assist-btn")).to_be_disabled()


def test_assist_defaults_to_megadetector_and_sends_the_animal_label(
    server, client, settings, shots
):
    base, _ = server
    ds = _detection_dataset(client)
    snap = client.post(f"/v1/datasets/{ds['id']}/snapshots").json()
    yolo = _insert_model(settings, snap["id"], "yolo_detector")
    add_user(settings, "alice", PASSWORD)
    bodies = []

    def models(route):
        route.fulfill(
            json={
                "models": [
                    {
                        "id": "megadetector",
                        "name": "MegaDetector v6 (compact)",
                        "labels": ["animal", "person", "vehicle"],
                        "available": True,
                        "default": True,
                    },
                    {"id": yolo, "available": True, "default": False},
                ]
            }
        )

    def assist(route):
        body = route.request.post_data_json
        bodies.append(body)
        n = len(body["images"])
        route.fulfill(
            json={
                "model_id": body["model_id"],
                "conf": body["conf"],
                "processed": n,
                "items_written": n,
                "boxes": n,
                "unknown_labels": [],
                "skipped": [],
                "items": [],
            }
        )

    with _page(base, shots / "ft24-ui-assist-megadetector.png") as page:
        page.route("**/v1/assist/models", models)
        page.route("**/v1/datasets/*/assist", assist)
        page.route("**/v1/datasets/*/assist/summary", _summary({"images": 3, "empty": 3}))
        page.reload()
        _enable_assist(page)
        sel = page.locator("#assist-model")
        expect(sel).to_have_value("megadetector")
        expect(sel.locator("option").nth(0)).to_have_text("MegaDetector v6 (compact) (default)")
        expect(page.locator("#assist-animal-row")).to_be_visible()
        page.fill("#assist-animal-label", "deer")
        _confirm_assist(page)
        expect(page.locator("#assist-status")).to_contain_text("for review")
        sel.select_option(str(yolo))
        expect(page.locator("#assist-animal-row")).to_be_hidden()
    assert bodies and all(
        b["model_id"] == "megadetector" and b["animal_label"] == "deer" for b in bodies
    )


def test_assist_warns_and_confirms_before_touching_annotated_or_studio_images(
    server, client, settings, shots
):
    base, _ = server
    ds = _detection_dataset(client)
    snap = client.post(f"/v1/datasets/{ds['id']}/snapshots").json()
    yolo = _insert_model(settings, snap["id"], "yolo_detector")
    real = client.get(f"/v1/datasets/{ds['id']}/images").json()["images"]
    add_user(settings, "alice", PASSWORD)
    bodies = []
    # Tag the last image as Studio-imported in the browser's view of the dataset.
    tagged = [dict(i, source="studio" if n == len(real) - 1 else "upload") for n, i in enumerate(real)]

    def assist(route):
        body = route.request.post_data_json
        bodies.append(body)
        route.fulfill(json={"model_id": body["model_id"], "conf": body["conf"], "processed": 1,
                            "items_written": 1, "boxes": 1, "unknown_labels": [], "skipped": [], "items": []})

    with _page(base, shots / "ft24-ui-assist-confirm.png") as page:
        page.route("**/v1/datasets/*/images", lambda r: r.fulfill(json={"images": tagged}))
        page.route("**/v1/datasets/*/assist", assist)
        page.route(
            "**/v1/datasets/*/assist/summary",
            _summary({"images": 4, "empty": 1, "with_boxes": 1, "confirmed": 1, "excluded": 1},
                     {"images": 2, "empty": 1, "with_boxes": 1}),
        )
        page.reload()
        _enable_assist(page)
        expect(page.locator("#assist-model")).to_have_value(str(yolo))
        warnings = page.locator("#assist-warnings")
        expect(warnings.locator(".assist-warning")).to_have_count(1)
        expect(warnings).to_contain_text("2 images already have boxes or are confirmed. They are left alone")
        expect(warnings).to_contain_text("2 Studio-imported images are left out.")
        expect(page.locator("#assist-source-all-text")).to_have_text(
            "Uploaded and Studio-imported images (6 images, 2 from Studio)"
        )

        page.check('input[name="assist-source"][value="all"]')
        expect(warnings.locator(".assist-warning")).to_have_count(2)
        expect(warnings).to_contain_text("2 Studio-imported images are included.")
        page.check("#assist-overwrite")
        expect(warnings).to_contain_text("their boxes will be replaced by the detector's")

        page.click("#assist-btn")
        confirm = page.locator("#assist-confirm")
        expect(confirm).to_be_visible()
        expect(confirm).to_contain_text("5 images will get pending boxes, including Studio-imported images.")
        expect(confirm).to_contain_text("3 images will lose their current boxes.")
        expect(confirm).to_contain_text("1 excluded image is skipped.")
        expect(page.locator("#assist-btn")).to_be_disabled()
        page.click("#assist-cancel-btn")
        expect(confirm).to_be_hidden()
        assert bodies == []

        page.uncheck("#assist-overwrite")
        page.click("#assist-btn")
        expect(confirm).to_contain_text("2 images will get pending boxes")
        expect(confirm).to_contain_text("3 images with boxes or confirmed stay unchanged.")
        # Changing an option withdraws the pending confirmation.
        page.check('input[name="assist-source"][value="upload"]')
        expect(confirm).to_be_hidden()
        page.check('input[name="assist-source"][value="all"]')
        _confirm_assist(page)
        expect(page.locator("#assist-status")).to_contain_text("for review")
    assert len(bodies) == 1
    assert bodies[0]["sources"] == ["upload", "studio"] and bodies[0]["overwrite"] is False
    assert bodies[0]["images"] == [i["name"] for i in real]


# ------------------------------------------------------------------ 5. automatic batch


def _capture_job_params(page):
    """Answer job submissions in the page with an error and return a getter for the sent params.

    Stubbing fetch in the page (not page.route) keeps the refusal off the network, so the
    browser logs no console error for it.
    """
    page.evaluate("""() => {
        window.__sentParams = [];
        const real = window.fetch;
        window.fetch = (url, opts = {}) => {
            if (opts.method === "POST" && String(url).endsWith("/v1/jobs")) {
                window.__sentParams.push(JSON.parse(opts.body).params);
                return Promise.resolve(new Response(JSON.stringify({detail: "stopped by the test"}),
                    {status: 400, headers: {"content-type": "application/json"}}));
            }
            return real(url, opts);
        };
    }""")
    return lambda: page.evaluate("() => window.__sentParams")


def _open_settings(page):
    """Open the collapsed "Training settings" section of the training form."""
    expect(page.locator("#training-settings")).to_be_visible()
    if not page.locator("#training-settings").evaluate("d => d.open"):
        page.click("#training-settings summary")


def test_yolo_batch_auto_and_hardware_on_job(server, client, settings, shots):
    base, worker_errors = server
    build_snapshot(client, task="detection", labels=("cat", "dog"))
    add_user(settings, "alice", PASSWORD)
    with _page(base, shots / "ft24-ui-batch.png") as page:
        expect(page.locator("#pipeline-select")).to_have_value("yolo_detector")
        guide = page.locator("#pipeline-guide")
        expect(guide).to_be_visible()
        expect(guide).to_contain_text("Fine-tunes an Ultralytics YOLO detector")
        expect(guide.locator("ol li")).to_have_count(4)
        expect(guide).to_contain_text("mAP@50 and mAP@50-95")
        # The picker is a normal-height dropdown sitting above the explanation.
        select_box = page.locator("#pipeline-select").bounding_box()
        summary_box = page.locator(".pipeline-summary").bounding_box()
        assert select_box["height"] <= 48
        assert summary_box["y"] >= select_box["y"] + select_box["height"]
        _open_settings(page)
        expect(page.locator("#batch-field")).to_be_visible()
        expect(page.locator("#batch-size")).to_have_value("16")
        page.check("#batch-auto")
        expect(page.locator("#batch-size")).to_be_disabled()
        page.screenshot(path=str(shots / "ft24-ui-batch-form.png"), full_page=True)
        with page.expect_request(
            lambda r: r.method == "POST" and r.url.endswith("/v1/jobs")
        ) as req:
            page.click("#start-training")
        assert req.value.post_data_json["params"] == {
            "model": "yolo11n.pt",
            "batch": "auto",
            "imgsz": 640,
        }
        expect(page.locator("#view-jobs")).to_be_visible()
        expect(page.locator("#job-hardware")).to_have_text(
            "Hardware: cpu, batch 8 (auto)", timeout=30_000
        )
    assert not worker_errors


def test_job_events_list_has_fixed_height_and_scrolls(server, client, settings, shots):
    base, worker_errors = server
    build_snapshot(client, task="detection", labels=("cat", "dog"))
    add_user(settings, "alice", PASSWORD)
    with _page(base, shots / "ft24-ui-events.png") as page:
        page.click("#start-training")
        events = page.locator("#event-list")
        expect(events).to_be_visible()
        expect(events.locator("li").first).to_be_visible(timeout=30_000)
        before = events.bounding_box()["height"]
        page.evaluate("""() => {
            const list = document.querySelector('#event-list');
            for (let i = 0; i < 60; i++) {
                const li = document.createElement('li');
                li.innerHTML = '<time>now</time>filler event ' + i;
                list.prepend(li);
            }
        }""")
        after = events.bounding_box()["height"]
        assert after == before, (before, after)
        assert page.evaluate(
            "() => { const l = document.querySelector('#event-list');"
            " return l.scrollHeight > l.clientHeight && getComputedStyle(l).overflowY === 'auto'; }"
        )
    assert not worker_errors


def test_batch_field_only_for_yolo_and_number_sent(server, client, settings, shots):
    base, _ = server
    build_snapshot(client, task="detection", labels=("cat", "dog"))
    add_user(settings, "alice", PASSWORD)
    with _page(base, shots / "ft24-ui-batch-number.png") as page:
        _open_settings(page)
        page.fill("#batch-size", "300")
        page.click("#start-training")
        expect(page.locator("#error-text")).to_contain_text(
            "Batch size must be a whole number from 1 to 256."
        )
        page.fill("#batch-size", "4")
        with page.expect_request(
            lambda r: r.method == "POST" and r.url.endswith("/v1/jobs")
        ) as req:
            page.click("#start-training")
        assert req.value.post_data_json["params"] == {"model": "yolo11n.pt", "batch": 4, "imgsz": 640}
        expect(page.locator("#job-hardware")).to_contain_text("batch 4", timeout=30_000)


def test_yolo_starting_weights_offer_megadetector_and_follow_base_model(
    server, client, settings, shots
):
    base, _ = server
    snap = build_snapshot(client, task="detection", labels=("cat", "dog"))
    add_user(settings, "alice", PASSWORD)
    with _page(base, shots / "ft24-ui-start-weights.png") as page:
        field = page.locator("#start-weights-field")
        expect(field).to_be_visible()
        expect(page.locator("#start-weights")).to_have_value("yolo11n.pt")
        expect(page.locator("#start-weights option")).to_have_text(
            ["YOLO11n (general objects)", "MegaDetector v6 compact", "MegaDetector v6 extra"]
        )
        expect(page.locator("#base-model-info")).to_contain_text("starting weights above")
        page.select_option("#start-weights", "mdv6-yolov10-e")
        expect(page.locator("#start-weights-help")).to_contain_text("128 MB download")
        # FT-31: the extra model was trained at 1280 px, so the image size follows it.
        _open_settings(page)
        expect(page.locator("#imgsz-field")).to_be_visible()
        expect(page.locator("#imgsz")).to_have_value("1280")
        page.screenshot(path=str(shots / "ft31-ui-imgsz-extra.png"), full_page=True)
        page.select_option("#start-weights", "mdv6-yolov10-c")
        expect(page.locator("#start-weights-help")).to_contain_text("camera-trap photos")
        expect(page.locator("#imgsz")).to_have_value("640")
        page.screenshot(path=str(shots / "ft24-ui-start-weights-c.png"), full_page=True)
        with page.expect_request(
            lambda r: r.method == "POST" and r.url.endswith("/v1/jobs")
        ) as req:
            page.click("#start-training")
        assert req.value.post_data_json["params"] == {
            "model": "mdv6-yolov10-c",
            "batch": 16,
            "imgsz": 640,
        }
        job = req.value.response().json()

        # A Studio base model fixes the network: the select hides and the job trains the base's arch.
        page.evaluate("""() => {
            state.baseModel = {row: {id: 7, pipeline: "yolo_detector", model_arch: "mdv6-yolov10-e",
                                     labels: ["cat", "dog"], studio_import_id: 3},
                               name: "MD fine-tune", projectName: "P"};
            renderBaseModel();
        }""")
        expect(field).to_be_hidden()
        # Base model 7 exists only in the page; answer the job POST with the job created above.
        page.route(
            "**/v1/jobs",
            lambda route: route.fulfill(json=job)
            if route.request.method == "POST"
            else route.continue_(),
        )
        with page.expect_request(
            lambda r: r.method == "POST" and r.url.endswith("/v1/jobs")
        ) as req:
            page.evaluate("startTraining()")
        params = req.value.post_data_json["params"]
        assert params["base_model_id"] == 7 and params["model"] == "mdv6-yolov10-e"
    assert snap


# ------------------------------------------------------------------ 6. calibrated threshold


def test_classifier_threshold_is_explained(server, client, settings, shots):
    base, _ = server
    snap = build_snapshot(client)["snapshot"]
    _insert_model(
        settings,
        snap["id"],
        "conventional_classifier",
        metrics={"val_accuracy": 0.9},
        metadata={
            "calibration": {
                "calibrated_threshold": None,
                "reason": "too few validation crops (3)",
            }
        },
    )
    newest = _insert_model(
        settings,
        snap["id"],
        "conventional_classifier",
        metrics={
            "val_accuracy": 0.91,
            "calibrated_threshold": 0.62,
            "threshold_coverage": 0.8,
            "threshold_accuracy": 0.955,
        },
        metadata={
            "calibrated_threshold": 0.62,
            "abstain_label": "unknown",
            "calibration": {
                "calibrated_threshold": 0.62,
                "abstain_label": "unknown",
                "threshold_coverage": 0.8,
                "threshold_accuracy": 0.955,
                "target_accuracy": 0.95,
            },
        },
    )
    add_user(settings, "alice", PASSWORD)
    with _page(base, shots / "ft24-ui-threshold.png") as page:
        page.locator('.sidebar-nav [data-page="models"]').click()
        box = page.locator("#threshold-info")
        expect(box).to_be_visible()
        expect(box).to_contain_text("0.6200")
        expect(box).to_contain_text("80.0%")
        expect(box).to_contain_text("95.5%")
        expect(box).to_contain_text(
            'Predictions with confidence below 0.6200 are labelled "unknown" instead of a class.'
        )
        expect(page.locator("#per-class-section")).to_be_hidden()
        page.screenshot(path=str(shots / "ft24-ui-threshold-set.png"), full_page=True)
        page.locator(f'.model-list-item:not([data-model-id="{newest}"])').click()
        expect(box).to_contain_text("No confidence threshold was set")
        expect(box).to_contain_text("too few validation crops (3)")


def test_low_threshold_coverage_warns_and_quotes_lower_targets(server, client, settings, shots):
    # FT-30: a small-data classifier that labels most images "unknown" says so and what to change.
    base, _ = server
    snap = build_snapshot(client)["snapshot"]
    cal = {
        "calibrated_threshold": 0.9,
        "abstain_label": "unknown",
        "threshold_coverage": 0.3,
        "threshold_accuracy": 0.96,
        "target_accuracy": 0.95,
        "coverage_by_target": [
            {"target": 0.95, "threshold": 0.9, "coverage": 0.3},
            {"target": 0.9, "threshold": 0.7, "coverage": 0.55},
            {"target": 0.85, "threshold": 0.5, "coverage": 0.72},
            {"target": 0.8, "threshold": None, "coverage": 0.0},
        ],
    }
    _insert_model(
        settings,
        snap["id"],
        "conventional_classifier",
        metrics={"val_accuracy": 0.8, "calibrated_threshold": 0.9, "threshold_coverage": 0.3},
        metadata={"calibrated_threshold": 0.9, "abstain_label": "unknown", "calibration": cal},
    )
    add_user(settings, "alice", PASSWORD)
    with _page(base, shots / "ft30-ui-low-coverage.png") as page:
        page.locator('.sidebar-nav [data-page="models"]').click()
        warning = page.locator("#threshold-info .threshold-warning")
        expect(warning).to_be_visible()
        expect(warning).to_contain_text("Only 30.0% of validation images get a class")
        expect(warning).to_contain_text("90.0% target → 55.0% get a class")
        expect(warning).to_contain_text("85.0% target → 72.0% get a class")
        expect(warning).not_to_contain_text("80.0% target")
        page.screenshot(path=str(shots / "ft30-ui-low-coverage.png"), full_page=True)


def test_classifier_form_sends_the_target_accuracy(server, client, settings, shots):
    base, _ = server
    build_snapshot(client)
    add_user(settings, "alice", PASSWORD)
    with _page(base, shots / "ft30-ui-target-accuracy.png") as page:
        # The test server registers no classifier trainer; list it as available for the form.
        def pipelines(route):
            body = route.fetch().json()
            for row in body["pipelines"]:
                if row["id"] == "conventional_classifier":
                    row["status"] = "available"
            route.fulfill(json=body)

        page.route("**/v1/pipelines", pipelines)
        page.reload()
        page.select_option("#pipeline-select", "conventional_classifier")
        _open_settings(page)
        field = page.locator("#target-accuracy-field")
        expect(field).to_be_visible()
        # FT-36: the image classifier has a batch size too, defaulting to 32.
        expect(page.locator("#batch-field")).to_be_visible()
        expect(page.locator("#batch-size")).to_have_value("32")
        expect(page.locator("#batch-help")).to_contain_text("Default 32")
        page.select_option("#target-accuracy", "0.85")
        page.screenshot(path=str(shots / "ft30-ui-target-accuracy-form.png"), full_page=True)
        # Queue the same params on the registered test classifier instead of the real trainer.
        def jobs(route):
            if route.request.method != "POST":
                return route.fallback()
            body = route.request.post_data_json
            route.continue_(post_data=json.dumps({**body, "pipeline": "fake_cls"}))

        page.route("**/v1/jobs", jobs)
        with page.expect_request(
            lambda r: r.method == "POST" and r.url.endswith("/v1/jobs")
        ) as req:
            page.click("#start-training")
        assert req.value.post_data_json["params"]["threshold_target_accuracy"] == 0.85
        assert req.value.post_data_json["params"]["batch"] == 32
        page.select_option("#pipeline-select", "fake_cls")
        expect(field).to_be_hidden()


def test_training_settings_send_only_changed_values(server, client, settings, shots):
    base, _ = server
    build_snapshot(client)
    add_user(settings, "alice", PASSWORD)
    with _page(base, shots / "ui-training-settings.png") as page:
        def pipelines(route):
            body = route.fetch().json()
            for row in body["pipelines"]:
                if row["id"] in ("conventional_classifier", "dinov3_linear"):
                    row["status"] = "available"
            route.fulfill(json=body)

        page.route("**/v1/pipelines", pipelines)
        page.reload()
        sent = _capture_job_params(page)
        page.select_option("#pipeline-select", "conventional_classifier")
        expect(page.locator("#training-settings")).to_be_visible()
        expect(page.locator("#epochs")).to_be_hidden()  # collapsed by default
        _open_settings(page)
        expect(page.locator("#epochs")).to_have_value("30")
        expect(page.locator("#patience")).to_have_value("5")
        expect(page.locator("#val-fraction")).to_have_value("0.2")
        expect(page.locator("#augment")).to_be_checked()
        expect(page.locator("#learning-speed")).to_have_value("1")
        page.screenshot(path=str(shots / "ui-training-settings-classifier.png"), full_page=True)
        page.click("#start-training")
        expect(page.locator("#error-text")).to_contain_text("stopped by the test")
        assert sent()[-1] == {"batch": 32, "threshold_target_accuracy": 0.95}

        page.fill("#epochs", "0")
        page.click("#start-training")
        expect(page.locator("#error-text")).to_contain_text(
            "Training length must be a whole number from 1 to 1000."
        )
        page.fill("#epochs", "12")
        page.uncheck("#early-stop")
        expect(page.locator("#patience")).to_be_disabled()
        page.select_option("#val-fraction", "0.3")
        page.uncheck("#augment")
        page.select_option("#learning-speed", "0.5")
        n = len(sent())
        page.click("#start-training")
        expect(page.locator("#error-text")).to_contain_text("stopped by the test")
        assert len(sent()) == n + 1
        assert sent()[-1] == {
            "batch": 32, "threshold_target_accuracy": 0.95, "epochs": 12, "patience": 0,
            "val_fraction": 0.3, "augment": False, "lr": 0.0005,
        }

        # Another pipeline starts from its own defaults and shows only the settings it uses.
        page.select_option("#pipeline-select", "dinov3_linear")
        expect(page.locator("#epochs")).to_have_value("50")
        expect(page.locator("#val-fraction")).to_have_value("0.2")
        for hidden in ("#patience-field", "#augment-field", "#learning-speed-field", "#batch-field"):
            expect(page.locator(hidden)).to_be_hidden()


def test_training_settings_for_yolo_use_its_names_and_defaults(server, client, settings, shots):
    base, _ = server
    build_snapshot(client, task="detection", labels=("cat", "dog"))
    add_user(settings, "alice", PASSWORD)
    with _page(base, shots / "ui-training-settings-yolo.png") as page:
        sent = _capture_job_params(page)
        _open_settings(page)
        expect(page.locator("#epochs")).to_have_value("50")
        expect(page.locator("#patience")).to_have_value("10")
        expect(page.locator("#early-stop")).to_be_checked()
        expect(page.locator("#augment-field")).to_be_hidden()
        page.screenshot(path=str(shots / "ui-training-settings-yolo.png"), full_page=True)
        page.fill("#patience", "20")
        page.select_option("#learning-speed", "2")
        page.click("#start-training")
        expect(page.locator("#error-text")).to_contain_text("stopped by the test")
        assert sent()[-1] == {
            "model": "yolo11n.pt", "batch": 16, "imgsz": 640, "patience": 20, "lr0": 0.02,
        }


# ------------------------------------------------------------------ image viewer


def _viewer_caption(page):
    return page.locator("#image-viewer-caption")


def test_clicking_a_classification_photo_opens_a_large_viewer(
    server, client, settings, shots
):
    """2026-10-28: clicking a review-grid photo did nothing; it now opens a viewer
    with the full-size image, its label and status, and previous/next navigation."""
    base, _ = server
    ws = create_workspace(client)
    ds = create_dataset(client, ws["id"], task="classification")
    for i, name in enumerate(["a.png", "b.png", "c.png"]):
        upload_image(client, ds["id"], name, seed=i + 1)
    r = import_annotations(
        client, ds["id"], [{"image": "a.png", "annotations": [{"label": "deer"}]}]
    )
    assert r.status_code == 200, r.text
    add_user(settings, "alice", PASSWORD)
    with _page(base, shots / "image-viewer.png") as page:
        order = page.locator(".example-card").evaluate_all(
            "cards => cards.map(c => c.dataset.image)"
        )
        assert sorted(order) == ["a.png", "b.png", "c.png"]
        first = order[0]
        dialog = page.locator("#image-viewer")
        expect(dialog).to_be_hidden()

        page.locator(f'.example-card[data-image="{first}"] .photo-wrap img').click()
        expect(dialog).to_be_visible()
        expect(_viewer_caption(page)).to_contain_text(first)
        expect(_viewer_caption(page)).to_contain_text("1 of 3")
        page.wait_for_function(
            "() => { const i = document.querySelector('#image-viewer-img');"
            " return i.complete && i.naturalWidth > 0; }"
        )
        big = page.locator("#image-viewer-img").bounding_box()
        thumb = page.locator(
            f'.example-card[data-image="{first}"] .photo-wrap img'
        ).bounding_box()
        assert big["width"] > 1.5 * thumb["width"], (big, thumb)
        expect(page.locator("#image-viewer-prev")).to_be_disabled()

        page.keyboard.press("ArrowRight")
        expect(_viewer_caption(page)).to_contain_text(order[1])
        expect(_viewer_caption(page)).to_contain_text("2 of 3")
        page.click("#image-viewer-next")
        expect(_viewer_caption(page)).to_contain_text(order[2])
        expect(page.locator("#image-viewer-next")).to_be_disabled()
        page.keyboard.press("ArrowLeft")
        expect(_viewer_caption(page)).to_contain_text(order[1])
        page.screenshot(path=str(shots / "image-viewer-open.png"))

        page.keyboard.press("Escape")
        expect(dialog).to_be_hidden()

        # The labelled image shows its label; keyboard users open it with Enter.
        photo = page.locator('.example-card[data-image="a.png"] .photo-wrap')
        photo.focus()
        page.keyboard.press("Enter")
        expect(dialog).to_be_visible()
        expect(_viewer_caption(page)).to_contain_text("deer")
        page.click("#image-viewer-close")
        expect(dialog).to_be_hidden()


def test_detection_photo_viewer_draws_the_boxes(server, client, settings, shots):
    base, _ = server
    _detection_dataset(client)
    add_user(settings, "alice", PASSWORD)
    with _page(base, shots / "image-viewer-detection.png") as page:
        page.locator('.example-card[data-image="a.png"] .photo-wrap img').click()
        expect(page.locator("#image-viewer")).to_be_visible()
        boxes = page.locator("#image-viewer-boxes .box")
        expect(boxes).to_have_count(2)
        expect(boxes.first).to_have_text("cat")


def test_assist_suggests_a_lower_confidence_when_nothing_scores_high_enough(
    server, client, settings, shots
):
    """FT-32: a small-data detector that draws nothing tells the user the best score it saw."""
    base, _ = server
    ds = _detection_dataset(client)
    snap = client.post(f"/v1/datasets/{ds['id']}/snapshots").json()
    _insert_model(settings, snap["id"], "yolo_detector")
    add_user(settings, "alice", PASSWORD)
    bodies = []

    def assist(route):
        body = route.request.post_data_json
        bodies.append(body)
        n = len(body["images"])
        drawn = body["conf"] < 0.25
        route.fulfill(
            json={
                "model_id": body["model_id"],
                "conf": body["conf"],
                "processed": n,
                "items_written": n if drawn else 0,
                "boxes": n if drawn else 0,
                "unknown_labels": [],
                "best_below_conf": None if drawn else 0.08,
                "skipped": [] if drawn else [{"image": i, "reason": "no detections"} for i in body["images"]],
                "items": [],
            }
        )

    with _page(base, shots / "ft32-ui-assist-low-conf-hint.png") as page:
        page.route("**/v1/datasets/*/assist", assist)
        page.route("**/v1/datasets/*/assist/summary", _summary({"images": 3, "empty": 3}))
        page.reload()
        _enable_assist(page)
        expect(page.locator("#assist-conf")).to_have_value("0.25")
        _confirm_assist(page)
        status = page.locator("#assist-status")
        expect(status).to_contain_text("the best box scored 0.08")
        expect(status).to_contain_text('Lower "Minimum confidence"')
        page.fill("#assist-conf", "0.05")
        _confirm_assist(page)
        expect(status).to_contain_text("for review")
        expect(status).not_to_contain_text("best box scored")
    assert [b["conf"] for b in bodies][:1] == [0.25] and bodies[-1]["conf"] == 0.05


def test_classification_assist_end_to_end_boxes_inherit_the_image_label(
    server, client, settings, shots, monkeypatch
):
    """FT-35: on a classification dataset MegaDetector's animal boxes land on the review cards,
    labelled with the image's label; people are dropped."""
    from sparrow_finetune import assist

    class MD:
        labels = ["animal", "person", "vehicle"]

        def predict(self, rgb, conf, stats=None):
            return [
                assist.Detection("animal", [4.0, 4.0, 20.0, 20.0], 0.9),
                assist.Detection("person", [1.0, 1.0, 5.0, 5.0], 0.8),
            ]

    monkeypatch.setattr(assist, "megadetector_labels", lambda d: list(MD.labels))
    monkeypatch.setattr(assist, "load_megadetector", lambda d: MD())
    base, _ = server
    ws = create_workspace(client)
    ds = create_dataset(client, ws["id"], task="classification")
    for i, name in enumerate(["a.png", "b.png"]):
        upload_image(client, ds["id"], name, seed=i + 1)
    r = import_annotations(client, ds["id"], [{"image": "a.png", "annotations": [{"label": "deer"}]}])
    assert r.status_code == 200, r.text
    add_user(settings, "alice", PASSWORD)
    with _page(base, shots / "ft35-ui-classification-assist.png") as page:
        _enable_assist(page)
        expect(page.locator("#assist-model")).to_have_value("megadetector")
        expect(page.locator("#assist-animal-label")).to_have_attribute("placeholder", "the image's label")
        _confirm_assist(page)
        expect(page.locator("#assist-status")).to_contain_text("Added 2 boxes to 2 images")
        card_a = page.locator('.example-card[data-image="a.png"]')
        expect(card_a.locator(".box")).to_have_count(1)
        expect(card_a.locator(".box")).to_have_text("deer")
        card_b = page.locator('.example-card[data-image="b.png"]')
        expect(card_b.locator(".box")).to_have_text("animal")
        # The reviewer corrects b's box to a species through the ordinary label field.
        card_b.locator("[data-label-input]").fill("fox")
        card_b.locator('[data-example-action="correct"]').click()
        expect(page.locator('.example-card[data-image="b.png"] .box')).to_have_text("fox")
    items = _items(client, ds["id"])
    assert [(a["label"], a["bbox"]) for a in items["b.png"]["annotations"]] == [("fox", [4.0, 4.0, 20.0, 20.0])]
    assert items["b.png"]["review_status"] == "confirmed"
