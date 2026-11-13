"""AI-assisted annotation: intake, review, and export.

Three things are worth protecting here, in order.

Export writes outside the workbench — the only code path that does — so its
allowlist gets the same treatment as the artifact route and the folder picker.

Reviewing must be durable: a prediction pass that overwrote an afternoon of
corrections would be worse than no assistance at all.

And provenance must be honest. If the record cannot distinguish a box a model
guessed from one a person drew, it is not a record, it is a rumour.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

flask = pytest.importorskip("flask", reason="Flask not installed")

from sparrow_web import annotation
from sparrow_web.annotation import Batches
from sparrow_web.app import create_app
from sparrow_web.db import utcnow
from sparrow_web.repo import Projects
from sparrow_web.settings import Settings
from sparrow_yolo.xml_utils import read_voc_boxes


def jpeg(path: Path, size=(64, 48)) -> Path:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (30, 90, 120)).save(path, "JPEG")
    return path


@pytest.fixture
def app(tmp_path):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    exports = tmp_path / "exports"
    exports.mkdir()
    settings = Settings(
        state_dir=tmp_path / "state",
        prepared_root=tmp_path / "prepared",
        runs_root=tmp_path / "runs",
        annotations_root=tmp_path / "annotations",
        dataset_search_roots=[datasets],
        export_roots=[exports],
    )
    application = create_app(settings, start_supervisor=False)
    application.config["TESTING"] = True
    application.config.update(DATASETS=datasets, EXPORTS=exports)
    return application


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def project(app):
    root = app.config["DATASETS"] / "Sub_Saharan"
    root.mkdir(parents=True, exist_ok=True)
    projects = Projects(app.config["DB"])
    return projects.get(projects.create("Sub Saharan", str(root)))


@pytest.fixture
def source(app):
    """A folder of untagged images, as the field would deliver them."""
    folder = app.config["DATASETS"] / "incoming"
    for i in range(4):
        jpeg(folder / f"img{i}.jpg")
    (folder / "notes.txt").write_text("not an image")
    return folder


def make_batch(app, client, project, source, name="May 2026"):
    client.post(f"/p/{project['slug']}/annotate",
                data={"name": name, "source_path": str(source)})
    return app.config["DB"].one(
        "SELECT * FROM annotation_batches ORDER BY id DESC LIMIT 1")


def add_model(app, project, classes='{"oryx": 0}'):
    db = app.config["DB"]
    with db.write() as c:
        c.execute("INSERT INTO runs(project_id,name,run_dir,state,queued_at,updated_at)"
                  " VALUES (?,'r','/runs/r','completed',?,?)",
                  (project["id"], utcnow(), utcnow()))
        run_id = int(c.execute("SELECT last_insert_rowid() i").fetchone()[0])
        cur = c.execute(
            "INSERT INTO model_versions(project_id,run_id,version,weights_path,"
            "class_map_json,n_classes,created_at) VALUES (?,?,1,'/w/best.pt',?,1,?)",
            (project["id"], run_id, classes, utcnow()))
        return int(cur.lastrowid)


def predict_into(app, batch, *, boxes_per_image=1, label="oryx", model_id=None):
    """Stand in for a prediction pass without loading a model.

    The worker's handler needs a GPU and Ultralytics; what the rest of the system
    has to get right is what happens to the XML and the aggregates afterwards, so
    the tests drive that directly.
    """
    from sparrow_yolo.xml_utils import write_voc_xml

    batches = Batches(app.config["DB"])
    batch_dir = Path(batch["dir_path"])
    for row in batches.images(batch["id"], "all"):
        rel = row["rel_path"]
        boxes = [{"label": label, "xmin": 5, "ymin": 5, "xmax": 40, "ymax": 30,
                  "score": 0.4 + 0.1 * i, "source": "model"}
                 for i in range(boxes_per_image)]
        write_voc_xml(annotation.xml_for(batch_dir, rel), batch_dir / "images" / rel,
                      64, 48, boxes, predicted_by="v1")
        batches.record_boxes(batch["id"], rel, boxes, state="predicted",
                             model_version_id=model_id, width=64, height=48)


class TestIntake:
    def test_creates_a_batch_and_copies_the_images(self, app, client, project, source):
        batch = make_batch(app, client, project, source)
        assert batch["n_images"] == 4
        images = sorted(p.name for p in (Path(batch["dir_path"]) / "images").iterdir())
        assert images == ["img0.jpg", "img1.jpg", "img2.jpg", "img3.jpg"]

    def test_non_images_are_left_behind(self, app, client, project, source):
        batch = make_batch(app, client, project, source)
        assert not (Path(batch["dir_path"]) / "images" / "notes.txt").exists()

    def test_the_source_folder_is_not_touched(self, app, client, project, source):
        before = sorted(p.name for p in source.iterdir())
        make_batch(app, client, project, source)
        assert sorted(p.name for p in source.iterdir()) == before

    def test_the_source_path_is_captured_not_referenced(self, app, client, project, source):
        """The batch has to survive the original folder moving or going away."""
        batch = make_batch(app, client, project, source)
        import shutil
        shutil.rmtree(source)
        assert batch["source_path"] == str(source)
        assert len(list((Path(batch["dir_path"]) / "images").iterdir())) == 4

    def test_a_folder_outside_the_allowlist_is_refused(self, app, client, project):
        r = client.post(f"/p/{project['slug']}/annotate",
                        data={"name": "bad", "source_path": "/etc"})
        assert r.status_code == 400
        assert app.config["DB"].one("SELECT COUNT(*) c FROM annotation_batches")["c"] == 0

    def test_an_empty_folder_leaves_nothing_behind(self, app, client, project):
        empty = app.config["DATASETS"] / "empty"
        empty.mkdir()
        r = client.post(f"/p/{project['slug']}/annotate",
                        data={"name": "e", "source_path": str(empty)})
        assert r.status_code == 400
        assert b"No images" in r.data
        # No half-created batch, and no orphaned directory.
        assert app.config["DB"].one("SELECT COUNT(*) c FROM annotation_batches")["c"] == 0
        assert not list((app.config["SETTINGS"].annotations_root).rglob("images"))

    def test_names_colliding_across_subfolders_all_survive(self, app, client, project):
        """Flattening is what the review UI and VOC both want; losing an image to
        it would be invisible until someone counted."""
        folder = app.config["DATASETS"] / "nested"
        jpeg(folder / "a" / "IMG_0001.jpg")
        jpeg(folder / "b" / "IMG_0001.jpg")
        batch = make_batch(app, client, project, folder, name="nested")
        assert batch["n_images"] == 2
        assert len(list((Path(batch["dir_path"]) / "images").iterdir())) == 2


class TestReview:
    def test_the_editor_reads_boxes_and_size(self, app, client, project, source):
        batch = make_batch(app, client, project, source)
        predict_into(app, batch)
        r = client.get(f"/api/batches/{batch['id']}/images/img0.jpg")
        body = json.loads(r.data)
        assert body["width"] == 64 and body["height"] == 48
        assert body["boxes"][0]["label"] == "oryx"
        assert body["boxes"][0]["source"] == "model"

    def test_saving_marks_it_reviewed_and_counts_provenance(self, app, client, project, source):
        batch = make_batch(app, client, project, source)
        predict_into(app, batch)
        r = client.post(f"/api/batches/{batch['id']}/images/img0.jpg", json={
            "width": 64, "height": 48,
            "boxes": [
                {"label": "oryx", "xmin": 5, "ymin": 5, "xmax": 40, "ymax": 30,
                 "score": 0.9, "source": "model"},
                {"label": "zebra", "xmin": 10, "ymin": 10, "xmax": 50, "ymax": 40,
                 "source": "human"},
            ]})
        assert r.status_code == 200

        row = Batches(app.config["DB"]).image(batch["id"], "img0.jpg")
        assert row["state"] == "reviewed"
        assert row["n_boxes"] == 2
        assert row["n_ai_boxes"] == 1 and row["n_human_boxes"] == 1
        assert row["reviewed_at"]

    def test_saved_provenance_survives_into_the_xml(self, app, client, project, source):
        batch = make_batch(app, client, project, source)
        predict_into(app, batch)
        client.post(f"/api/batches/{batch['id']}/images/img0.jpg", json={
            "width": 64, "height": 48,
            "boxes": [{"label": "zebra", "xmin": 1, "ymin": 1, "xmax": 40,
                       "ymax": 30, "source": "human"}]})
        boxes = read_voc_boxes(annotation.xml_for(Path(batch["dir_path"]), "img0.jpg"))
        assert boxes[0]["source"] == "human"

    def test_an_empty_save_records_a_confirmed_blank(self, app, client, project, source):
        """Deleting every box is a real answer — this frame has no animals — and
        must produce an object-free XML rather than no file."""
        batch = make_batch(app, client, project, source)
        predict_into(app, batch)
        client.post(f"/api/batches/{batch['id']}/images/img0.jpg",
                    json={"width": 64, "height": 48, "boxes": []})
        xml = annotation.xml_for(Path(batch["dir_path"]), "img0.jpg")
        assert xml.exists() and read_voc_boxes(xml) == []
        assert Batches(app.config["DB"]).image(batch["id"], "img0.jpg")["state"] == "reviewed"

    def test_an_unknown_image_is_404_not_a_new_row(self, app, client, project, source):
        batch = make_batch(app, client, project, source)
        assert client.get(f"/api/batches/{batch['id']}/images/nope.jpg").status_code == 404


class TestImageServing:
    def test_serves_an_image_from_the_batch(self, app, client, project, source):
        batch = make_batch(app, client, project, source)
        r = client.get(f"/api/batches/{batch['id']}/file/img0.jpg")
        assert r.status_code == 200 and r.data[:2] == b"\xff\xd8"

    @pytest.mark.parametrize("attack", [
        "../../../../etc/passwd",
        "..%2f..%2f..%2fsecret.txt",
        "sub/../../../secret.txt",
    ])
    def test_path_traversal_is_refused(self, app, client, project, source, attack):
        """Runs as root with no auth; a path built from a URL is only served
        after proving it stayed inside the batch."""
        batch = make_batch(app, client, project, source)
        (Path(batch["dir_path"]).parent / "secret.txt").write_text("do not serve me")
        r = client.get(f"/api/batches/{batch['id']}/file/{attack}")
        assert r.status_code == 404
        assert b"do not serve me" not in r.data


class TestFilters:
    def test_each_filter_selects_what_it_says(self, app, client, project, source):
        batch = make_batch(app, client, project, source)
        batches = Batches(app.config["DB"])
        predict_into(app, batch)

        # One reviewed, one blank prediction, the rest as predicted.
        client.post(f"/api/batches/{batch['id']}/images/img0.jpg",
                    json={"width": 64, "height": 48,
                          "boxes": [{"label": "oryx", "xmin": 1, "ymin": 1,
                                     "xmax": 30, "ymax": 20, "source": "human"}]})
        batches.record_boxes(batch["id"], "img1.jpg", [], state="predicted")

        def names(f):
            return sorted(r["rel_path"] for r in batches.images(batch["id"], f))

        assert names("reviewed") == ["img0.jpg"]
        assert "img0.jpg" not in names("unreviewed")
        assert names("blank") == ["img1.jpg"]
        assert len(names("all")) == 4
        # Predictions here score 0.4, below the 0.5 default.
        assert "img2.jpg" in names("low")


class TestPredictionDoesNotDiscardReview:
    def test_a_reviewed_image_keeps_its_state_and_counts(self, app, client, project, source):
        """Re-running at a lower threshold is normal. Losing an afternoon of
        corrections to it would not be."""
        from sparrow_web.supervisor import Supervisor

        batch = make_batch(app, client, project, source)
        predict_into(app, batch)
        client.post(f"/api/batches/{batch['id']}/images/img0.jpg",
                    json={"width": 64, "height": 48,
                          "boxes": [{"label": "zebra", "xmin": 2, "ymin": 2,
                                     "xmax": 30, "ymax": 20, "source": "human"}]})

        db = app.config["DB"]
        with db.write() as c:
            c.execute("INSERT INTO jobs(project_id,kind,state,work_dir,params_json,"
                      "created_at,updated_at) VALUES (?,'predict','running','',?,?,?)",
                      (project["id"], json.dumps({"batch_id": batch["id"], "conf": 0.1}),
                       utcnow(), utcnow()))
        job = db.one("SELECT * FROM jobs WHERE kind='predict'")

        supervisor = Supervisor(db, app.config["SETTINGS"], app.config["BUS"])
        supervisor._ingest_prediction(job, {
            "n_images": 4, "n_boxes": 8, "n_blank": 0, "n_errors": 0,
            "images": [{"rel_path": f"img{i}.jpg", "n_boxes": 2, "width": 64,
                        "height": 48, "min_confidence": 0.2} for i in range(4)],
        })

        reviewed = Batches(db).image(batch["id"], "img0.jpg")
        assert reviewed["state"] == "reviewed"
        assert reviewed["n_human_boxes"] == 1, "the human's work was overwritten"

        other = Batches(db).image(batch["id"], "img1.jpg")
        assert other["state"] == "predicted" and other["n_boxes"] == 2

    def test_the_pass_is_recorded_even_though_nothing_changed(self, app, client,
                                                              project, source):
        from sparrow_web.supervisor import Supervisor

        batch = make_batch(app, client, project, source)
        db = app.config["DB"]
        with db.write() as c:
            c.execute("INSERT INTO jobs(project_id,kind,state,work_dir,params_json,"
                      "created_at,updated_at) VALUES (?,'predict','running','',?,?,?)",
                      (project["id"],
                       json.dumps({"batch_id": batch["id"], "conf": 0.35, "iou": 0.45,
                                   "imgsz": 1024, "force_class": "oryx"}),
                       utcnow(), utcnow()))
        job = db.one("SELECT * FROM jobs WHERE kind='predict'")
        Supervisor(db, app.config["SETTINGS"], app.config["BUS"])._ingest_prediction(
            job, {"n_images": 4, "n_boxes": 0, "n_blank": 4, "n_errors": 0, "images": []})

        passes = Batches(db).passes(batch["id"])
        assert len(passes) == 1
        assert passes[0]["force_class"] == "oryx" and passes[0]["conf"] == 0.35


class TestExport:
    def _reviewed_batch(self, app, client, project, source):
        batch = make_batch(app, client, project, source)
        predict_into(app, batch)
        for name in ("img0.jpg", "img1.jpg"):
            client.post(f"/api/batches/{batch['id']}/images/{name}",
                        json={"width": 64, "height": 48,
                              "boxes": [{"label": "oryx", "xmin": 2, "ymin": 2,
                                         "xmax": 30, "ymax": 20, "source": "human"}]})
        return batch

    def test_writes_images_annotations_and_a_manifest(self, app, client, project, source):
        batch = self._reviewed_batch(app, client, project, source)
        dest = app.config["EXPORTS"] / "out"
        r = client.post(f"/p/{project['slug']}/annotate/{batch['id']}/publish",
                        data={"dest_path": str(dest), "reviewed_only": "on",
                              "include_images": "on"})
        assert r.status_code == 302
        assert sorted(p.name for p in (dest / "images").iterdir()) == \
            ["img0.jpg", "img1.jpg"]
        assert sorted(p.name for p in (dest / "annotations").iterdir()) == \
            ["img0.xml", "img1.xml"]
        manifest = json.loads((dest / "manifest.json").read_text())
        assert manifest["n_images"] == 2 and manifest["reviewed_only"] is True

    def test_reviewed_only_leaves_unchecked_guesses_behind(self, app, client,
                                                           project, source):
        batch = self._reviewed_batch(app, client, project, source)
        dest = app.config["EXPORTS"] / "out"
        client.post(f"/p/{project['slug']}/annotate/{batch['id']}/publish",
                    data={"dest_path": str(dest), "reviewed_only": "on",
                          "include_images": "on"})
        assert not (dest / "images" / "img2.jpg").exists()

    def test_including_everything_takes_the_guesses_too(self, app, client, project, source):
        batch = self._reviewed_batch(app, client, project, source)
        dest = app.config["EXPORTS"] / "all"
        client.post(f"/p/{project['slug']}/annotate/{batch['id']}/publish",
                    data={"dest_path": str(dest), "include_images": "on"})
        assert len(list((dest / "images").iterdir())) == 4

    def test_annotations_only(self, app, client, project, source):
        batch = self._reviewed_batch(app, client, project, source)
        dest = app.config["EXPORTS"] / "xmlonly"
        client.post(f"/p/{project['slug']}/annotate/{batch['id']}/publish",
                    data={"dest_path": str(dest), "reviewed_only": "on"})
        assert not (dest / "images").exists()
        assert len(list((dest / "annotations").iterdir())) == 2

    def test_the_exported_xml_parses_with_the_pipeline_reader(self, app, client,
                                                              project, source):
        """The whole point is that this becomes training data."""
        from sparrow_yolo.xml_utils import parse_voc_xml

        batch = self._reviewed_batch(app, client, project, source)
        dest = app.config["EXPORTS"] / "out"
        client.post(f"/p/{project['slug']}/annotate/{batch['id']}/publish",
                    data={"dest_path": str(dest), "reviewed_only": "on",
                          "include_images": "on"})
        boxes = parse_voc_xml(dest / "annotations" / "img0.xml")
        assert len(boxes) == 1 and boxes[0].class_name == "oryx"

    @pytest.mark.parametrize("attack", ["/etc", "/", "/root"])
    def test_a_destination_outside_the_allowlist_is_refused(self, app, client,
                                                            project, source, attack):
        batch = self._reviewed_batch(app, client, project, source)
        r = client.post(f"/p/{project['slug']}/annotate/{batch['id']}/publish",
                        data={"dest_path": attack, "reviewed_only": "on"})
        assert r.status_code == 403
        assert app.config["DB"].one(
            "SELECT COUNT(*) c FROM annotation_publishes")["c"] == 0

    def test_traversal_out_of_the_allowlist_is_refused(self, app, client, project, source):
        batch = self._reviewed_batch(app, client, project, source)
        dest = str(app.config["EXPORTS"] / ".." / ".." / "escaped")
        r = client.post(f"/p/{project['slug']}/annotate/{batch['id']}/publish",
                        data={"dest_path": dest, "reviewed_only": "on"})
        assert r.status_code == 403

    def test_exporting_nothing_says_so_rather_than_writing_an_empty_folder(
            self, app, client, project, source):
        batch = make_batch(app, client, project, source)   # nothing reviewed
        r = client.post(f"/p/{project['slug']}/annotate/{batch['id']}/publish",
                        data={"dest_path": str(app.config["EXPORTS"] / "out"),
                              "reviewed_only": "on"})
        assert r.status_code == 400 and b"Nothing to export" in r.data

    def test_every_export_is_recorded_and_none_replaces_another(self, app, client,
                                                                project, source):
        batch = self._reviewed_batch(app, client, project, source)
        for name in ("first", "second"):
            client.post(f"/p/{project['slug']}/annotate/{batch['id']}/publish",
                        data={"dest_path": str(app.config["EXPORTS"] / name),
                              "reviewed_only": "on", "include_images": "on"})
        rows = Batches(app.config["DB"]).publishes(batch["id"])
        assert len(rows) == 2
        assert {Path(r["dest_path"]).name for r in rows} == {"first", "second"}

    def test_the_manifest_records_which_model_produced_the_guesses(self, app, client,
                                                                   project, source):
        batch = self._reviewed_batch(app, client, project, source)
        Batches(app.config["DB"]).record_pass(
            batch["id"], {"conf": 0.3, "iou": 0.45, "imgsz": 1024},
            {"n_images": 4, "n_boxes": 4})
        dest = app.config["EXPORTS"] / "out"
        client.post(f"/p/{project['slug']}/annotate/{batch['id']}/publish",
                    data={"dest_path": str(dest), "reviewed_only": "on"})
        manifest = json.loads((dest / "manifest.json").read_text())
        assert manifest["passes"] and manifest["passes"][0]["conf"] == 0.3


class TestWritableBrowsing:
    def test_the_writable_picker_lists_the_export_roots(self, app, client):
        (app.config["EXPORTS"] / "somewhere").mkdir()
        data = json.loads(client.get("/api/browse?writable=1").data)
        assert "somewhere" in [e["name"] for e in data["entries"]]

    def test_the_two_allowlists_do_not_leak_into_each_other(self, app, client):
        """A dataset root must not be offered as a place to write, and an export
        root must not be offered as a dataset."""
        datasets, exports = app.config["DATASETS"], app.config["EXPORTS"]
        assert client.get(f"/api/browse?writable=1&path={datasets}").status_code == 403
        assert client.get(f"/api/browse?path={exports}").status_code == 403

    def test_writable_browsing_says_so_when_nothing_is_mounted(self, tmp_path):
        settings = Settings(state_dir=tmp_path / "s", dataset_search_roots=[tmp_path],
                            export_roots=[])
        app = create_app(settings, start_supervisor=False)
        app.config["TESTING"] = True
        r = app.test_client().get("/api/browse?writable=1")
        assert r.status_code == 409 and b"writable" in r.data


class TestPages:
    def test_the_project_page_links_to_annotation(self, app, client, project):
        r = client.get(f"/p/{project['slug']}")
        assert f'href="/p/{project["slug"]}/annotate"'.encode() in r.data

    def test_the_index_renders_without_batches(self, app, client, project):
        r = client.get(f"/p/{project['slug']}/annotate")
        assert r.status_code == 200 and b"New batch" in r.data

    def test_no_models_is_explained_rather_than_left_blank(self, app, client, project):
        r = client.get(f"/p/{project['slug']}/annotate")
        assert b"No trained models yet" in r.data

    def test_the_batch_page_renders_the_editor(self, app, client, project, source):
        batch = make_batch(app, client, project, source)
        r = client.get(f"/p/{project['slug']}/annotate/{batch['id']}")
        assert r.status_code == 200
        assert b'id="canvas"' in r.data and b"annotate.js" in r.data

    def test_a_batch_from_another_project_is_404(self, app, client, project, source):
        batch = make_batch(app, client, project, source)
        projects = Projects(app.config["DB"])
        other = projects.get(projects.create("Other", str(app.config["DATASETS"])))
        assert client.get(
            f"/p/{other['slug']}/annotate/{batch['id']}").status_code == 404


class TestExportScope:
    def test_untouched_images_are_never_exported(self, app, client, project, source):
        """Unticking "reviewed only" means "include the AI's guesses", not
        "include images no model has looked at". Those carry no annotation, and
        exporting them would quietly add untagged files to someone's dataset."""
        batch = make_batch(app, client, project, source)
        batches = Batches(app.config["DB"])

        # Two predicted, two never run.
        from sparrow_yolo.xml_utils import write_voc_xml
        batch_dir = Path(batch["dir_path"])
        for name in ("img0.jpg", "img1.jpg"):
            boxes = [{"label": "oryx", "xmin": 5, "ymin": 5, "xmax": 40,
                      "ymax": 30, "score": 0.8, "source": "model"}]
            write_voc_xml(annotation.xml_for(batch_dir, name),
                          batch_dir / "images" / name, 64, 48, boxes)
            batches.record_boxes(batch["id"], name, boxes, state="predicted",
                                 width=64, height=48)

        dest = app.config["EXPORTS"] / "out"
        client.post(f"/p/{project['slug']}/annotate/{batch['id']}/publish",
                    data={"dest_path": str(dest), "include_images": "on"})
        assert sorted(p.name for p in (dest / "images").iterdir()) == \
            ["img0.jpg", "img1.jpg"]
        assert len(list((dest / "annotations").iterdir())) == 2


class TestProvenanceNaming:
    def test_the_pass_records_which_project_and_version_predicted(self, app, client,
                                                                  project, source):
        """`predicted_by` ends up inside every exported XML, so 'v1' alone would
        be ambiguous once a second project exists."""
        batch = make_batch(app, client, project, source)
        db = app.config["DB"]
        with db.write() as c:
            c.execute("INSERT INTO runs(project_id,name,run_dir,state,queued_at,"
                      "updated_at) VALUES (?,'r','/runs/r','completed',?,?)",
                      (project["id"], utcnow(), utcnow()))
            run_id = int(c.execute("SELECT last_insert_rowid() i").fetchone()[0])
            c.execute("INSERT INTO model_versions(project_id,run_id,version,"
                      "weights_path,class_map_json,n_classes,created_at)"
                      " VALUES (?,?,1,'/w/best.pt','{\"oryx\": 0}',1,?)",
                      (project["id"], run_id, utcnow()))
            model_id = int(c.execute("SELECT last_insert_rowid() i").fetchone()[0])

        client.post(f"/p/{project['slug']}/annotate/{batch['id']}/run",
                    data={"model_version_id": str(model_id), "conf": "0.3"})
        job = db.one("SELECT * FROM jobs WHERE kind='predict'")
        params = json.loads(job["params_json"])
        assert params["predicted_by"] == f"{project['slug']} v1"
        assert params["device"] == str(app.config["SETTINGS"].infer_gpu)
        # The inference GPU, so training on the other card is never blocked.
        assert f"gpu:{app.config['SETTINGS'].infer_gpu}" in \
            json.loads(job["needs_resources_json"])


class TestNewSpecies:
    """Tagging an animal the model has never seen is the normal case for a new
    species, not an edge case — so no species field may be a closed list."""

    def test_a_label_the_model_does_not_know_is_saved(self, app, client, project, source):
        batch = make_batch(app, client, project, source)
        predict_into(app, batch)
        r = client.post(f"/api/batches/{batch['id']}/images/img0.jpg", json={
            "width": 64, "height": 48,
            "boxes": [{"label": "pangolin", "xmin": 5, "ymin": 5, "xmax": 40,
                       "ymax": 30, "source": "human"}]})
        assert r.status_code == 200
        boxes = read_voc_boxes(annotation.xml_for(Path(batch["dir_path"]), "img0.jpg"))
        assert boxes[0]["label"] == "pangolin"

    def test_a_new_species_is_offered_next_time(self, app, client, project, source):
        """Typed once, it has to be suggested on the following image and after a
        reload — otherwise you retype it for every frame in the batch."""
        batch = make_batch(app, client, project, source)
        client.post(f"/api/batches/{batch['id']}/images/img0.jpg", json={
            "width": 64, "height": 48,
            "boxes": [{"label": "pangolin", "xmin": 5, "ymin": 5, "xmax": 40,
                       "ymax": 30, "source": "human"}]})
        assert annotation.known_labels(Path(batch["dir_path"])) == ["pangolin"]
        r = client.get(f"/p/{project['slug']}/annotate/{batch['id']}")
        assert b'value="pangolin"' in r.data

    def test_the_species_fields_are_typeable_not_dropdowns(self, app, client,
                                                           project, source):
        add_model(app, project)          # the run form only renders with one
        batch = make_batch(app, client, project, source)
        r = client.get(f"/p/{project['slug']}/annotate/{batch['id']}")
        assert b'<datalist id="species">' in r.data
        assert b'id="force_class"' in r.data and b'list="species"' in r.data
        assert b'id="box-label"' in r.data
        # A <select> here would make an unknown species unreachable.
        assert b'<select id="force_class"' not in r.data
        assert b'<select id="box-label"' not in r.data

    def test_forcing_an_unknown_species_reaches_the_job(self, app, client,
                                                        project, source):
        batch = make_batch(app, client, project, source)
        model_id = add_model(app, project)
        client.post(f"/p/{project['slug']}/annotate/{batch['id']}/run",
                    data={"model_version_id": str(model_id), "force_class": "pangolin"})
        job = app.config["DB"].one("SELECT * FROM jobs WHERE kind='predict'")
        assert json.loads(job["params_json"])["force_class"] == "pangolin"

    def test_suggestions_include_the_project_dataset_species(self, app, client,
                                                             project, source):
        """The model is one source of names, not the only one: a species already
        in the dataset should not have to be retyped."""
        batch = make_batch(app, client, project, source)
        db = app.config["DB"]
        with db.write() as c:
            cur = c.execute(
                "INSERT INTO dataset_snapshots(project_id,dataset_root,audit_params_json,"
                "content_hash,quick_hash,artifact_dir,summary_json,last_verified_at,"
                "created_at) VALUES (?,?,'{}','h','q','','{}',?,?)",
                (project["id"], project["dataset_root"], utcnow(), utcnow()))
            snapshot_id = int(cur.lastrowid)
            c.execute("INSERT INTO snapshot_class_counts(snapshot_id,class_name,"
                      "n_boxes,n_images) VALUES (?,'aardvark',10,5)", (snapshot_id,))
        r = client.get(f"/p/{project['slug']}/annotate/{batch['id']}")
        assert b'value="aardvark"' in r.data

    def test_a_blank_species_is_never_written(self, app, client, project, source):
        """write_voc_xml drops unnamed boxes; this pins that the editor's save
        path does not smuggle one through."""
        batch = make_batch(app, client, project, source)
        client.post(f"/api/batches/{batch['id']}/images/img0.jpg", json={
            "width": 64, "height": 48,
            "boxes": [{"label": "   ", "xmin": 5, "ymin": 5, "xmax": 40, "ymax": 30,
                       "source": "human"}]})
        assert read_voc_boxes(annotation.xml_for(Path(batch["dir_path"]), "img0.jpg")) == []
        assert annotation.known_labels(Path(batch["dir_path"])) == []


class TestDeleteImage:
    """Removing a frame from a batch — out of focus, fired by a branch, or just
    not wanted. The batch's copy goes; the folder it came from does not."""

    def test_removes_the_image_and_its_annotation(self, app, client, project, source):
        batch = make_batch(app, client, project, source)
        predict_into(app, batch)
        batch_dir = Path(batch["dir_path"])
        assert annotation.xml_for(batch_dir, "img0.jpg").exists()

        r = client.delete(f"/api/batches/{batch['id']}/images/img0.jpg")
        assert r.status_code == 200
        body = json.loads(r.data)
        assert body["removed"] == {"image": True, "annotation": True}
        assert not (batch_dir / "images" / "img0.jpg").exists()
        assert not annotation.xml_for(batch_dir, "img0.jpg").exists()

    def test_an_image_with_no_annotation_deletes_cleanly(self, app, client,
                                                         project, source):
        batch = make_batch(app, client, project, source)      # nothing predicted
        r = client.delete(f"/api/batches/{batch['id']}/images/img0.jpg")
        assert r.status_code == 200
        assert json.loads(r.data)["removed"] == {"image": True, "annotation": False}

    def test_the_source_folder_keeps_its_copy(self, app, client, project, source):
        """The whole reason images are copied into a batch."""
        batch = make_batch(app, client, project, source)
        client.delete(f"/api/batches/{batch['id']}/images/img0.jpg")
        assert (source / "img0.jpg").exists()

    def test_the_row_and_the_count_go_too(self, app, client, project, source):
        batch = make_batch(app, client, project, source)
        client.delete(f"/api/batches/{batch['id']}/images/img0.jpg")
        batches = Batches(app.config["DB"])
        assert batches.image(batch["id"], "img0.jpg") is None
        assert batches.get(batch["id"])["n_images"] == 3
        assert batches.progress(batch["id"])["total"] == 3

    def test_a_deleted_image_is_not_exported(self, app, client, project, source):
        batch = make_batch(app, client, project, source)
        predict_into(app, batch)
        for name in ("img0.jpg", "img1.jpg"):
            client.post(f"/api/batches/{batch['id']}/images/{name}",
                        json={"width": 64, "height": 48,
                              "boxes": [{"label": "oryx", "xmin": 2, "ymin": 2,
                                         "xmax": 30, "ymax": 20, "source": "human"}]})
        client.delete(f"/api/batches/{batch['id']}/images/img0.jpg")

        dest = app.config["EXPORTS"] / "out"
        client.post(f"/p/{project['slug']}/annotate/{batch['id']}/publish",
                    data={"dest_path": str(dest), "reviewed_only": "on",
                          "include_images": "on"})
        assert [p.name for p in (dest / "images").iterdir()] == ["img1.jpg"]

    def test_an_unknown_image_is_404(self, app, client, project, source):
        batch = make_batch(app, client, project, source)
        assert client.delete(
            f"/api/batches/{batch['id']}/images/nope.jpg").status_code == 404

    @pytest.mark.parametrize("attack", [
        "../../../../etc/passwd", "sub/../../../secret.txt",
    ])
    def test_traversal_cannot_delete_outside_the_batch(self, app, client, project,
                                                       source, attack):
        """Deletion takes a path from a URL, so it gets the same containment
        check as serving one."""
        batch = make_batch(app, client, project, source)
        victim = Path(batch["dir_path"]).parent / "secret.txt"
        victim.write_text("do not delete me")
        assert client.delete(
            f"/api/batches/{batch['id']}/images/{attack}").status_code == 404
        assert victim.exists()

    def test_the_button_and_shortcut_are_offered(self, app, client, project, source):
        batch = make_batch(app, client, project, source)
        r = client.get(f"/p/{project['slug']}/annotate/{batch['id']}")
        assert b'id="del-image"' in r.data
        assert b"Shift+Delete" in r.data
