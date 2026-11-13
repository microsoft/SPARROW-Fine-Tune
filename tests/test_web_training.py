"""The training setup form, and the record it leaves behind.

Two things are being protected here. First, a bad number must be refused before
a run is queued — a typo that surfaces four hours into training is expensive.
Second, every setting chosen must be stored against the run: a model whose
hyperparameters were not recorded cannot be reproduced or compared.
"""

from __future__ import annotations

import json

import pytest

flask = pytest.importorskip("flask", reason="Flask not installed")

from sparrow_web import training
from sparrow_web.app import create_app
from sparrow_web.db import utcnow
from sparrow_web.repo import Projects
from sparrow_web.settings import Settings


@pytest.fixture
def app(tmp_path):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    settings = Settings(
        state_dir=tmp_path / "state",
        prepared_root=tmp_path / "prepared",
        runs_root=tmp_path / "runs",
        dataset_search_roots=[datasets],
    )
    application = create_app(settings, start_supervisor=False)
    application.config["TESTING"] = True
    application.config["DATASETS"] = datasets
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


def add_prepared(app, project):
    """A ready training set, which is what unlocks the train step."""
    db = app.config["DB"]
    with db.write() as c:
        cur = c.execute(
            "INSERT INTO dataset_snapshots(project_id,dataset_root,audit_params_json,"
            "content_hash,quick_hash,artifact_dir,summary_json,last_verified_at,created_at)"
            " VALUES (?,?,'{}','h','q','','{}',?,?)",
            (project["id"], project["dataset_root"], utcnow(), utcnow()))
        snapshot_id = int(cur.lastrowid)
    d = app.config["SETTINGS"].prepared_root / "sub_saharan" / "1"
    d.mkdir(parents=True)
    (d / "data.yaml").write_text("names: [a]")
    with db.write() as c:
        c.execute(
            "INSERT INTO prepared_datasets(project_id,snapshot_id,dir_path,data_yaml,"
            "class_map_json,n_classes,split_config_json,background_policy,splits_hash,"
            "n_train,n_val,n_test,created_at)"
            " VALUES (?,?,?,?,'{}',4,'{}','reviewed','h',800,100,100,?)",
            (project["id"], snapshot_id, str(d), str(d / "data.yaml"), utcnow()))
    return d


def form(**overrides):
    """A complete, valid submission — the form posts every field."""
    data = {s.name: s.default for s in training.SETTINGS}
    data.update({"model": "yolo11l.pt", "device": "0", "batch": "auto",
                 "start_from": "coco"})
    # Checkboxes only appear in a POST when ticked.
    data = {k: v for k, v in data.items()
            if not (training.BY_NAME.get(k) and training.BY_NAME[k].kind == "bool"
                    and not v)}
    data.update(overrides)
    return {k: ("on" if v is True else str(v)) for k, v in data.items()}


class TestSettingsSpec:
    def test_every_setting_has_a_group_we_render(self):
        for s in training.SETTINGS:
            assert s.group in training.GROUP_ORDER, s.name

    def test_names_are_unique(self):
        names = [s.name for s in training.SETTINGS]
        assert len(names) == len(set(names))

    def test_defaults_validate(self):
        """The form's initial state must be submittable without edits."""
        values = training.parse(form())
        assert values["epochs"] == 100 and values["imgsz"] == 1024

    def test_launcher_keys_are_split_from_passthrough(self):
        params = training.split_params(training.parse(form()))
        assert params["imgsz"] == 1024 and params["epochs"] == 100
        # Anything the launcher does not name goes through `extra` untouched.
        assert params["extra"]["cos_lr"] is True
        assert "imgsz" not in params["extra"] and "model" not in params["extra"]


class TestBatch:
    @pytest.mark.parametrize("raw,expected", [
        ("auto", "auto"), ("", "auto"), (None, "auto"),
        ("16", 16), ("-1", -1), ("0.8", 0.8), ("0.85", 0.85),
    ])
    def test_accepted_forms(self, raw, expected):
        assert training.parse_batch(raw) == expected

    def test_a_fraction_stays_a_float(self):
        """0.7 is Ultralytics' GPU-utilisation target; as an int it becomes 0."""
        assert isinstance(training.parse_batch("0.7"), float)

    @pytest.mark.parametrize("raw", ["big", "0", "-4", "1.5"])
    def test_rejected_forms(self, raw):
        with pytest.raises(ValueError):
            training.parse_batch(raw)


class TestValidation:
    def test_out_of_range_is_reported_per_field(self):
        with pytest.raises(training.ValidationError) as e:
            training.parse(form(epochs="0", lr0="5"))
        assert set(e.value.errors) == {"epochs", "lr0"}

    def test_image_size_must_be_a_multiple_of_32(self):
        with pytest.raises(training.ValidationError) as e:
            training.parse(form(imgsz="1000"))
        assert "imgsz" in e.value.errors

    def test_an_unknown_model_is_refused(self):
        with pytest.raises(training.ValidationError) as e:
            training.parse(form(model="yolo99z.pt"))
        assert "model" in e.value.errors

    def test_an_unticked_checkbox_means_false(self):
        values = training.parse({k: v for k, v in form().items() if k != "amp"})
        assert values["amp"] is False

    def test_a_zero_time_limit_is_dropped(self):
        """Ultralytics reads `time` as a limit whenever it is present."""
        assert "time" not in training.parse(form(time="0"))
        assert training.parse(form(time="6"))["time"] == 6.0


class TestPage:
    def test_locked_until_a_training_set_exists(self, app, client, project):
        r = client.get(f"/p/{project['slug']}/train")
        assert r.status_code == 200
        assert b"Build a training set first" in r.data

    def test_renders_the_form(self, app, client, project):
        add_prepared(app, project)
        r = client.get(f"/p/{project['slug']}/train")
        assert r.status_code == 200
        for name in ("epochs", "batch", "imgsz", "lr0", "mosaic", "freeze"):
            assert f'name="{name}"'.encode() in r.data

    def test_explains_transfer_learning_rather_than_assuming_it(self, app, client, project):
        add_prepared(app, project)
        r = client.get(f"/p/{project['slug']}/train")
        assert b"transfer learning" in r.data

    def test_batch_preview_endpoint(self, app, client, project):
        r = client.get(f"/api/projects/{project['id']}/batch?batch=32&imgsz=1024&device=0")
        assert r.status_code == 200
        assert json.loads(r.data)["batch"] == 32


class TestLaunch:
    def test_records_every_setting_against_the_run(self, app, client, project):
        add_prepared(app, project)
        r = client.post(f"/p/{project['slug']}/train",
                        data=form(epochs="42", lr0="0.002", freeze="10", mosaic="0.5"))
        assert r.status_code == 302

        run = app.config["DB"].one("SELECT * FROM runs WHERE project_id=?", (project["id"],))
        stored = json.loads(run["config_json"])
        assert stored["epochs"] == 42
        assert stored["lr0"] == 0.002
        assert stored["freeze"] == 10
        assert stored["mosaic"] == 0.5
        # Not a subset: the whole spec is captured, so a run stays explicable
        # even after the defaults in the code change.
        assert set(stored) >= {s.name for s in training.SETTINGS} - {"time"}

    def test_the_columns_used_for_listing_are_filled(self, app, client, project):
        add_prepared(app, project)
        client.post(f"/p/{project['slug']}/train", data=form(epochs="42", imgsz="1280"))
        run = app.config["DB"].one("SELECT * FROM runs WHERE project_id=?", (project["id"],))
        assert run["epochs"] == 42 and run["imgsz"] == 1280
        assert run["state"] == "queued" and run["batch"] == "auto"

    def test_the_job_carries_the_settings_to_the_trainer(self, app, client, project):
        add_prepared(app, project)
        client.post(f"/p/{project['slug']}/train", data=form(epochs="42", lr0="0.002"))
        job = app.config["DB"].one("SELECT * FROM jobs WHERE kind='train'")
        params = json.loads(job["params_json"])
        assert params["epochs"] == 42
        assert params["extra"]["lr0"] == 0.002
        assert params["data_yaml"].endswith("data.yaml")

    def test_it_leases_the_chosen_gpu(self, app, client, project):
        """The lease is the queue; without it two runs share one card."""
        add_prepared(app, project)
        client.post(f"/p/{project['slug']}/train", data=form(device="1"))
        job = app.config["DB"].one("SELECT * FROM jobs WHERE kind='train'")
        assert "gpu:1" in json.loads(job["needs_resources_json"])

    def test_transfer_from_a_previous_model_is_recorded(self, app, client, project):
        add_prepared(app, project)
        with app.config["DB"].write() as c:
            c.execute("INSERT INTO runs(project_id,name,run_dir,state,queued_at,updated_at)"
                      " VALUES (?,'earlier','/runs/earlier','completed',?,?)",
                      (project["id"], utcnow(), utcnow()))
            run_id = int(c.execute("SELECT last_insert_rowid() i").fetchone()[0])
            cur = c.execute(
                "INSERT INTO model_versions(project_id,run_id,version,weights_path,"
                "class_map_json,n_classes,created_at) VALUES (?,?,1,'/w/best.pt','{}',4,?)",
                (project["id"], run_id, utcnow()))
            model_id = int(cur.lastrowid)
        client.post(f"/p/{project['slug']}/train", data=form(start_from=f"model:{model_id}"))
        run = app.config["DB"].one(
            "SELECT * FROM runs WHERE project_id=? ORDER BY id DESC LIMIT 1",
            (project["id"],))
        assert run["base_kind"] == "model_version"
        assert run["base_model_version_id"] == model_id
        assert run["base_weights"] == "/w/best.pt"

    def test_a_bad_value_queues_nothing(self, app, client, project):
        add_prepared(app, project)
        r = client.post(f"/p/{project['slug']}/train", data=form(epochs="-5"))
        assert r.status_code == 400
        assert b"Nothing was started" in r.data
        assert app.config["DB"].one("SELECT COUNT(*) c FROM runs")["c"] == 0

    def test_a_bad_value_keeps_the_rest_of_the_form(self, app, client, project):
        """Losing thirty other settings to one typo is its own bug."""
        add_prepared(app, project)
        r = client.post(f"/p/{project['slug']}/train", data=form(epochs="-5", lr0="0.002"))
        assert b'value="0.002"' in r.data

    def test_a_second_launch_does_not_start_a_second_run(self, app, client, project):
        add_prepared(app, project)
        client.post(f"/p/{project['slug']}/train", data=form())
        client.post(f"/p/{project['slug']}/train", data=form())
        assert app.config["DB"].one("SELECT COUNT(*) c FROM runs")["c"] == 1

    def test_without_a_training_set_it_redirects_rather_than_queueing(
            self, app, client, project):
        r = client.post(f"/p/{project['slug']}/train", data=form())
        assert r.status_code == 302 and "/dataset" in r.headers["Location"]
        assert app.config["DB"].one("SELECT COUNT(*) c FROM runs")["c"] == 0


class TestRunPageShowsWhatWasChosen:
    def test_settings_are_listed_by_their_labels(self, app, client, project):
        add_prepared(app, project)
        client.post(f"/p/{project['slug']}/train", data=form(epochs="42", freeze="10"))
        run = app.config["DB"].one("SELECT * FROM runs")
        r = client.get(f"/p/{project['slug']}/runs/{run['id']}")
        assert r.status_code == 200
        assert b"Settings used" in r.data
        assert b"Early stopping patience" in r.data
        assert b"Freeze the first" in r.data

    def test_the_form_carries_the_last_run_forward(self, app, client, project):
        add_prepared(app, project)
        client.post(f"/p/{project['slug']}/train", data=form(epochs="42"))
        r = client.get(f"/p/{project['slug']}/train")
        assert b'value="42"' in r.data


class TestBatchPreview:
    """The preview must agree with the launch — it runs the same parser."""

    def test_a_fraction_is_understood(self, app, client, project):
        r = client.get(f"/api/projects/{project['id']}/batch?batch=0.8&imgsz=1024&device=0")
        body = json.loads(r.data)
        assert "error" not in body
        assert body["batch"] == 0.8
        assert "80%" in body["label"]

    def test_auto_is_measured_where_measuring_works(self, app, client, project,
                                                     monkeypatch):
        monkeypatch.setattr("sparrow_yolo.process_runner.is_wsl", lambda: False)
        body = json.loads(
            client.get(f"/api/projects/{project['id']}/batch?batch=auto&device=0").data)
        assert body["batch"] == -1 and "measured" in body["label"]

    def test_auto_is_a_concrete_number_on_wsl(self, app, client, project, monkeypatch):
        """WDDM hides real memory use, so the preview must not promise a
        measurement that will not happen."""
        monkeypatch.setattr("sparrow_yolo.process_runner.is_wsl", lambda: True)
        body = json.loads(
            client.get(f"/api/projects/{project['id']}/batch?batch=auto&device=0").data)
        assert isinstance(body["batch"], int) and body["batch"] > 0
        assert "images per step" in body["label"]

    def test_a_bad_value_answers_rather_than_erroring(self, app, client, project):
        """The form calls this on every keystroke-ish change; a 500 there would
        blank the hint while the user is mid-edit."""
        r = client.get(f"/api/projects/{project['id']}/batch?batch=nonsense&device=0")
        assert r.status_code == 200 and "error" in json.loads(r.data)


class TestReachability:
    """A settings page nobody can find is the same as no settings page."""

    def test_the_project_page_links_to_it(self, app, client, project):
        add_prepared(app, project)
        r = client.get(f"/p/{project['slug']}")
        assert f'href="/p/{project["slug"]}/train"'.encode() in r.data

    def test_the_action_does_not_claim_to_start_the_run(self, app, client, project):
        """It opens a form; only the form's own button launches anything."""
        add_prepared(app, project)
        r = client.get(f"/p/{project['slug']}")
        assert b"Set up training" in r.data

    def test_the_dataset_page_leads_on_to_it(self, app, client, project):
        add_prepared(app, project)
        r = client.get(f"/p/{project['slug']}/dataset")
        assert f'href="/p/{project["slug"]}/train"'.encode() in r.data

    def test_a_finished_run_offers_another(self, app, client, project):
        add_prepared(app, project)
        client.post(f"/p/{project['slug']}/train", data=form())
        run = app.config["DB"].one("SELECT * FROM runs")
        with app.config["DB"].write() as c:
            c.execute("UPDATE runs SET state='completed' WHERE id=?", (run["id"],))
        r = client.get(f"/p/{project['slug']}/runs/{run['id']}")
        assert b"Set up another run" in r.data


class TestRunPageMarkup:
    def test_the_epoch_counter_is_not_escaped(self, app, client, project):
        """`|safe` binds tighter than `~`, so marking only the last term of a
        concatenation printed the raw <span> on screen."""
        add_prepared(app, project)
        client.post(f"/p/{project['slug']}/train", data=form(epochs="42"))
        run = app.config["DB"].one("SELECT * FROM runs")
        r = client.get(f"/p/{project['slug']}/runs/{run['id']}")
        assert b'<span id="m-epoch">0</span>/42' in r.data
        assert b"&lt;span" not in r.data

    def test_every_headline_figure_has_a_target_to_update(self, app, client, project):
        """Each stat is filled in live from the stream; a missing id is a tile
        that stays at '-' for the whole run."""
        add_prepared(app, project)
        client.post(f"/p/{project['slug']}/train", data=form())
        run = app.config["DB"].one("SELECT * FROM runs")
        r = client.get(f"/p/{project['slug']}/runs/{run['id']}")
        for key in ("map50_95", "map50", "precision", "recall", "epoch", "eta"):
            assert f'id="m-{key}"'.encode() in r.data
