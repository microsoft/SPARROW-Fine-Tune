"""HTTP surface: pages render, the API answers, and the artifact route is safe.

The path-traversal test is the important one — this process runs as root with no
authentication, so an unvalidated file route would serve anything on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

flask = pytest.importorskip("flask", reason="Flask not installed")

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
    # No supervisor: background threads make tests non-deterministic, and every
    # behaviour they drive is tested directly elsewhere.
    application = create_app(settings, start_supervisor=False)
    application.config["TESTING"] = True
    application.config["DATASETS"] = datasets
    return application


@pytest.fixture
def client(app):
    return app.test_client()


def make_project(app, name="Sub Saharan", subdir="Sub_Saharan"):
    root = app.config["DATASETS"] / subdir
    root.mkdir(parents=True, exist_ok=True)
    projects = Projects(app.config["DB"])
    return projects.get(projects.create(name, str(root)))


class TestPages:
    def test_index_renders_empty_state(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert b"No projects yet" in r.data

    def test_index_lists_projects(self, app, client):
        make_project(app)
        r = client.get("/")
        assert b"Sub Saharan" in r.data

    def test_project_page_shows_all_steps(self, app, client):
        project = make_project(app)
        r = client.get(f"/p/{project['slug']}")
        assert r.status_code == 200
        for label in (b"Data", b"Review", b"Prepare", b"Train", b"Evaluate", b"Export"):
            assert label in r.data

    def test_locked_steps_explain_themselves(self, app, client):
        """A disabled control with no reason is the worst part of a stepped UI."""
        project = make_project(app)
        r = client.get(f"/p/{project['slug']}")
        assert b"Scan the dataset first" in r.data

    def test_review_page_before_scanning(self, app, client):
        project = make_project(app)
        r = client.get(f"/p/{project['slug']}/review")
        assert r.status_code == 200
        assert b"Not scanned yet" in r.data

    def test_system_page(self, client):
        assert client.get("/system").status_code == 200

    def test_unknown_project_is_404(self, client):
        assert client.get("/p/nope").status_code == 404

    def test_healthz(self, client):
        assert json.loads(client.get("/healthz").data)["ok"] is True


class TestCreateProject:
    def test_happy_path_redirects_to_the_project(self, app, client):
        (app.config["DATASETS"] / "ds").mkdir()
        r = client.post("/projects", data={
            "name": "My Project", "dataset_root": str(app.config["DATASETS"] / "ds")})
        assert r.status_code == 302
        assert "/p/my_project" in r.headers["Location"]

    def test_rejects_a_root_outside_the_allowlist(self, client):
        """No auth means anyone reachable could otherwise point this at /."""
        r = client.post("/projects", data={"name": "Bad", "dataset_root": "/etc"})
        assert r.status_code == 400
        assert b"outside the allowed locations" in r.data

    def test_rejects_a_missing_folder(self, app, client):
        r = client.post("/projects", data={
            "name": "Bad", "dataset_root": str(app.config["DATASETS"] / "nope")})
        assert r.status_code == 400
        assert b"does not exist" in r.data

    def test_requires_name_and_root(self, client):
        assert client.post("/projects", data={"name": "", "dataset_root": ""}).status_code == 400


class TestApi:
    def test_gpus(self, client):
        assert "gpus" in json.loads(client.get("/api/gpus").data)

    def test_steps(self, app, client):
        project = make_project(app)
        data = json.loads(client.get(f"/api/projects/{project['id']}/steps").data)
        assert [s["id"] for s in data["steps"]] == \
            ["data", "review", "prepare", "train", "evaluate", "export"]

    def test_api_404_returns_json_not_html(self, client):
        r = client.get("/api/projects/999/steps")
        assert r.status_code == 404
        assert r.headers["Content-Type"].startswith("application/json")

    def test_metrics_for_a_run(self, app, client, tmp_path):
        project = make_project(app)
        db = app.config["DB"]
        with db.write() as c:
            c.execute("INSERT INTO runs(project_id,name,run_dir,state,epochs,queued_at,updated_at)"
                      " VALUES (?,'r',?,'running',100,?,?)",
                      (project["id"], str(tmp_path / "run"), utcnow(), utcnow()))
            c.executemany("INSERT INTO metric_samples(run_id,epoch,key,value) VALUES (1,?,?,?)",
                          [(1, "map50", 0.5), (2, "map50", 0.6)])
        data = json.loads(client.get("/api/runs/1/metrics").data)
        assert data["series"]["map50"] == [[1, 0.5], [2, 0.6]]
        assert data["run"]["state"] == "running"


class TestArtifactSafety:
    def _run(self, app, tmp_path):
        project = make_project(app)
        run_dir = tmp_path / "runs" / "r1"
        run_dir.mkdir(parents=True)
        (run_dir / "confusion_matrix.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
        with app.config["DB"].write() as c:
            c.execute("INSERT INTO runs(project_id,name,run_dir,save_dir,state,queued_at,"
                      "updated_at) VALUES (?,'r1',?,?,'completed',?,?)",
                      (project["id"], str(run_dir), str(run_dir), utcnow(), utcnow()))
        return run_dir

    def test_serves_a_real_artifact(self, app, client, tmp_path):
        self._run(app, tmp_path)
        r = client.get("/runs/1/artifacts/confusion_matrix.png")
        assert r.status_code == 200 and r.data.startswith(b"\x89PNG")

    @pytest.mark.parametrize("attack", [
        "../../../../etc/passwd",
        "..%2f..%2f..%2fetc%2fpasswd",
        "subdir/../../../secret.txt",
    ])
    def test_path_traversal_is_refused(self, app, client, tmp_path, attack):
        """This process runs as root with no auth; the route must be sealed."""
        run_dir = self._run(app, tmp_path)
        (tmp_path / "secret.txt").write_text("do not serve me")
        r = client.get(f"/runs/1/artifacts/{attack}")
        assert r.status_code == 404
        assert b"do not serve me" not in r.data

    def test_missing_artifact_is_404(self, app, client, tmp_path):
        self._run(app, tmp_path)
        assert client.get("/runs/1/artifacts/nope.png").status_code == 404


class TestStream:
    def test_rejects_beyond_the_client_limit(self, tmp_path):
        """Abandoned tabs must not exhaust the worker's thread pool."""
        settings = Settings(state_dir=tmp_path / "s", dataset_search_roots=[tmp_path],
                            max_sse_clients=1)
        app = create_app(settings, start_supervisor=False)
        app.config["TESTING"] = True
        bus = app.config["BUS"]
        bus.subscribe({"gpu"})          # occupy the only slot
        assert app.test_client().get("/stream?topics=gpu").status_code == 503


class TestFolderBrowser:
    """The picker walks the filesystem, so its allowlist check is load-bearing."""

    def _tree(self, app):
        root = app.config["DATASETS"]
        (root / "Sub_Saharan" / "images" / "v8").mkdir(parents=True)
        (root / "Sub_Saharan" / "images" / "v8" / "a.jpg").write_bytes(b"x")
        (root / "Sub_Saharan" / "images" / "v8" / "a.xml").write_text("<annotation/>")
        (root / "Empty_Project").mkdir()
        return root

    def test_lists_subfolders_from_the_root(self, app, client):
        root = self._tree(app)
        data = json.loads(client.get(f"/api/browse?path={root}").data)
        names = [e["name"] for e in data["entries"]]
        assert "Sub_Saharan" in names and "Empty_Project" in names
        assert data["parent"] is None, "must not offer to navigate above the root"

    def test_marks_folders_that_hold_images(self, app, client):
        root = self._tree(app)
        target = root / "Sub_Saharan" / "images"
        data = json.loads(client.get(f"/api/browse?path={target}").data)
        v8 = next(e for e in data["entries"] if e["name"] == "v8")
        assert v8["is_dataset"] is True and v8["images"] == 1 and v8["xmls"] == 1

    def test_reports_counts_for_the_current_folder(self, app, client):
        root = self._tree(app)
        target = root / "Sub_Saharan" / "images" / "v8"
        data = json.loads(client.get(f"/api/browse?path={target}").data)
        assert data["here"]["images"] == 1
        assert data["parent"] is not None

    def test_breadcrumbs_stop_at_the_allowed_root(self, app, client):
        root = self._tree(app)
        target = root / "Sub_Saharan" / "images" / "v8"
        data = json.loads(client.get(f"/api/browse?path={target}").data)
        assert [c["name"] for c in data["breadcrumbs"]][-3:] == ["Sub_Saharan", "images", "v8"]

    @pytest.mark.parametrize("attack", ["/etc", "/", "/root"])
    def test_refuses_paths_outside_the_allowlist(self, app, client, attack):
        """Without this, ?path=/ walks the whole filesystem as root."""
        self._tree(app)
        r = client.get(f"/api/browse?path={attack}")
        assert r.status_code == 403

    def test_traversal_out_of_the_root_is_refused(self, app, client):
        root = self._tree(app)
        r = client.get(f"/api/browse?path={root}/../../../../etc")
        assert r.status_code == 403

    def test_missing_folder_is_404(self, app, client):
        root = self._tree(app)
        assert client.get(f"/api/browse?path={root}/nope").status_code == 404

    def test_new_project_page_renders_the_picker(self, app, client):
        self._tree(app)
        r = client.get("/projects/new")
        assert b'id="browser"' in r.data and b"Choose a folder" in r.data


class TestScanUx:
    """The scan button must start work without leaving the page, and must not be
    clickable twice — both were reported after the first round of testing."""

    def _snapshot(self, app, project):
        with app.config["DB"].write() as c:
            c.execute(
                "INSERT INTO dataset_snapshots(project_id,dataset_root,audit_params_json,"
                "content_hash,quick_hash,n_images,n_readable,n_labelled,n_background,"
                "n_boxes,n_classes,n_issues,blank_empty_xml,blank_confirmed_dir,"
                "blank_unknown,artifact_dir,summary_json,last_verified_at,created_at)"
                " VALUES (?,?,'{}','h','q',100,100,90,10,300,4,0,7,3,0,?,'{}',?,?)",
                (project["id"], project["dataset_root"], project["dataset_root"],
                 utcnow(), utcnow()))

    def test_scan_returns_json_to_fetch(self, app, client):
        project = make_project(app)
        r = client.post(f"/p/{project['slug']}/scan", headers={"Accept": "application/json"})
        assert r.status_code == 202
        assert json.loads(r.data)["already_running"] is False

    def test_second_scan_does_not_queue_a_duplicate(self, app, client):
        project = make_project(app)
        client.post(f"/p/{project['slug']}/scan", headers={"Accept": "application/json"})
        r = client.post(f"/p/{project['slug']}/scan", headers={"Accept": "application/json"})
        assert json.loads(r.data)["already_running"] is True
        n = app.config["DB"].one(
            "SELECT COUNT(*) c FROM jobs WHERE project_id=? AND kind='audit'",
            (project["id"],))["c"]
        assert n == 1

    def test_step_action_posts_instead_of_navigating(self, app, client):
        """Clicking 'Scan dataset' used to navigate to another page."""
        project = make_project(app)
        r = client.get(f"/p/{project['slug']}")
        assert f'data-post="/p/{project["slug"]}/scan"'.encode() in r.data

    def test_review_page_has_a_back_button(self, app, client):
        project = make_project(app)
        r = client.get(f"/p/{project['slug']}/review")
        assert f'href="/p/{project["slug"]}"'.encode() in r.data
        assert "← Back".encode() in r.data


class TestDatasetPage:
    """'Build dataset' 404'd because this page did not exist."""

    def test_page_renders_before_any_scan(self, app, client):
        project = make_project(app)
        r = client.get(f"/p/{project['slug']}/dataset")
        assert r.status_code == 200
        assert b"Scan the dataset first" in r.data

    def test_page_offers_a_build_once_scanned(self, app, client):
        project = make_project(app)
        TestScanUx()._snapshot(app, project)
        r = client.get(f"/p/{project['slug']}/dataset")
        assert r.status_code == 200
        assert b"Build training set" in r.data
        assert f'action="/p/{project["slug"]}/prepare"'.encode() in r.data

    def test_prepare_without_a_scan_redirects_to_review(self, app, client):
        project = make_project(app)
        r = client.post(f"/p/{project['slug']}/prepare")
        assert r.status_code == 302 and "/review" in r.headers["Location"]

    def test_prepare_queues_one_job(self, app, client):
        project = make_project(app)
        TestScanUx()._snapshot(app, project)
        client.post(f"/p/{project['slug']}/prepare")
        client.post(f"/p/{project['slug']}/prepare")
        n = app.config["DB"].one(
            "SELECT COUNT(*) c FROM jobs WHERE project_id=? AND kind='prepare'",
            (project["id"],))["c"]
        assert n == 1


class TestDeleteProject:
    """Removing a project must be reachable, guarded, and obviously safe for the
    source data — the user's whole concern was the dataset folder."""

    def test_the_project_page_offers_it(self, app, client):
        project = make_project(app)
        r = client.get(f"/p/{project['slug']}")
        assert f'href="/p/{project["slug"]}/delete"'.encode() in r.data

    def test_the_page_says_the_dataset_is_safe(self, app, client):
        project = make_project(app)
        r = client.get(f"/p/{project['slug']}/delete")
        assert r.status_code == 200
        assert b"Your images and annotations are safe" in r.data
        assert project["dataset_root"].encode() in r.data

    def test_a_wrong_confirmation_deletes_nothing(self, app, client):
        project = make_project(app)
        r = client.post(f"/p/{project['slug']}/delete", data={"confirm": "sub saharan"})
        assert r.status_code == 400
        assert client.get(f"/p/{project['slug']}").status_code == 200

    def test_deleting_removes_it_and_leaves_the_folder(self, app, client):
        project = make_project(app)
        root = Path(project["dataset_root"])
        (root / "keep.jpg").write_bytes(b"jpeg")

        r = client.post(f"/p/{project['slug']}/delete",
                        data={"confirm": project["name"], "remove_files": "on"})
        assert r.status_code == 302 and r.headers["Location"].endswith("/")
        assert client.get(f"/p/{project['slug']}").status_code == 404
        assert (root / "keep.jpg").read_bytes() == b"jpeg"

    def test_a_running_job_blocks_it(self, app, client):
        project = make_project(app)
        client.post(f"/p/{project['slug']}/scan", headers={"Accept": "application/json"})
        r = client.post(f"/p/{project['slug']}/delete", data={"confirm": project["name"]})
        assert r.status_code == 409
        assert b"Something is still running" in r.data
        assert client.get(f"/p/{project['slug']}").status_code == 200


class TestPreparedDirectories:
    """Re-preparing must not overwrite the tree an existing model was tested on."""

    def _snapshot(self, app, project):
        with app.config["DB"].write() as c:
            cur = c.execute(
                "INSERT INTO dataset_snapshots(project_id,dataset_root,audit_params_json,"
                "content_hash,quick_hash,artifact_dir,summary_json,last_verified_at,created_at)"
                " VALUES (?,?,'{}','h','q','','{}',?,?)",
                (project["id"], project["dataset_root"], utcnow(), utcnow()))
            return int(cur.lastrowid)

    def _queued_dir(self, app, client, project):
        client.post(f"/p/{project['slug']}/prepare")
        job = app.config["DB"].one(
            "SELECT * FROM jobs WHERE kind='prepare' ORDER BY id DESC LIMIT 1")
        return json.loads(job["params_json"])["prepared_name"]

    def test_first_prepare_is_named_after_the_snapshot(self, app, client):
        project = make_project(app)
        snapshot_id = self._snapshot(app, project)
        assert self._queued_dir(app, client, project) == str(snapshot_id)

    def test_a_second_prepare_of_the_same_snapshot_gets_its_own_directory(
            self, app, client):
        """Otherwise the rebuild deletes the tree the previous model was
        evaluated against, and its test score stops being reproducible."""
        project = make_project(app)
        snapshot_id = self._snapshot(app, project)
        first = self._queued_dir(app, client, project)

        prepared = app.config["SETTINGS"].prepared_root / project["slug"] / first
        prepared.mkdir(parents=True)
        with app.config["DB"].write() as c:
            c.execute(
                "INSERT INTO prepared_datasets(project_id,snapshot_id,dir_path,data_yaml,"
                "class_map_json,n_classes,split_config_json,background_policy,splits_hash,"
                "created_at) VALUES (?,?,?,'','{}',1,'{}','reviewed','h',?)",
                (project["id"], snapshot_id, str(prepared), utcnow()))
            c.execute("UPDATE jobs SET state='succeeded' WHERE kind='prepare'")

        second = self._queued_dir(app, client, project)
        assert second != first
        assert second == f"{snapshot_id}-2"

    def test_a_directory_left_on_disk_is_also_avoided(self, app, client):
        """A prepare that crashed after writing leaves a tree with no row."""
        project = make_project(app)
        snapshot_id = self._snapshot(app, project)
        (app.config["SETTINGS"].prepared_root / project["slug"] / str(snapshot_id)
         ).mkdir(parents=True)
        assert self._queued_dir(app, client, project) == f"{snapshot_id}-2"
