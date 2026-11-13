"""A fake Sparrow Studio for the connector tests, served through ``httpx.MockTransport``.

Implements the 2A contract (docs/review/implement-phase2-service/spec.md, "Studio
connector"), base ``<studio_url>/api/finetune/v1``:

- ``GET /projects/{project}/export?reviewed=1|0&cursor=&limit=`` -- paginated; the
  fake caps every page at ``page_size`` so pagination is always exercised.
- ``GET /images/{image_id}`` -- PNG bytes of the declared width/height.
- ``POST /projects/{project}/models`` (multipart manifest + bundle) -> 201 ``{id}``.
- ``GET /projects/{project}/detectors`` -> ``{detectors}`` (``self.detectors``);
- ``POST /projects/{project}/models/{id}/activate`` -> 202 ``{id, install_status}``
  (the JSON body, if any, is kept in ``activation_bodies``);
  ``fail_activate = True`` makes it answer 422.
- ``GET /projects`` -> ``{projects: [{id, name}]}``: the fake's project plus
  ``extra_projects``.
- ``GET /projects/{project}/models`` -> ``{models: [...]}``: this project's models (the
  uploads plus anything added with ``add_model``).
- ``GET /projects/{project}/models/{id}`` -> ``{id, install_status, install_error}``;
- ``GET /projects/{project}/models/{id}/bundle`` -> the stored tar.gz bundle bytes with
  ``X-Artifact-Sha256``; another project's id answers 404.

Every request is recorded. ``fail = True`` makes every route answer 500;
``bad_json = True`` makes the export answer 200 with a non-JSON body.
"""

from __future__ import annotations

import json
import threading
from typing import Optional

import httpx

from .helpers import png_bytes

API = "/api/finetune/v1"


def studio_item(
    image_id: int,
    name: str,
    boxes: list,
    reviewed: bool = True,
    size: tuple[int, int] = (64, 48),
) -> dict:
    """``boxes``: ``[(box_id, label, [x, y, w, h]), ...]``."""
    return {
        "image_id": image_id,
        "name": name,
        "width": size[0],
        "height": size[1],
        "reviewed": reviewed,
        "boxes": [
            {"id": bid, "label": label, "bbox": list(bbox)}
            for bid, label, bbox in boxes
        ],
    }


def default_items() -> list:
    """Five Studio images: three reviewed with boxes, one unreviewed, one reviewed with
    zero boxes (must be skipped)."""
    return [
        studio_item(
            101, "a.jpg", [(1, "cat", [2, 3, 20, 10]), (2, "dog", [30, 10, 16, 20])]
        ),
        studio_item(102, "b.png", [(3, "cat", [0, 0, 64, 48])]),
        studio_item(103, "c.png", [(4, "fox", [10, 8, 12, 12])]),
        studio_item(104, "d.png", [(5, "dog", [5, 5, 10, 10])], reviewed=False),
        studio_item(105, "e.png", [], reviewed=True),
    ]


class FakeStudio:
    def __init__(
        self,
        project: str = "p1",
        items: Optional[list] = None,
        page_size: int = 2,
        project_name: str = "Project One",
    ) -> None:
        self.project = project
        self.project_name = project_name
        self.items = default_items() if items is None else items
        self.page_size = page_size
        self.requests: list[httpx.Request] = []
        self.uploads: list[httpx.Request] = []
        self.fail = False
        self.bad_json = False
        self.fail_activate = False
        self.activations: list[str] = []
        self.activation_bodies: list[Optional[dict]] = []
        self.detectors: list[dict] = [
            {
                "pipeline_id": "1",
                "detector_key": "MDV6-yolov10-e",
                "name": "MegaDetector V6",
                "fine_tuned": False,
                "detector_kind": "generic",
                "usable": True,
                "reason": None,
                "current": True,
            },
            {
                "pipeline_id": "9",
                "detector_key": "ft-p1-7",
                "name": "Fine-tuned species detector",
                "fine_tuned": True,
                "detector_kind": "species",
                "usable": False,
                "reason": "a fine-tuned species detector is used alone",
                "current": False,
            },
        ]
        # Studio id -> {install_status, install_error}; activation sets "pending".
        self.installs: dict[str, dict] = {}
        # When set, activation waits for this event (lets a test see the busy state).
        self.activate_gate: Optional[threading.Event] = None
        self._next_model = 0
        # id -> {"project", "manifest", "bundle", "sha256"}
        self.models: dict[str, dict] = {}
        self.extra_projects: list[dict] = []

    # ----------------------------------------------------------------- helpers

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def image_bytes(self, image_id: int) -> bytes:
        item = next(i for i in self.items if i["image_id"] == image_id)
        return png_bytes(seed=image_id, size=(item["width"], item["height"]))

    def export_requests(self) -> list:
        return [r for r in self.requests if r.url.path.endswith("/export")]

    def add_model(
        self,
        manifest: dict,
        bundle: bytes,
        project: Optional[str] = None,
        sha256: Optional[str] = None,
    ) -> str:
        """Store a model as if it had been pushed; return its Studio id.

        ``sha256`` overrides the ``X-Artifact-Sha256`` header (default: the manifest's)."""
        self._next_model += 1
        model_id = f"studio-model-{self._next_model}"
        self.models[model_id] = {
            "project": project or self.project,
            "manifest": manifest,
            "bundle": bundle,
            "sha256": manifest.get("sha256") if sha256 is None else sha256,
        }
        return model_id

    # ----------------------------------------------------------------- routes

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.fail:
            return httpx.Response(500, json={"error": "simulated Studio failure"})
        path = request.url.path
        if not path.startswith(API + "/"):
            return httpx.Response(404, json={"error": f"no route {path}"})
        parts = path[len(API) + 1 :].split("/")
        if request.method == "GET" and parts == ["projects"]:
            projects = [{"id": self.project, "name": self.project_name}]
            return httpx.Response(200, json={"projects": projects + self.extra_projects})
        if (
            request.method == "GET"
            and len(parts) == 3
            and parts[0] == "projects"
            and parts[2] == "export"
        ):
            return self._export(request, parts[1])
        if (
            request.method == "GET"
            and len(parts) == 3
            and parts[0] == "projects"
            and parts[2] == "detectors"
        ):
            if parts[1] != self.project:
                return httpx.Response(404, json={"error": f"unknown project {parts[1]}"})
            return httpx.Response(200, json={"detectors": self.detectors})
        if request.method == "GET" and len(parts) == 2 and parts[0] == "images":
            return self._image(parts[1])
        if (
            request.method == "POST"
            and len(parts) == 3
            and parts[0] == "projects"
            and parts[2] == "models"
        ):
            return self._upload(request, parts[1])
        if (
            request.method == "GET"
            and len(parts) == 3
            and parts[0] == "projects"
            and parts[2] == "models"
        ):
            return self._list_models(parts[1])
        if (
            request.method == "GET"
            and len(parts) == 5
            and parts[0] == "projects"
            and parts[2] == "models"
            and parts[4] == "bundle"
        ):
            return self._model_bundle(parts[1], parts[3])
        if (
            request.method == "POST"
            and len(parts) == 5
            and parts[0] == "projects"
            and parts[2] == "models"
            and parts[4] == "activate"
        ):
            if parts[1] != self.project:
                return httpx.Response(
                    404, json={"error": f"unknown project {parts[1]}"}
                )
            if self.fail_activate:
                return httpx.Response(
                    422, json={"error": "simulated: no engine package"}
                )
            if self.activate_gate is not None:
                self.activate_gate.wait(10)
            self.activations.append(parts[3])
            self.activation_bodies.append(
                json.loads(request.content) if request.content else None
            )
            self.installs[parts[3]] = {"install_status": "pending", "install_error": None}
            return httpx.Response(
                202, json={"id": parts[3], "install_status": "pending"}
            )
        if (
            request.method == "GET"
            and len(parts) == 4
            and parts[0] == "projects"
            and parts[2] == "models"
        ):
            return self._install_status(parts[1], parts[3])
        return httpx.Response(404, json={"error": f"no route {request.method} {path}"})

    def _install_status(self, project: str, model_id: str) -> httpx.Response:
        model = self.models.get(model_id)
        if model is None or model["project"] != project:
            return httpx.Response(404, json={"error": "model not found"})
        state = self.installs.get(
            model_id, {"install_status": "not_installed", "install_error": None}
        )
        return httpx.Response(200, json={"id": model_id, **state, "engine_model_id": None})

    def _export(self, request: httpx.Request, project: str) -> httpx.Response:
        if project != self.project:
            return httpx.Response(404, json={"error": f"unknown project {project}"})
        if self.bad_json:
            return httpx.Response(
                200,
                content=b"<html>not json</html>",
                headers={"Content-Type": "application/json"},
            )
        q = request.url.params
        reviewed_only = q.get("reviewed", "1") == "1"
        pool = [i for i in self.items if i["reviewed"] or not reviewed_only]
        start = int(q.get("cursor") or 0)
        limit = min(int(q.get("limit") or self.page_size), self.page_size)
        page = pool[start : start + limit]
        nxt = start + limit
        return httpx.Response(
            200,
            json={
                "project": {"id": self.project, "name": self.project_name},
                "items": page,
                "next_cursor": str(nxt) if nxt < len(pool) else None,
            },
        )

    def _image(self, image_id: str) -> httpx.Response:
        try:
            data = self.image_bytes(int(image_id))
        except (StopIteration, ValueError):
            return httpx.Response(404, json={"error": f"unknown image {image_id}"})
        return httpx.Response(200, content=data, headers={"Content-Type": "image/png"})

    def _upload(self, request: httpx.Request, project: str) -> httpx.Response:
        if project != self.project:
            return httpx.Response(404, json={"error": f"unknown project {project}"})
        request.read()
        self.uploads.append(request)
        parts = multipart_parts(request)
        model_id = self.add_model(manifest_from(parts), parts["bundle"], project)
        return httpx.Response(201, json={"id": model_id})

    def _list_models(self, project: str) -> httpx.Response:
        if project != self.project:
            return httpx.Response(404, json={"error": f"unknown project {project}"})
        out = []
        for model_id, m in self.models.items():
            if m["project"] != project:
                continue
            man = m["manifest"]
            out.append(
                {
                    "id": model_id,
                    "source_model_id": man.get("model_id"),
                    "pipeline": man.get("pipeline"),
                    "task": man.get("task"),
                    "format": man.get("format"),
                    "labels": man.get("labels"),
                    "metrics": man.get("metrics"),
                    "sha256": man.get("sha256"),
                    "size_bytes": len(m["bundle"]),
                    "created_at": man.get("created_at"),
                    "manifest": man,
                }
            )
        return httpx.Response(200, json={"models": out})

    def _model_bundle(self, project: str, model_id: str) -> httpx.Response:
        m = self.models.get(model_id)
        if m is None or m["project"] != project:
            return httpx.Response(404, json={"error": f"unknown model {model_id}"})
        headers = {"Content-Type": "application/gzip"}
        if m["sha256"]:
            headers["X-Artifact-Sha256"] = m["sha256"]
        return httpx.Response(200, content=m["bundle"], headers=headers)


def multipart_parts(request: httpx.Request) -> dict:
    """Parse a recorded multipart/form-data request -> ``{field name: bytes}``."""
    from email import policy
    from email.parser import BytesParser

    ctype = request.headers.get("content-type", "")
    assert ctype.startswith("multipart/form-data"), ctype
    raw = b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + request.content
    msg = BytesParser(policy=policy.HTTP).parsebytes(raw)
    assert msg.is_multipart(), "upload body is not multipart"
    parts = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        parts[name] = part.get_payload(decode=True)
    return parts


def manifest_from(parts: dict) -> dict:
    return json.loads(parts["manifest"])
