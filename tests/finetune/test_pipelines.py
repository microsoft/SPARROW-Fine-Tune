"""AC4: the pipeline catalogue merges planned pipelines with registered trainers."""

from __future__ import annotations

from sparrow_finetune.pipelines import Registry

from .fakes import PlannedOverrideTrainer
from .helpers import as_list, make_client

PLANNED = {
    "dinov3_linear": "classification",
    "conventional_classifier": "classification",
    "yolo_detector": "detection",
}


def _by_id(client) -> dict:
    r = client.get("/v1/pipelines")
    assert r.status_code == 200, r.text
    return {p["id"]: p for p in as_list(r.json(), "pipelines")}


def test_ac4_planned_pipelines_listed_as_planned(client):
    pipelines = _by_id(client)
    for pid, task in PLANNED.items():
        assert pipelines[pid]["status"] == "planned"
        assert pipelines[pid]["task"] == task


def test_ac4_registered_trainer_listed_as_available_with_task(client):
    pipelines = _by_id(client)
    assert pipelines["fake_cls"]["status"] == "available"
    assert pipelines["fake_cls"]["task"] == "classification"
    assert pipelines["fake_gpu"]["status"] == "available"


def test_ac4_registering_planned_id_makes_it_available(settings):
    reg = Registry()
    reg.register(PlannedOverrideTrainer())
    with make_client(settings, reg) as c:
        pipelines = _by_id(c)
    assert pipelines["dinov3_linear"]["status"] == "available"
    assert pipelines["dinov3_linear"]["task"] == "classification"
    assert pipelines["yolo_detector"]["status"] == "planned"


def test_ac4_registry_get_and_list():
    reg = Registry()
    trainer = PlannedOverrideTrainer()
    reg.register(trainer)
    assert reg.get("dinov3_linear") is trainer
    assert trainer in list(reg.list())
