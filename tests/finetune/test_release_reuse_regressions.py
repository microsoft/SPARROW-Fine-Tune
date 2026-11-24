from .service_helpers import make_service_client, studio_settings
from .studio_fake import FakeStudio
from .test_base_models import seed_model
from .test_release_model_flows import (
    EFF_PARAMS,
    run_job,
    snap,
    svc,
)


def test_studio_listing_never_fetches_bundle_bytes(settings, registry):
    studio = FakeStudio(project="p1")
    model_id = seed_model(studio)
    with make_service_client(studio_settings(settings), registry, studio.transport) as client:
        response = client.get("/v1/catalog")
        assert response.status_code == 200
        assert str(model_id) in {str(row["id"]) for row in response.json()["models"]}
        assert not any(r.url.path.endswith("/bundle") for r in studio.requests)


def test_retrain_uses_declared_architecture_without_model_override(svc):
    client, settings, registry = svc
    params = {k: v for k, v in EFF_PARAMS.items() if k != "model"}
    snapshot = snap(client)
    first = run_job(client, settings, registry, "conventional_classifier", snapshot,
                    EFF_PARAMS)
    assert first["job"]["params"]["model"] == "efficientnet_b0"
    second = run_job(client, settings, registry, "conventional_classifier",
                     snap(client, prefix="child-"), {**params, "parent_model_id": first["model"]["id"]})
    assert second["job"]["params"]["model"] == "efficientnet_b0"
    assert second["model"]["metadata"]["parent_model_id"] == first["model"]["id"]
