"""The web form's "Training settings" defaults match each pipeline's own defaults."""

import json
import re
from pathlib import Path

import pytest

from sparrow_finetune.pipelines import conventional_classifier, dinov3_linear, yolo_detector

APP_JS = Path(__file__).resolve().parents[2] / "src/sparrow_finetune/ui/app.js"
MODULES = {
    "yolo_detector": yolo_detector,
    "conventional_classifier": conventional_classifier,
    "dinov3_linear": dinov3_linear,
}


def _settings():
    m = re.search(r"const TRAINING_SETTINGS = (\{.*?\n\});", APP_JS.read_text(), re.S)
    assert m, "TRAINING_SETTINGS not found in app.js"
    return json.loads(m.group(1))


def test_every_pipeline_with_settings_is_known():
    assert set(_settings()) == set(MODULES)


@pytest.mark.parametrize("pipeline", sorted(MODULES))
def test_form_defaults_match_pipeline_params(pipeline):
    spec = _settings()[pipeline]
    params = MODULES[pipeline]._PARAMS
    default = lambda name: params[name][1]  # noqa: E731
    check = lambda name, v: params[name][2](v)  # noqa: E731
    epochs = spec["epochs"]
    assert default(epochs["param"]) == epochs["value"]
    assert check(epochs["param"], epochs["max"]) and not check(epochs["param"], epochs["max"] + 1)
    assert default("val_fraction") == spec["val_fraction"]
    assert all(check("val_fraction", v) for v in (0.1, 0.15, 0.2, 0.25, 0.3))
    if "patience" in spec:
        assert default("patience") == spec["patience"]
        assert check("patience", 0) and check("patience", 1000)
    else:
        assert "patience" not in params
    if "augment" in spec:
        assert default("augment") is spec["augment"]
    if "lr" in spec:
        assert default(spec["lr"]["param"]) == spec["lr"]["value"]
