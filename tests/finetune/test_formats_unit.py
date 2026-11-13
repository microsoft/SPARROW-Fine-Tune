"""Unit checks for the COCO / Camera Trap DP readers that the API tests don't reach."""

from __future__ import annotations

import io
import zipfile
from datetime import datetime, timezone

import pytest

from sparrow_finetune import camtrapdp, coco
from sparrow_finetune.interchange import FormatError


@pytest.mark.parametrize(
    "text",
    ["2024-05-01T10:00:00Z", "2024-05-01T12:00:00+0200", "2024-05-01T12:00:00+02:00", "2024-05-01T10:00:00"],
)
def test_parse_time_offsets(text):
    assert camtrapdp.parse_time(text) == datetime(2024, 5, 1, 10, tzinfo=timezone.utc)


def test_parse_time_rejects_garbage():
    assert camtrapdp.parse_time("") is None and camtrapdp.parse_time("yesterday") is None


def _zip(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, body in members.items():
            zf.writestr(name, body)
    return buf.getvalue()


@pytest.mark.parametrize(
    "body, message",
    [
        (b"[1, 2]", "images"),
        (b"\xff not json", "JSON"),
        (_zip({"a.json": "{}", "b.json": "{}"}), "exactly one"),
        (_zip({"../evil.json": "{}"}), "unsafe"),
    ],
)
def test_read_coco_refusals(body, message):
    with pytest.raises(FormatError, match=message):
        coco.read_coco(body)


def test_read_package_refuses_unsafe_and_unreadable_tables():
    with pytest.raises(FormatError, match="unsafe"):
        camtrapdp.read_package(_zip({"/abs/media.csv": "x", "observations.csv": "y"}))
    with pytest.raises(FormatError):
        camtrapdp.read_package(_zip({"media.csv": b"\xff\xfe\x00", "observations.csv": b"a\n"}))


def test_write_package_unreviewed_item_is_unclassified():
    rows = [
        {"image": "a.jpg", "width": 10, "height": 10, "review_status": "pending", "annotations": []},
        {"image": "b.jpg", "width": 10, "height": 10, "review_status": "pending", "annotations": [],
         "blank_if_empty": True},
    ]
    with zipfile.ZipFile(io.BytesIO(camtrapdp.write_package(rows, "classification", "t"))) as zf:
        text = zf.read("observations.csv").decode()
    assert "unclassified" in text and "blank" in text
