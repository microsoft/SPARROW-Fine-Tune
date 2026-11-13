"""Writing Pascal VOC.

These files leave the workbench and become someone's training data, so the
writer has to be strict about two things: a box it emits must survive the
pipeline that reads it back, and the record of where a box came from must
survive the round trip.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sparrow_yolo.xml_utils import (parse_voc_xml, read_voc_boxes, to_yolo_line,
                                    write_voc_xml)


def box(**kw):
    b = {"label": "oryx", "xmin": 10, "ymin": 20, "xmax": 110, "ymax": 140,
         "score": 0.91, "source": "model"}
    b.update(kw)
    return b


class TestRoundTrip:
    def test_what_we_write_the_pipeline_can_read(self, tmp_path):
        out = write_voc_xml(tmp_path / "a.xml", tmp_path / "a.jpg", 640, 480, [box()])
        parsed = parse_voc_xml(out)
        assert len(parsed) == 1
        b = parsed[0]
        assert b.class_name == "oryx"
        assert (b.xmin, b.ymin, b.xmax, b.ymax) == (10, 20, 110, 140)
        assert (b.width, b.height) == (640, 480)
        assert b.image == "a.jpg"

    def test_every_written_box_converts_to_a_yolo_label(self, tmp_path):
        """The writer's contract: nothing it emits is rejected downstream."""
        out = write_voc_xml(tmp_path / "a.xml", tmp_path / "a.jpg", 640, 480,
                            [box(), box(xmin=600, xmax=700, ymin=400, ymax=500)])
        for b in parse_voc_xml(out):
            assert to_yolo_line(0, b.xmin, b.ymin, b.xmax, b.ymax, 640, 480) is not None

    def test_provenance_survives(self, tmp_path):
        out = write_voc_xml(
            tmp_path / "a.xml", tmp_path / "a.jpg", 640, 480,
            [box(source="human", score=1.0), box(label="zebra", source="model")],
            reviewed=True, predicted_by="sub_saharan v1")
        boxes = read_voc_boxes(out)
        assert [b["source"] for b in boxes] == ["human", "model"]
        assert boxes[1]["score"] == pytest.approx(0.91)
        text = out.read_text(encoding="utf-8")
        assert 'verified="yes"' in text
        assert "sub_saharan v1" in text

    def test_an_unknown_source_reads_back_as_model(self, tmp_path):
        """Never silently claim a person stood behind a box."""
        out = write_voc_xml(tmp_path / "a.xml", tmp_path / "a.jpg", 640, 480,
                            [box(source="magic")])
        assert read_voc_boxes(out)[0]["source"] == "model"


class TestClamping:
    def test_coordinates_are_clamped_to_the_image(self, tmp_path):
        out = write_voc_xml(tmp_path / "a.xml", tmp_path / "a.jpg", 100, 80,
                            [box(xmin=-30, ymin=-10, xmax=500, ymax=900)])
        b = parse_voc_xml(out)[0]
        assert (b.xmin, b.ymin, b.xmax, b.ymax) == (0, 0, 100, 80)

    def test_reversed_corners_are_normalised(self, tmp_path):
        """The editor lets you drag a box inside out."""
        out = write_voc_xml(tmp_path / "a.xml", tmp_path / "a.jpg", 640, 480,
                            [box(xmin=200, xmax=50, ymin=300, ymax=100)])
        b = parse_voc_xml(out)[0]
        assert b.xmin < b.xmax and b.ymin < b.ymax

    @pytest.mark.parametrize("bad", [
        {"xmin": 50, "xmax": 50},                  # zero width
        {"ymin": 20, "ymax": 20},                  # zero height
        {"xmin": 700, "xmax": 900},                # entirely outside
        {"label": ""},                             # unnamed
        {"label": "   "},
    ])
    def test_unusable_boxes_are_dropped_not_written(self, tmp_path, bad):
        out = write_voc_xml(tmp_path / "a.xml", tmp_path / "a.jpg", 640, 480,
                            [box(**bad)])
        assert parse_voc_xml(out) == []


class TestBlankFrames:
    def test_no_boxes_writes_a_valid_object_free_file(self, tmp_path):
        """A reviewed blank frame is a negative, and negatives are the single
        biggest quality lever in this pipeline. It must not be an empty file."""
        out = write_voc_xml(tmp_path / "a.xml", tmp_path / "a.jpg", 640, 480, [],
                            reviewed=True)
        assert out.exists()
        assert parse_voc_xml(out) == []
        assert "<size>" in out.read_text(encoding="utf-8")


class TestFileHandling:
    def test_it_creates_the_parent_directory(self, tmp_path):
        out = write_voc_xml(tmp_path / "deep" / "er" / "a.xml", tmp_path / "a.jpg",
                            640, 480, [box()])
        assert out.exists()

    def test_missing_file_reads_as_no_boxes(self, tmp_path):
        assert read_voc_boxes(tmp_path / "nope.xml") == []

    def test_not_reviewed_omits_the_verified_flag(self, tmp_path):
        out = write_voc_xml(tmp_path / "a.xml", tmp_path / "a.jpg", 640, 480, [box()])
        assert "verified" not in out.read_text(encoding="utf-8")
