"""VOC parsing and YOLO coordinate conversion."""

from __future__ import annotations

import pytest

from sparrow_yolo.xml_utils import Box, parse_voc_xml, to_yolo_line

from .conftest import write_xml


class TestToYoloLine:
    def test_centre_box(self):
        # 50x50 box at (25,25)-(75,75) in a 100x100 image -> centred, half size.
        assert to_yolo_line(0, 25, 25, 75, 75, 100, 100) == "0 0.500000 0.500000 0.500000 0.500000"

    def test_full_image_box(self):
        assert to_yolo_line(3, 0, 0, 200, 100, 200, 100) == "3 0.500000 0.500000 1.000000 1.000000"

    def test_clips_to_image_bounds(self):
        # Overhanging box is clipped, never allowed to produce coords > 1.
        line = to_yolo_line(0, -50, -50, 150, 150, 100, 100)
        cls, xc, yc, w, h = line.split()
        assert (float(xc), float(yc), float(w), float(h)) == (0.5, 0.5, 1.0, 1.0)

    @pytest.mark.parametrize(
        "args",
        [
            (0, 10, 10, 10, 50, 100, 100),   # zero width
            (0, 10, 10, 50, 10, 100, 100),   # zero height
            (0, 60, 10, 40, 50, 100, 100),   # inverted x
            (0, 10, 10, 50, 50, 0, 100),     # image width 0
            (0, 10, 10, 50, 50, 100, 0),     # image height 0
            (0, 200, 10, 300, 50, 100, 100), # entirely outside
        ],
    )
    def test_degenerate_returns_none(self, args):
        assert to_yolo_line(*args) is None

    def test_never_emits_out_of_range_coordinates(self):
        for line in (
            to_yolo_line(0, 0, 0, 100, 100, 100, 100),
            to_yolo_line(0, -1e6, -1e6, 1e6, 1e6, 640, 480),
        ):
            assert line is not None
            for v in (float(x) for x in line.split()[1:]):
                assert 0.0 <= v <= 1.0


class TestBox:
    def test_uses_override_dimensions(self):
        """The image's real size must win over the XML's claim.

        This is the whole point of passing dimensions in: normalising a box
        against a stale XML <size> silently rescales it.
        """
        b = Box("i.jpg", 400, 200, "lion", 0, 0, 200, 100, "i.xml")
        assert b.to_yolo(0) == "0 0.250000 0.250000 0.500000 0.500000"
        assert b.to_yolo(0, 200, 100) == "0 0.500000 0.500000 1.000000 1.000000"

    def test_is_valid_respects_override(self):
        b = Box("i.jpg", 200, 100, "lion", 150, 10, 190, 90, "i.xml")
        assert b.is_valid()
        # Against a 100px-wide image the box starts beyond the right edge.
        assert not b.is_valid(100, 100)

    def test_clipped_is_idempotent(self):
        b = Box("i.jpg", 100, 100, "x", -10, -10, 200, 200, "i.xml")
        once = b.clipped()
        assert (once.xmin, once.ymin, once.xmax, once.ymax) == (0, 0, 100, 100)
        assert once.clipped() == once


class TestParseVocXml:
    def test_reads_objects_and_size(self, tmp_path):
        p = write_xml(tmp_path / "a.xml", filename="a.jpg", width=640, height=480,
                      objects=[("lion", 1, 2, 3, 4), ("Elephant", 5, 6, 7, 8)])
        boxes = parse_voc_xml(p)
        assert [b.class_name for b in boxes] == ["lion", "Elephant"]
        assert (boxes[0].width, boxes[0].height) == (640, 480)
        assert (boxes[0].xmin, boxes[0].ymin, boxes[0].xmax, boxes[0].ymax) == (1, 2, 3, 4)

    def test_strips_windows_path_from_filename(self, tmp_path):
        """<filename> is often a stale absolute path from the tagging machine."""
        p = write_xml(tmp_path / "a.xml", filename=r"Z:\Tagged\Sub_Saharan\0001.jpg",
                      width=10, height=10, objects=[("lion", 1, 1, 5, 5)])
        assert parse_voc_xml(p)[0].image == "0001.jpg"

    def test_missing_size_yields_zero(self, tmp_path):
        (tmp_path / "a.xml").write_text(
            "<annotation><filename>a.jpg</filename>"
            "<object><name>lion</name>"
            "<bndbox><xmin>1</xmin><ymin>1</ymin><xmax>5</xmax><ymax>5</ymax></bndbox>"
            "</object></annotation>",
            encoding="utf-8",
        )
        b = parse_voc_xml(tmp_path / "a.xml")[0]
        assert (b.width, b.height) == (0, 0)
        assert not b.is_valid()  # unusable without a real size

    def test_float_and_whitespace_coordinates(self, tmp_path):
        (tmp_path / "a.xml").write_text(
            "<annotation><filename>a.jpg</filename>"
            "<size><width>100.0</width><height>50.0</height></size>"
            "<object><name>  lion  </name>"
            "<bndbox><xmin> 1.5 </xmin><ymin>2.5</ymin><xmax>10.5</xmax><ymax>20.5</ymax></bndbox>"
            "</object></annotation>",
            encoding="utf-8",
        )
        b = parse_voc_xml(tmp_path / "a.xml")[0]
        assert b.class_name == "lion"
        assert (b.width, b.height, b.xmin) == (100, 50, 1.5)

    def test_object_without_bndbox_is_skipped(self, tmp_path):
        (tmp_path / "a.xml").write_text(
            "<annotation><size><width>10</width><height>10</height></size>"
            "<object><name>lion</name></object>"
            "<object><name>ok</name>"
            "<bndbox><xmin>1</xmin><ymin>1</ymin><xmax>5</xmax><ymax>5</ymax></bndbox>"
            "</object></annotation>",
            encoding="utf-8",
        )
        assert [b.class_name for b in parse_voc_xml(tmp_path / "a.xml")] == ["ok"]

    def test_difficult_flag(self, tmp_path):
        p = write_xml(tmp_path / "a.xml", filename="a.jpg", width=10, height=10,
                      objects=[("lion", 1, 1, 5, 5)], difficult=1)
        assert parse_voc_xml(p)[0].difficult is True

    def test_malformed_raises(self, tmp_path):
        (tmp_path / "bad.xml").write_text("<annotation><size>", encoding="utf-8")
        with pytest.raises(Exception):
            parse_voc_xml(tmp_path / "bad.xml")
