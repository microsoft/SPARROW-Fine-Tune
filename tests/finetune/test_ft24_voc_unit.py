"""FT-24: Pascal VOC writer/reader (``sparrow_finetune.voc``) without the API.

Writer parity is checked against the workbench writer ``sparrow_yolo.xml_utils.write_voc_xml``
semantics (integer clamped corners, swapped corners ordered, degenerate boxes dropped,
``verified="yes"`` on reviewed images). The reader is fed hostile input: DTD/entity payloads,
UTF-16 hiding a DOCTYPE, traversal member names, oversize members and archives.
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
import zipfile

import pytest

from sparrow_finetune import voc


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_writer_clamps_orders_and_drops_like_the_workbench():
    xml = voc.voc_xml(
        "a.png",
        64,
        48,
        [
            {"label": "cat", "bbox": [10.4, 5.6, 20.0, 10.0]},
            {"label": "dog", "bbox": [60, 40, 10, 10]},  # runs off the image: clamped
            {
                "label": "neg",
                "bbox": [30, 30, -10, -10],
            },  # negative size: corners swapped
            {"label": "flat", "bbox": [5, 5, 0.2, 10]},  # rounds to zero width: dropped
            {"label": " ", "bbox": [1, 1, 5, 5]},  # empty name: dropped
            {"label": "nobox", "bbox": None},
        ],
        reviewed=True,
    )
    root = ET.fromstring(xml)
    assert root.get("verified") == "yes"
    assert root.findtext("filename") == "a.png"
    assert [root.findtext(f"size/{t}") for t in ("width", "height", "depth")] == [
        "64",
        "48",
        "3",
    ]
    got = [
        (
            o.findtext("name"),
            [int(o.findtext(f"bndbox/{t}")) for t in ("xmin", "ymin", "xmax", "ymax")],
        )
        for o in root.findall("object")
    ]
    assert got == [
        ("cat", [10, 6, 30, 16]),
        ("dog", [60, 40, 64, 48]),
        ("neg", [20, 20, 30, 30]),
    ]


def test_unreviewed_and_blank_files():
    root = ET.fromstring(voc.voc_xml("b.png", 10, 10, []))
    assert root.get("verified") is None and root.findall("object") == []


def test_zip_member_names_follow_stem_and_disambiguate():
    rows = [
        {"image": n, "width": 8, "height": 8, "annotations": [], "reviewed": False}
        for n in ("x.png", "y.jpg", "y.png")
    ]
    with zipfile.ZipFile(io.BytesIO(voc.voc_zip(rows))) as zf:
        assert sorted(zf.namelist()) == [
            "annotations/x.xml",
            "annotations/y.jpg.xml",
            "annotations/y.png.xml",
        ]


def test_round_trip_through_reader():
    data = voc.voc_zip(
        [
            {
                "image": "a.png",
                "width": 64,
                "height": 48,
                "reviewed": False,
                "annotations": [{"label": "cat", "bbox": [4, 4, 20, 20]}],
            }
        ]
    )
    files, skipped = voc.read_voc_zip(data)
    assert skipped == []
    (f,) = files
    assert (f.member, f.filename, f.width, f.height) == (
        "annotations/a.xml",
        "a.png",
        64,
        48,
    )
    assert [(o.label, o.corners) for o in f.objects] == [
        ("cat", (4.0, 4.0, 24.0, 24.0))
    ]
    assert voc.to_bbox(f.objects[0].corners, 64, 48) == [4.0, 4.0, 20.0, 20.0]


def test_to_bbox_clamps_and_rejects_empty():
    assert voc.to_bbox((-5, -5, 10, 100), 64, 48) == [0.0, 0.0, 10.0, 48.0]
    assert voc.to_bbox((30, 10, 20, 5), 64, 48) == [20.0, 5.0, 10.0, 5.0]
    assert voc.to_bbox((70, 10, 80, 20), 64, 48) is None


@pytest.mark.parametrize(
    "payload",
    [
        b'<?xml version="1.0"?><!DOCTYPE a [<!ENTITY x "y">]><annotation>&x;</annotation>',
        b'<!doctype annotation SYSTEM "file:///etc/passwd"><annotation/>',
        b'<?xml version="1.0"?>\n<! ENTITY x "y"><annotation/>',
        '<?xml version="1.0" encoding="utf-16"?><!DOCTYPE a [<!ENTITY x "y">]><annotation/>'.encode(
            "utf-16"
        ),
    ],
)
def test_reader_refuses_dtd_and_entities(payload):
    with pytest.raises(voc.VocError):
        voc.parse_voc(payload)


def test_reader_reports_bad_objects_and_wrong_root():
    with pytest.raises(voc.VocError, match="not <annotation>"):
        voc.parse_voc(b"<root/>")
    with pytest.raises(voc.VocError, match="malformed"):
        voc.parse_voc(b"<annotation>")
    f = voc.parse_voc(
        b"<annotation><path>C:\\imgs\\p.jpg</path><size><width>x</width></size>"
        b"<object><name>cat</name></object>"
        b"<object><bndbox><xmin>1</xmin></bndbox></object>"
        b"<object><name>dog</name><bndbox><xmin>1</xmin><ymin>a</ymin><xmax>3</xmax><ymax>4</ymax></bndbox></object>"
        b"<object><name>ok</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>3</xmax><ymax>4</ymax></bndbox></object>"
        b"</annotation>"
    )
    assert (f.filename, f.width, f.height) == ("p.jpg", 0, 0)
    assert [o.label for o in f.objects] == ["ok"]
    assert len(f.skipped) == 3


def test_zip_member_safety():
    good = voc.voc_xml("a.png", 8, 8, [])
    data = _zip(
        {
            "../evil.xml": good,
            "/abs.xml": good,
            "C:/win.xml": good,
            "dir/..\\up.xml": good,
            "__MACOSX/._a.xml": b"junk",
            "notes.txt": b"hi",
            "bad.xml": b"<!DOCTYPE x><annotation/>",
            "ok/a.xml": good,
        }
    )
    files, skipped = voc.read_voc_zip(data)
    assert [f.member for f in files] == ["ok/a.xml"]
    reasons = {s["file"]: s["reason"] for s in skipped}
    assert (
        "traversal" in reasons["../evil.xml"]
        and "traversal" in reasons["dir/..\\up.xml"]
    )
    assert "absolute" in reasons["/abs.xml"] and "absolute" in reasons["C:/win.xml"]
    assert reasons["notes.txt"] == "not an .xml file"
    assert "DOCTYPE" in reasons["bad.xml"]
    assert "__MACOSX/._a.xml" not in reasons


def test_zip_limits(monkeypatch):
    with pytest.raises(voc.VocError, match="not a zip"):
        voc.read_voc_zip(b"plainly not a zip")

    # A highly compressible member past the per-file cap is refused on the bytes read.
    big = b"<annotation>" + b" " * (voc.MAX_MEMBER_BYTES + 10) + b"</annotation>"
    files, skipped = voc.read_voc_zip(_zip({"big.xml": big}))
    assert files == [] and "larger than" in skipped[0]["reason"]

    monkeypatch.setattr(voc, "MAX_MEMBERS", 3)
    with pytest.raises(voc.VocError, match="members"):
        voc.read_voc_zip(_zip({f"{i}.xml": b"<annotation/>" for i in range(4)}))

    monkeypatch.setattr(voc, "MAX_MEMBERS", 100)
    monkeypatch.setattr(voc, "MAX_TOTAL_BYTES", 100)
    doc = b"<annotation>" + b" " * 60 + b"</annotation>"
    with pytest.raises(voc.VocError, match="expands"):
        voc.read_voc_zip(_zip({"a.xml": doc, "b.xml": doc}))
