"""Synthetic VOC datasets for the test suite.

Everything here is deliberately small and deterministic so the whole suite runs
in a second without a GPU or a real dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import pytest
from PIL import Image

VOC_TEMPLATE = """<annotation>
  <filename>{filename}</filename>
  <size>
    <width>{width}</width>
    <height>{height}</height>
    <depth>3</depth>
  </size>
{objects}</annotation>
"""

OBJECT_TEMPLATE = """  <object>
    <name>{name}</name>
    <difficult>{difficult}</difficult>
    <bndbox>
      <xmin>{xmin}</xmin>
      <ymin>{ymin}</ymin>
      <xmax>{xmax}</xmax>
      <ymax>{ymax}</ymax>
    </bndbox>
  </object>
"""


def write_image(path: Path, size: Tuple[int, int] = (200, 100), colour: str = "green") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path)
    return path


def write_xml(
    path: Path,
    *,
    filename: str,
    width: int,
    height: int,
    objects: Iterable[Tuple[str, float, float, float, float]] = (),
    difficult: int = 0,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(
        OBJECT_TEMPLATE.format(name=n, xmin=a, ymin=b, xmax=c, ymax=d, difficult=difficult)
        for n, a, b, c, d in objects
    )
    path.write_text(
        VOC_TEMPLATE.format(filename=filename, width=width, height=height, objects=body),
        encoding="utf-8",
    )
    return path


def write_pair(
    root: Path,
    stem: str,
    *,
    size: Tuple[int, int] = (200, 100),
    objects: Sequence[Tuple[str, float, float, float, float]] = (),
    xml_size: Optional[Tuple[int, int]] = None,
    subdir: str = "",
) -> Tuple[Path, Path]:
    """Write a matching image/XML pair. ``xml_size`` forces a size disagreement."""
    base = root / subdir if subdir else root
    img = write_image(base / f"{stem}.jpg", size=size)
    w, h = xml_size or size
    xml = write_xml(base / f"{stem}.xml", filename=f"{stem}.jpg", width=w, height=h, objects=objects)
    return img, xml


@pytest.fixture
def voc_root(tmp_path: Path) -> Path:
    """A small dataset exercising every branch of the audit.

    Contents:
      * ``a``-``f``  - normal, 200x100, one or two boxes
      * ``bg``       - readable image with no XML  (background candidate)
      * ``empty``    - XML with no <object>        (background candidate)
      * ``mismatch`` - XML claims 400x200, image is 200x100
      * ``broken``   - XML is not well-formed
      * ``noimage``  - XML with no corresponding image
      * ``sub/a``    - same stem as top-level ``a``, in a subfolder
    """
    root = tmp_path / "voc"
    root.mkdir()

    # Enough images per class that stratified splitting has something to work with.
    for i, stem in enumerate("abcdef"):
        write_pair(
            root,
            stem,
            objects=[("Elephant", 10, 10, 60, 60)] + ([("lion", 100, 20, 150, 80)] if i % 2 else []),
        )
    for i in range(6):
        write_pair(root, f"g{i}", objects=[("lion", 20, 20, 90, 90)])

    write_image(root / "bg.jpg")
    write_xml(root / "empty.xml", filename="empty.jpg", width=200, height=100, objects=[])
    write_image(root / "empty.jpg")

    # The XML dimensions are wrong; normalising against them would scale the box
    # by 2x in each axis.
    write_pair(root, "mismatch", size=(200, 100), xml_size=(400, 200),
               objects=[("Elephant", 10, 10, 60, 60)])

    write_image(root / "broken.jpg")
    (root / "broken.xml").write_text("<annotation><size><width>200", encoding="utf-8")

    write_xml(root / "noimage.xml", filename="noimage.jpg", width=200, height=100,
              objects=[("ghost", 1, 1, 10, 10)])

    # Same stem as the top-level 'a'; flattening naively would collide.
    write_pair(root, "a", subdir="sub", objects=[("lion", 5, 5, 50, 50)])

    return root


@pytest.fixture
def simple_root(tmp_path: Path) -> Path:
    """A clean dataset: 12 images, two classes, no issues."""
    root = tmp_path / "simple"
    root.mkdir()
    for i in range(12):
        write_pair(
            root,
            f"img{i:02d}",
            objects=[("elephant", 10, 10, 60, 60)] if i % 2 == 0 else [("lion", 20, 20, 90, 90)],
        )
    return root
