"""Padded JPEGs must reach the trainer.

Ultralytics checks a JPEG's last two bytes and, if they are not the end-of-image
marker, rewrites the file in place to repair it. Camera-trap firmware pads past
that marker routinely. When the source is a read-only mount the rewrite fails
with EROFS and Ultralytics drops the image from training — silently.

On this project's dataset that was 9,349 of 47,829 training images: a fifth of
the data gone, with nothing on screen to say so. These tests pin the behaviour
that prevents it.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pytest

from sparrow_yolo.dataset import (copy_truncated_at_eoi, needs_trailer_repair,
                                  prepare_yolo_dataset)

EOI = b"\xff\xd9"


def jpeg(path: Path, *, padding: int = 0, truncated: bool = False,
         size=(160, 120)) -> Path:
    """A real JPEG, optionally with a trailing block or with its tail cut off.

    Real rather than synthetic because the repair decodes the candidate before
    accepting an early cut, and a hand-rolled byte string fails that check for
    the wrong reason.
    """
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, (40, 90, 130)).save(buf, "JPEG", quality=90)
    data = buf.getvalue()
    if truncated:
        data = data[:len(data) // 2]          # no marker anywhere
    else:
        data = data + b"\x00" * padding
    path.write_bytes(data)
    return path


def jpeg_with_embedded_thumbnail(path: Path) -> Path:
    """A truncated JPEG whose only end-of-image marker belongs to a thumbnail.

    This is the shape that makes searching the whole file risky: an APP1 segment
    carries a complete little JPEG, so the last marker in the file sits near the
    start. Cutting there would produce a small, valid, wrong image.
    """
    from PIL import Image
    thumb = io.BytesIO()
    Image.new("RGB", (16, 16), (200, 10, 10)).save(thumb, "JPEG")
    payload = thumb.getvalue()

    main = io.BytesIO()
    Image.new("RGB", (800, 600), (30, 90, 120)).save(main, "JPEG", quality=95)
    body = main.getvalue()

    segment = b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload
    # SOI, then the thumbnail-bearing segment, then the main image's own
    # header and scan data with the final marker cut away.
    path.write_bytes(body[:2] + segment + body[2:len(body) // 2])
    return path


class TestDetection:
    def test_a_clean_jpeg_needs_nothing(self, tmp_path):
        assert needs_trailer_repair(jpeg(tmp_path / "a.jpg")) is False

    def test_a_padded_jpeg_is_flagged(self, tmp_path):
        """The exact case Ultralytics would try to rewrite."""
        assert needs_trailer_repair(jpeg(tmp_path / "a.jpg", padding=64)) is True

    def test_a_single_pad_byte_is_enough_to_trip_it(self, tmp_path):
        assert needs_trailer_repair(jpeg(tmp_path / "a.jpg", padding=1)) is True

    @pytest.mark.parametrize("name", ["a.JPG", "a.jpeg", "a.JPEG", "a.jpe"])
    def test_the_check_is_case_and_extension_tolerant(self, tmp_path, name):
        """Half this dataset's files are .JPG, not .jpg."""
        assert needs_trailer_repair(jpeg(tmp_path / name, padding=8)) is True

    def test_non_jpeg_formats_are_left_alone(self, tmp_path):
        p = tmp_path / "a.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40)
        assert needs_trailer_repair(p) is False

    def test_a_missing_file_is_not_our_problem(self, tmp_path):
        assert needs_trailer_repair(tmp_path / "nope.jpg") is False


class TestRepair:
    def test_padding_is_removed_and_pixels_are_untouched(self, tmp_path):
        """Byte-exact, not a re-encode: the image data is already valid."""
        src = jpeg(tmp_path / "src.jpg", padding=100)
        dst = tmp_path / "dst.jpg"
        assert copy_truncated_at_eoi(src, dst) is True

        original = src.read_bytes()
        assert dst.read_bytes() == original[:original.rfind(EOI) + 2]
        assert dst.read_bytes().endswith(EOI)
        assert not needs_trailer_repair(dst)

    def test_the_source_is_never_modified(self, tmp_path):
        """The whole reason this happens in prepare and not in the trainer."""
        src = jpeg(tmp_path / "src.jpg", padding=100)
        before = src.read_bytes()
        copy_truncated_at_eoi(src, tmp_path / "dst.jpg")
        assert src.read_bytes() == before

    def test_a_genuinely_truncated_file_is_refused(self, tmp_path):
        """No marker anywhere means the file really is cut short. Say so rather
        than writing a fabricated one."""
        src = jpeg(tmp_path / "src.jpg", truncated=True)
        assert copy_truncated_at_eoi(src, tmp_path / "dst.jpg") is False

    @pytest.mark.parametrize("padding", [40_000, 71_089, 500_000])
    def test_it_finds_a_marker_behind_any_amount_of_padding(self, tmp_path, padding):
        """A 64KB window left 2,702 real images unrepaired: this dataset's
        trailing block measured ~71KB. The whole file is searched now."""
        src = jpeg(tmp_path / "src.jpg", padding=padding)
        dst = tmp_path / "dst.jpg"
        assert copy_truncated_at_eoi(src, dst) is True
        assert dst.read_bytes().endswith(EOI)
        assert not needs_trailer_repair(dst)

    def test_the_repaired_image_still_decodes_to_the_same_picture(self, tmp_path):
        from PIL import Image
        src = jpeg(tmp_path / "src.jpg", padding=71_089, size=(320, 240))
        dst = tmp_path / "dst.jpg"
        copy_truncated_at_eoi(src, dst)
        with Image.open(dst) as im:
            im.load()
            assert im.size == (320, 240)

    def test_a_thumbnail_marker_is_not_mistaken_for_the_image(self, tmp_path):
        """The risk created by searching the whole file: the last marker can
        belong to an embedded thumbnail rather than to the picture."""
        broken = jpeg_with_embedded_thumbnail(tmp_path / "broken.jpg")
        assert needs_trailer_repair(broken) is True    # it does look repairable
        assert copy_truncated_at_eoi(broken, tmp_path / "out.jpg") is False
        assert not (tmp_path / "out.jpg").exists()


@pytest.mark.usefixtures("tmp_path")
class TestPrepareIntegration:
    """The behaviour that actually protects a run."""

    def _dataset(self, tmp_path, n_padded=3, n_clean=2):
        from PIL import Image
        root = tmp_path / "src"
        root.mkdir()
        names = []
        for i in range(n_clean + n_padded):
            p = root / f"img{i}.jpg"
            Image.new("RGB", (64, 48), (i * 10 % 255, 80, 90)).save(p, "JPEG")
            if i >= n_clean:                       # pad the tail of the set
                with open(p, "ab") as fh:
                    fh.write(b"\x00" * 128)
            (root / f"img{i}.xml").write_text(
                "<annotation><size><width>64</width><height>48</height></size>"
                "<object><name>oryx</name><bndbox><xmin>5</xmin><ymin>5</ymin>"
                "<xmax>40</xmax><ymax>30</ymax></bndbox></object></annotation>")
            names.append(p)
        return root, names

    def test_padded_images_become_real_files_not_links(self, tmp_path):
        from sparrow_yolo.dataset import audit_dataset, split_images

        root, _ = self._dataset(tmp_path)
        audit = audit_dataset(root)
        split = split_images(audit.images, audit.boxes, train=1.0, val=0.0,
                             test=0.0, stratify="none")
        data_yaml = prepare_yolo_dataset(
            audit, tmp_path / "out", "proj", {"oryx": 0}, split,
            overwrite=True, link_mode="symlink", progress=False)

        out = data_yaml.parent
        images = sorted((out / "images" / "train").glob("*.jpg"))
        assert images, "nothing was written"

        for path in images:
            # Whether repaired or linked, what the trainer opens must already
            # satisfy the check that would otherwise make it rewrite the file.
            assert not needs_trailer_repair(path), path.name

        repaired = [p for p in images if not p.is_symlink()]
        assert repaired, "the padded images should have been copied, not linked"
        for path in repaired:
            assert path.read_bytes().endswith(EOI)

    def test_a_read_only_source_still_yields_a_usable_tree(self, tmp_path):
        """The failure mode in production: the source cannot be written to."""
        from sparrow_yolo.dataset import audit_dataset, split_images

        root, files = self._dataset(tmp_path)
        audit = audit_dataset(root)
        split = split_images(audit.images, audit.boxes, train=1.0, val=0.0,
                             test=0.0, stratify="none")
        for f in files:
            os.chmod(f, 0o444)
        try:
            data_yaml = prepare_yolo_dataset(
                audit, tmp_path / "out", "proj", {"oryx": 0}, split,
                overwrite=True, link_mode="symlink", progress=False)
            out = data_yaml.parent / "images" / "train"
            written = sorted(out.glob("*.jpg"))
            assert len(written) == len(files)
            assert not any(needs_trailer_repair(p) for p in written)
        finally:
            for f in files:
                os.chmod(f, 0o644)
