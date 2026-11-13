"""Audit, class map, split, and prepare.

These cover the bugs that silently corrupted training data in the previous
version, so each test names the failure it guards against.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from sparrow_yolo.dataset import (
    audit_dataset,
    flat_stem,
    load_or_create_class_map,
    load_pinned_splits,
    make_class_map,
    prepare_yolo_dataset,
    split_images,
    verify_prepared,
)

from .conftest import write_image, write_pair


class TestAudit:
    def test_finds_images_and_boxes(self, voc_root):
        r = audit_dataset(voc_root, progress=False)
        assert r.summary["num_boxes"] > 0
        assert r.summary["num_images"] == len(list(voc_root.rglob("*.jpg")))

    def test_images_frame_is_returned(self, voc_root):
        """The per-image frame used to be built and then thrown away.

        Every image was opened and measured for nothing, and the real dimensions
        were never available to compare against the XML.
        """
        r = audit_dataset(voc_root, progress=False)
        assert not r.images.empty
        assert {"img_width", "img_height", "readable", "n_boxes"} <= set(r.images.columns)
        normal = r.images[r.images["filename"] == "a.jpg"].iloc[0]
        assert (normal["img_width"], normal["img_height"]) == (200, 100)

    def test_real_image_size_overrides_xml(self, voc_root):
        """'mismatch' claims 400x200 in XML but is really 200x100.

        Normalising against the XML would halve every normalised coordinate.
        """
        r = audit_dataset(voc_root, progress=False)
        box = r.boxes[r.boxes["filename"] == "mismatch.jpg"].iloc[0]
        assert (box["img_width"], box["img_height"]) == (200, 100)
        assert (box["xml_width"], box["xml_height"]) == (400, 200)
        assert "size_mismatch" in set(r.issues["type"])

    def test_class_names_normalised(self, voc_root):
        r = audit_dataset(voc_root, normalise_class_names=True, progress=False)
        assert "elephant" in set(r.boxes["class_name"])
        assert "Elephant" not in set(r.boxes["class_name"])

    def test_class_names_preserved_when_disabled(self, voc_root):
        r = audit_dataset(voc_root, normalise_class_names=False, progress=False)
        assert "Elephant" in set(r.boxes["class_name"])

    def test_merge_classes_folds_typo_into_one_label(self, tmp_path, capsys):
        """Taxonomy typos split one species across two classes.

        Real case: 'hylochoerusmeinertzhageni' (1375 boxes) and
        'hylochoerusmeinhertzegeni' (655) were the same animal. Left separate,
        the model is guaranteed to confuse them.
        """
        root = tmp_path / "d"
        for i in range(6):
            write_pair(root, f"good{i}", objects=[("Hylochoerusmeinertzhageni", 10, 10, 60, 60)])
        for i in range(3):
            write_pair(root, f"typo{i}", objects=[("hylochoerusmeinhertzegeni", 10, 10, 60, 60)])

        r = audit_dataset(root, merge_classes={"hylochoerusmeinhertzegeni": "hylochoerusmeinertzhageni"},
                          progress=False)
        assert set(r.boxes["class_name"]) == {"hylochoerusmeinertzhageni"}
        assert len(r.boxes) == 9
        # Returned structurally so a UI can show it, and printed for the CLI.
        assert any("Merged 3 box(es)" in w for w in r.warnings)
        assert "Merged 3 box(es)" in capsys.readouterr().out

    def test_merge_classes_warns_when_unused(self, simple_root):
        r = audit_dataset(simple_root, merge_classes={"nosuchclass": "lion"}, progress=False)
        assert any("matched nothing" in w for w in r.warnings)

    def test_merge_classes_absent_by_default(self, tmp_path):
        root = tmp_path / "d"
        for i in range(4):
            write_pair(root, f"a{i}", objects=[("lion", 10, 10, 60, 60)])
        for i in range(2):
            write_pair(root, f"b{i}", objects=[("lionn", 10, 10, 60, 60)])
        r = audit_dataset(root, progress=False)
        assert {"lion", "lionn"} <= set(r.boxes["class_name"])

    def test_reports_issue_types(self, voc_root):
        r = audit_dataset(voc_root, progress=False)
        types = set(r.issues["type"])
        assert "corrupt_xml" in types      # broken.xml
        assert "empty_xml" in types        # empty.xml has no <object>
        assert "missing_xml" in types      # bg.jpg has no XML
        assert "missing_image" in types    # noimage.xml references nothing

    def test_orphan_xml_contributes_no_boxes(self, voc_root):
        """A box whose image cannot be found must not reach the class map.

        Previously such rows were kept with an empty image path, so a class that
        existed only in an orphaned XML still consumed a class id and shifted
        every id after it.
        """
        r = audit_dataset(voc_root, progress=False)
        assert "ghost" not in set(r.boxes["class_name"])
        assert (r.boxes["image"] == "").sum() == 0
        assert r.boxes["image"].notna().all()

    def test_backgrounds_are_tracked(self, voc_root):
        r = audit_dataset(voc_root, progress=False)
        bg = r.images[r.images["filename"] == "bg.jpg"].iloc[0]
        assert bg["readable"] and bg["n_boxes"] == 0
        assert r.summary["background_images"] >= 2  # bg.jpg and empty.jpg

    def test_two_kinds_of_blank_are_distinguished(self, voc_root):
        """An empty XML means 'reviewed, nothing here'. A missing XML is ambiguous.

        Collapsing them risks training untagged images as negatives.
        """
        r = audit_dataset(voc_root, progress=False)
        # empty.jpg has an XML with no <object>; bg.jpg has no XML at all.
        empty = r.images.set_index("filename").loc["empty.jpg"]
        nox = r.images.set_index("filename").loc["bg.jpg"]
        assert empty["has_xml"] and empty["n_boxes"] == 0
        assert not nox["has_xml"] and nox["n_boxes"] == 0
        assert r.summary["blank_with_empty_xml"] >= 1
        assert r.summary["blank_unknown"] >= 1

    @pytest.mark.parametrize(
        "policy,expect_empty_xml,expect_no_xml",
        [("reviewed", True, False), ("all", True, True), ("none", False, False)],
    )
    def test_background_policy_selection(self, voc_root, policy, expect_empty_xml, expect_no_xml):
        r = audit_dataset(voc_root, progress=False)
        names = {Path(p).name for p in r.background_images(policy)["image"]}
        assert ("empty.jpg" in names) is expect_empty_xml
        assert ("bg.jpg" in names) is expect_no_xml

    def test_background_policy_rejects_unknown(self, voc_root):
        r = audit_dataset(voc_root, progress=False)
        with pytest.raises(ValueError, match="background_policy"):
            r.background_images("maybe")

    def test_fingerprint_off_by_default(self, simple_root):
        r = audit_dataset(simple_root, progress=False)
        assert r.images["content_fp"].isna().all()

    def test_fingerprint_detects_content_change_not_touch(self, tmp_path):
        """mtime lies after a re-copy; the content fingerprint must not.

        Re-copying a dataset rewrites every mtime, which would otherwise report
        all 60,000 images as changed.
        """
        root = tmp_path / "d"
        for i in range(4):
            write_pair(root, f"i{i}", objects=[("lion", 10, 10, 60, 60)])

        before = audit_dataset(root, fingerprint=True, progress=False)
        fp_before = dict(zip(before.images["rel_path"], before.images["content_fp"]))
        assert all(v is not None for v in fp_before.values())

        # Touch every file: mtime moves, content does not.
        for p in root.glob("*.jpg"):
            os.utime(p, (1, 1))
        touched = audit_dataset(root, fingerprint=True, progress=False)
        fp_touched = dict(zip(touched.images["rel_path"], touched.images["content_fp"]))
        assert fp_touched == fp_before, "a touch must not look like a content change"

        # Now genuinely change one image.
        target = root / "i0.jpg"
        write_image(target, size=(321, 201))
        after = audit_dataset(root, fingerprint=True, progress=False)
        fp_after = dict(zip(after.images["rel_path"], after.images["content_fp"]))
        changed = [k for k in fp_before if fp_after[k] != fp_before[k]]
        assert changed == ["i0.jpg"]

    def test_progress_callback_reports_completion(self, simple_root):
        seen = []
        audit_dataset(simple_root, on_progress=lambda d, t, p: seen.append((d, t, p)),
                      progress=False)
        assert seen, "no progress reported"
        phases = {p for _, _, p in seen}
        assert "Checking images" in phases and "Reading XML" in phases
        for phase in phases:
            last = [(d, t) for d, t, p in seen if p == phase][-1]
            assert last[0] == last[1], f"{phase} never reached 100%"

    def test_progress_callback_failure_never_breaks_audit(self, simple_root):
        def boom(*_):
            raise RuntimeError("UI went away")

        r = audit_dataset(simple_root, on_progress=boom, progress=False)
        assert not r.boxes.empty

    def test_blank_kind_column(self, voc_root):
        r = audit_dataset(voc_root, progress=False)
        # rel_path, not filename: 'a.jpg' exists twice in this fixture.
        kinds = r.images.set_index("rel_path")["blank_kind"]
        assert kinds["a.jpg"] == "labelled"
        assert kinds["sub/a.jpg"] == "labelled"
        assert kinds["empty.jpg"] == "empty_xml"
        assert kinds["bg.jpg"] == "unknown"


class TestConfirmedBlankDirs:
    """A folder the operator knows is empty, with no XML files.

    Declaring the directory records that knowledge without writing anything into
    the dataset, which matters when it is mounted read-only.
    """

    @pytest.fixture
    def root_with_blank_dir(self, tmp_path):
        root = tmp_path / "ds"
        for i in range(8):
            write_pair(root, f"tag{i}", objects=[("lion", 10, 10, 60, 60)])
        for i in range(5):
            write_image(root / "blanks" / f"b{i}.jpg")
        write_image(root / "untriaged.jpg")   # loose, undeclared
        return root

    def test_undeclared_is_unknown(self, root_with_blank_dir):
        r = audit_dataset(root_with_blank_dir, progress=False)
        kinds = r.images.set_index("filename")["blank_kind"]
        assert kinds["b0.jpg"] == "unknown"
        assert kinds["untriaged.jpg"] == "unknown"

    def test_declared_dir_becomes_confirmed(self, root_with_blank_dir):
        r = audit_dataset(root_with_blank_dir, confirmed_blank_dirs=["blanks"], progress=False)
        kinds = r.images.set_index("filename")["blank_kind"]
        assert kinds["b0.jpg"] == "confirmed_dir"
        # A loose image outside the declared folder stays ambiguous.
        assert kinds["untriaged.jpg"] == "unknown"
        assert r.summary["blank_confirmed_dir"] == 5
        assert r.summary["blank_unknown"] == 1

    def test_glob_pattern(self, tmp_path):
        root = tmp_path / "ds"
        for i in range(6):
            write_pair(root, f"tag{i}", objects=[("lion", 10, 10, 60, 60)])
        for site in ("site_a", "site_b"):
            write_image(root / site / "no_animals" / "x.jpg")
        r = audit_dataset(root, confirmed_blank_dirs=["*/no_animals"], progress=False)
        assert r.summary["blank_confirmed_dir"] == 2

    def test_confirmed_blanks_are_used_by_default_policy(self, root_with_blank_dir):
        r = audit_dataset(root_with_blank_dir, confirmed_blank_dirs=["blanks"], progress=False)
        s = split_images(r.images, r.boxes, seed=1)
        names = {Path(p).name for p in s["image"]}
        assert "b0.jpg" in names            # declared blank, trusted
        assert "untriaged.jpg" not in names  # undeclared, excluded

    def test_not_reported_as_missing_xml(self, root_with_blank_dir):
        r = audit_dataset(root_with_blank_dir, confirmed_blank_dirs=["blanks"], progress=False)
        missing = r.issues[r.issues["type"] == "missing_xml"]
        assert {Path(p).name for p in missing["path"]} == {"untriaged.jpg"}
        assert (r.issues["type"] == "confirmed_blank").sum() == 5

    def test_blanks_get_empty_labels_when_prepared(self, root_with_blank_dir, tmp_path):
        r = audit_dataset(root_with_blank_dir, confirmed_blank_dirs=["blanks"], progress=False)
        cm = make_class_map(r.boxes)
        s = split_images(r.images, r.boxes, seed=1)
        data_yaml = prepare_yolo_dataset(r, tmp_path / "prep", "proj", cm, s,
                                         overwrite=True, progress=False)
        out = data_yaml.parent
        label = next(p for split in ("train", "val", "test")
                     for p in (out / "labels" / split).glob("blanks__b0.txt"))
        assert label.read_text(encoding="utf-8") == ""

    def test_detects_truncated_image(self, tmp_path):
        """A truncated JPEG must fail the audit.

        PIL's Image.open only parses the header, so the old check passed
        truncated files straight through to the trainer.
        """
        root = tmp_path / "d"
        write_pair(root, "good", objects=[("lion", 1, 1, 50, 50)])
        img = write_image(root / "trunc.jpg", size=(300, 300))
        data = img.read_bytes()
        img.write_bytes(data[: len(data) // 2])

        r = audit_dataset(root, progress=False)
        corrupt = r.issues[r.issues["type"] == "corrupt_image"]
        assert len(corrupt) == 1
        assert "trunc.jpg" in corrupt.iloc[0]["path"]
        assert not r.images.set_index("filename").loc["trunc.jpg", "readable"]

    def test_padded_jpeg_is_not_corrupt(self, tmp_path):
        """Zero-padding after the EOI marker is legitimate and must not be rejected.

        Camera-trap firmware pads files to block boundaries, so the end-of-image
        marker is not the last bytes. Condemning on the marker alone flagged
        10,171 of 60,049 real images (17%) as corrupt - every one of which
        decoded perfectly.
        """
        root = tmp_path / "d"
        for i in range(4):
            write_pair(root, f"ok{i}", objects=[("lion", 1, 1, 50, 50)])

        padded = write_image(root / "padded.jpg", size=(320, 200))
        data = padded.read_bytes()
        assert data.rfind(b"\xff\xd9") != -1
        padded.write_bytes(data + b"\x00" * 4096)   # far beyond any small window

        r = audit_dataset(root, progress=False)
        row = r.images.set_index("filename").loc["padded.jpg"]
        assert row["readable"], "padded but valid JPEG was wrongly rejected"
        assert (row["img_width"], row["img_height"]) == (320, 200)
        assert "padded.jpg" not in set(r.issues[r.issues["type"] == "corrupt_image"]["path"].apply(
            lambda p: Path(p).name))

    def test_genuinely_truncated_still_detected(self, tmp_path):
        """The padding tolerance must not let real truncation through."""
        root = tmp_path / "d"
        for i in range(4):
            write_pair(root, f"ok{i}", objects=[("lion", 1, 1, 50, 50)])
        img = write_image(root / "cut.jpg", size=(320, 200))
        img.write_bytes(img.read_bytes()[: len(img.read_bytes()) // 3])

        r = audit_dataset(root, progress=False)
        assert not r.images.set_index("filename").loc["cut.jpg", "readable"]
        assert "cut.jpg" in {Path(p).name for p in
                             r.issues[r.issues["type"] == "corrupt_image"]["path"]}

    def test_duplicate_stems_flagged(self, voc_root):
        """'a.jpg' exists at the root and in sub/; both must survive."""
        r = audit_dataset(voc_root, progress=False)
        stems = set(r.images["flat_stem"])
        assert "a" in stems and "sub__a" in stems
        assert len(stems) == len(r.images)   # no collisions
        assert "duplicate_flat_name" not in set(r.issues["type"])

    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            audit_dataset(tmp_path / "nope", progress=False)

    def test_empty_root_raises(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError):
            audit_dataset(tmp_path / "empty", progress=False)


def test_flat_stem_disambiguates_subdirectories(tmp_path):
    root = tmp_path / "r"
    assert flat_stem(root / "0001.jpg", root) == "0001"
    assert flat_stem(root / "site_a" / "cam3" / "0001.jpg", root) == "site_a__cam3__0001"


class TestClassMap:
    def test_ids_are_contiguous_and_sorted(self, simple_root):
        r = audit_dataset(simple_root, progress=False)
        cm = make_class_map(r.boxes)
        assert cm == {"elephant": 0, "lion": 1}

    def test_key_lookup_matches_frame(self, voc_root):
        """Guards the KeyError that made normalise_case unusable.

        make_class_map built lowercased keys while the writer looked up the raw
        name, so any capitalised class crashed the prepare step.
        """
        r = audit_dataset(voc_root, normalise_class_names=True, progress=False)
        cm = make_class_map(r.boxes)
        for name in r.boxes["class_name"]:
            assert name in cm

    def test_min_boxes_filter(self, voc_root):
        r = audit_dataset(voc_root, progress=False)
        cm = make_class_map(r.boxes, min_boxes=1000)
        assert cm == {}

    def test_drop_classes(self, simple_root):
        r = audit_dataset(simple_root, progress=False)
        assert make_class_map(r.boxes, drop_classes=["lion"]) == {"elephant": 0}

    def test_pinned_map_keeps_existing_ids(self, simple_root, tmp_path):
        """Adding a class must not renumber the others.

        Ids come from sorted order, so an inserted class would otherwise shift
        every later id and invalidate existing checkpoints.
        """
        pin = tmp_path / "classes.json"
        pin.write_text(json.dumps({"lion": 0, "elephant": 1}), encoding="utf-8")

        r = audit_dataset(simple_root, progress=False)
        cm, warnings = load_or_create_class_map(pin, r.boxes)
        assert cm == {"lion": 0, "elephant": 1}
        assert warnings == []

    def test_pinned_map_appends_new_class(self, simple_root, tmp_path):
        pin = tmp_path / "classes.json"
        pin.write_text(json.dumps({"aardvark": 0}), encoding="utf-8")

        r = audit_dataset(simple_root, progress=False)
        cm, warnings = load_or_create_class_map(pin, r.boxes)
        assert cm["aardvark"] == 0
        assert sorted(cm.values()) == [0, 1, 2]
        assert any("elephant" in w for w in warnings)

    def test_reuse_disabled_rebuilds(self, simple_root, tmp_path):
        pin = tmp_path / "classes.json"
        pin.write_text(json.dumps({"aardvark": 0}), encoding="utf-8")
        r = audit_dataset(simple_root, progress=False)
        cm, _ = load_or_create_class_map(pin, r.boxes, reuse=False)
        assert "aardvark" not in cm


class TestSplit:
    def test_fractions_must_sum_to_one(self, simple_root):
        r = audit_dataset(simple_root, progress=False)
        with pytest.raises(ValueError, match="sum to 1.0"):
            split_images(r.images, r.boxes, train=0.9, val=0.2, test=0.1)

    def test_every_image_assigned_once(self, simple_root):
        r = audit_dataset(simple_root, progress=False)
        s = split_images(r.images, r.boxes, seed=1)
        assert len(s) == len(r.usable_images())
        assert s["image"].is_unique
        assert set(s["split"]) <= {"train", "val", "test"}

    def test_deterministic_for_a_seed(self, simple_root):
        r = audit_dataset(simple_root, progress=False)
        a = split_images(r.images, r.boxes, seed=7).sort_values("image").reset_index(drop=True)
        b = split_images(r.images, r.boxes, seed=7).sort_values("image").reset_index(drop=True)
        assert a.equals(b)

    def test_default_policy_keeps_reviewed_blanks_only(self, voc_root):
        """empty.jpg was reviewed and is empty; bg.jpg has no XML, so it is unknown."""
        r = audit_dataset(voc_root, progress=False)
        names = {Path(p).name for p in split_images(r.images, r.boxes, seed=3)["image"]}
        assert "empty.jpg" in names
        assert "bg.jpg" not in names

    def test_policy_all_includes_untagged(self, voc_root):
        r = audit_dataset(voc_root, progress=False)
        names = {Path(p).name for p in
                 split_images(r.images, r.boxes, seed=3, background_policy="all")["image"]}
        assert "bg.jpg" in names and "empty.jpg" in names

    def test_policy_none_excludes_all_blanks(self, voc_root):
        r = audit_dataset(voc_root, progress=False)
        names = {Path(p).name for p in
                 split_images(r.images, r.boxes, seed=3, background_policy="none")["image"]}
        assert "bg.jpg" not in names and "empty.jpg" not in names

    def test_excluded_untagged_blanks_are_announced(self, voc_root, capsys):
        r = audit_dataset(voc_root, progress=False)
        split_images(r.images, r.boxes, seed=3)
        assert "no XML" in capsys.readouterr().out

    def test_rejects_unknown_background_policy(self, simple_root):
        r = audit_dataset(simple_root, progress=False)
        with pytest.raises(ValueError, match="background_policy"):
            split_images(r.images, r.boxes, background_policy="sometimes")

    def test_unreadable_images_excluded(self, tmp_path):
        root = tmp_path / "d"
        for i in range(6):
            write_pair(root, f"i{i}", objects=[("lion", 1, 1, 50, 50)])
        bad = write_image(root / "bad.jpg")
        bad.write_bytes(b"not an image")

        r = audit_dataset(root, progress=False)
        s = split_images(r.images, r.boxes, seed=1)
        assert "bad.jpg" not in {Path(p).name for p in s["image"]}

    def test_group_by_parent_dir_prevents_leakage(self, tmp_path):
        """Burst frames in one folder must not straddle train and val.

        Near-duplicate frames on both sides of the split inflate validation
        metrics to the point of being meaningless.
        """
        root = tmp_path / "grouped"
        for site in range(8):
            for frame in range(4):
                write_pair(root, f"f{frame}", subdir=f"site{site}",
                           objects=[("lion", 10, 10, 60, 60)])

        r = audit_dataset(root, progress=False)
        s = split_images(r.images, r.boxes, seed=5, group_by="parent_dir")
        by_group = {}
        for image, split in zip(s["image"], s["split"]):
            by_group.setdefault(Path(image).parent.name, set()).add(split)
        assert all(len(v) == 1 for v in by_group.values())

    def test_warns_when_one_group_dominates(self, tmp_path, capsys):
        """A blanks folder is not a burst sequence.

        Grouped whole, it lands entirely in one split - so all the negatives end
        up on one side. The warning has to name it.
        """
        root = tmp_path / "dom"
        for site in range(12):
            for frame in range(3):
                write_pair(root, f"f{frame}", subdir=f"site{site:02d}",
                           objects=[("lion", 10, 10, 60, 60)])
        for i in range(20):
            write_image(root / "blanks" / f"b{i}.jpg")

        r = audit_dataset(root, confirmed_blank_dirs=["blanks"], progress=False)
        split_images(r.images, r.boxes, seed=1, group_by="parent_dir")
        out = capsys.readouterr().out
        assert "'blanks'" in out and "entirely in one split" in out

    def test_warns_when_too_few_groups(self, tmp_path, capsys):
        """Whole groups are indivisible, so few groups cannot hit 80/10/10."""
        root = tmp_path / "few"
        for site in range(2):
            for frame in range(10):
                write_pair(root, f"f{frame}", subdir=f"site{site}",
                           objects=[("lion", 10, 10, 60, 60)])
        r = audit_dataset(root, progress=False)
        split_images(r.images, r.boxes, seed=1, group_by="parent_dir")
        assert "only 2 group(s)" in capsys.readouterr().out

    def test_proportions_are_close_with_many_groups(self, tmp_path):
        root = tmp_path / "many"
        for site in range(40):
            for frame in range(3):
                write_pair(root, f"f{frame}", subdir=f"s{site:02d}",
                           objects=[("lion", 10, 10, 60, 60)])
        r = audit_dataset(root, progress=False)
        s = split_images(r.images, r.boxes, seed=1, group_by="parent_dir")
        counts = s["split"].value_counts(normalize=True)
        assert counts["train"] == pytest.approx(0.8, abs=0.05)
        assert counts["val"] == pytest.approx(0.1, abs=0.05)

    def test_stratified_proportions(self, tmp_path):
        root = tmp_path / "strat"
        for i in range(100):
            write_pair(root, f"i{i:03d}",
                       objects=[("lion" if i % 2 else "elephant", 10, 10, 60, 60)])
        r = audit_dataset(root, progress=False)
        s = split_images(r.images, r.boxes, seed=4)
        counts = s["split"].value_counts()
        assert counts["train"] == 80 and counts["val"] == 10 and counts["test"] == 10

    def test_rare_class_not_always_in_train(self, tmp_path):
        """Deterministic remainder rounding sent every small class to train.

        Across seeds a 3-image class must sometimes land in val or test.
        """
        root = tmp_path / "rare"
        for i in range(60):
            write_pair(root, f"common{i:02d}", objects=[("lion", 10, 10, 60, 60)])
        for i in range(3):
            write_pair(root, f"rare{i}", objects=[("pangolin", 10, 10, 60, 60)])

        r = audit_dataset(root, progress=False)
        rare = {Path(p).name for p in r.boxes[r.boxes["class_name"] == "pangolin"]["image"]}
        placements = set()
        for seed in range(25):
            s = split_images(r.images, r.boxes, seed=seed)
            placements |= set(s[s["image"].map(lambda p: Path(p).name).isin(rare)]["split"])
        assert placements != {"train"}, "rare class never left train across 25 seeds"

    def test_rejects_unknown_stratify(self, simple_root):
        r = audit_dataset(simple_root, progress=False)
        with pytest.raises(ValueError, match="stratify"):
            split_images(r.images, r.boxes, stratify="bogus")

    def test_survives_singleton_class(self, tmp_path):
        """A class with one example cannot be stratified; it must not raise."""
        root = tmp_path / "d"
        for i in range(10):
            write_pair(root, f"i{i}", objects=[("lion", 1, 1, 50, 50)])
        write_pair(root, "rare", objects=[("pangolin", 1, 1, 50, 50)])

        r = audit_dataset(root, progress=False)
        s = split_images(r.images, r.boxes, seed=2)
        assert len(s) == 11


class TestPinnedSplits:
    """Growing a dataset must not move existing images between splits.

    The v1 model was evaluated on a test set; if adding 50k images reshuffles
    everything, some of those test images become v2 training data and v2's test
    score is inflated and no longer comparable.
    """

    def _dataset(self, root, n, start=0):
        for i in range(start, start + n):
            write_pair(root, f"img{i:04d}", objects=[("lion", 10, 10, 60, 60)])
        return root

    def test_existing_images_keep_their_split(self, tmp_path):
        root = self._dataset(tmp_path / "ds", 60)
        r1 = audit_dataset(root, progress=False)
        s1 = split_images(r1.images, r1.boxes, seed=1)
        pinned = dict(zip(r1.images.set_index("image").loc[s1["image"], "rel_path"], s1["split"]))

        # Dataset grows by 40 images.
        self._dataset(root, 40, start=60)
        r2 = audit_dataset(root, progress=False)
        s2 = split_images(r2.images, r2.boxes, seed=99, pinned=pinned)

        rel = dict(zip(r2.images["image"], r2.images["rel_path"]))
        after = {rel[i]: s for i, s in zip(s2["image"], s2["split"])}
        for rel_path, split in pinned.items():
            assert after[rel_path] == split, f"{rel_path} moved from {split} to {after[rel_path]}"

    def test_without_pinning_the_split_reshuffles(self, tmp_path):
        """Demonstrates the failure the pinning prevents."""
        root = self._dataset(tmp_path / "ds", 60)
        r1 = audit_dataset(root, progress=False)
        s1 = split_images(r1.images, r1.boxes, seed=1)
        before = dict(zip(s1["image"], s1["split"]))

        self._dataset(root, 40, start=60)
        r2 = audit_dataset(root, progress=False)
        s2 = split_images(r2.images, r2.boxes, seed=1)   # no pinned=
        after = dict(zip(s2["image"], s2["split"]))

        moved = [i for i in before if i in after and after[i] != before[i]]
        assert moved, "expected an unpinned re-split to move images"

    def test_new_images_restore_the_target_ratio(self, tmp_path):
        root = self._dataset(tmp_path / "ds", 50)
        r1 = audit_dataset(root, progress=False)
        s1 = split_images(r1.images, r1.boxes, seed=1)
        pinned = dict(zip(r1.images.set_index("image").loc[s1["image"], "rel_path"], s1["split"]))

        self._dataset(root, 150, start=50)
        r2 = audit_dataset(root, progress=False)
        s2 = split_images(r2.images, r2.boxes, seed=2, pinned=pinned)

        frac = s2["split"].value_counts(normalize=True)
        assert frac["train"] == pytest.approx(0.8, abs=0.03)
        assert frac["val"] == pytest.approx(0.1, abs=0.03)

    def test_stale_pins_are_ignored(self, tmp_path):
        root = self._dataset(tmp_path / "ds", 20)
        r = audit_dataset(root, progress=False)
        pinned = {"deleted_long_ago.jpg": "test", "img0000.jpg": "test"}
        s = split_images(r.images, r.boxes, seed=1, pinned=pinned)
        assert len(s) == 20
        rel = dict(zip(r.images["image"], r.images["rel_path"]))
        assert {rel[i]: v for i, v in zip(s["image"], s["split"])}["img0000.jpg"] == "test"

    def test_prepare_writes_splits_json(self, tmp_path):
        root = self._dataset(tmp_path / "ds", 30)
        r = audit_dataset(root, progress=False)
        cm = make_class_map(r.boxes)
        s = split_images(r.images, r.boxes, seed=1)
        data_yaml = prepare_yolo_dataset(r, tmp_path / "prep", "proj", cm, s,
                                         overwrite=True, progress=False)
        pinned = load_pinned_splits(data_yaml.parent / "splits.json")
        assert len(pinned) == 30
        assert set(pinned.values()) <= {"train", "val", "test"}

    def test_round_trip_through_disk_is_stable(self, tmp_path):
        root = self._dataset(tmp_path / "ds", 40)
        r = audit_dataset(root, progress=False)
        cm = make_class_map(r.boxes)
        s = split_images(r.images, r.boxes, seed=1)
        data_yaml = prepare_yolo_dataset(r, tmp_path / "prep", "proj", cm, s,
                                         overwrite=True, progress=False)
        pinned = load_pinned_splits(data_yaml.parent / "splits.json")

        self._dataset(root, 20, start=40)
        r2 = audit_dataset(root, progress=False)
        s2 = split_images(r2.images, r2.boxes, seed=7, pinned=pinned)
        rel = dict(zip(r2.images["image"], r2.images["rel_path"]))
        after = {rel[i]: v for i, v in zip(s2["image"], s2["split"])}
        assert all(after[k] == v for k, v in pinned.items())

    def test_missing_file_returns_empty(self, tmp_path):
        assert load_pinned_splits(tmp_path / "nope.json") == {}

    def test_corrupt_file_returns_empty(self, tmp_path):
        p = tmp_path / "splits.json"
        p.write_text("{not json", encoding="utf-8")
        assert load_pinned_splits(p) == {}


class TestPrepare:
    def _prepare(self, root, out, **kwargs):
        r = audit_dataset(root, progress=False)
        cm = make_class_map(r.boxes)
        s = split_images(r.images, r.boxes, seed=1, **kwargs.pop("split_kwargs", {}))
        return r, cm, prepare_yolo_dataset(r, out, "proj", cm, s, overwrite=True,
                                           progress=False, **kwargs)

    def test_writes_expected_layout(self, simple_root, tmp_path):
        _, cm, data_yaml = self._prepare(simple_root, tmp_path / "prep")
        assert data_yaml.exists()
        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        assert data["names"] == {i: c for c, i in sorted(cm.items(), key=lambda kv: kv[1])}
        out = data_yaml.parent
        for split in ("train", "val", "test"):
            assert (out / "images" / split).is_dir()
            assert (out / "labels" / split).is_dir()

    def test_overwrite_guard(self, simple_root, tmp_path):
        """Replacing a prepared dataset deletes the tree an in-flight run reads."""
        r = audit_dataset(simple_root, progress=False)
        cm = make_class_map(r.boxes)
        s = split_images(r.images, r.boxes, seed=1)
        prepare_yolo_dataset(r, tmp_path / "prep", "proj", cm, s, overwrite=True, progress=False)
        with pytest.raises(FileExistsError, match="overwrite=True"):
            prepare_yolo_dataset(r, tmp_path / "prep", "proj", cm, s, progress=False)

    def test_labels_are_correct(self, tmp_path):
        root = tmp_path / "d"
        for i in range(6):
            write_pair(root, f"i{i}", size=(200, 100), objects=[("lion", 0, 0, 100, 50)])
        _, _, data_yaml = self._prepare(root, tmp_path / "prep")

        out = data_yaml.parent
        labels = [p for split in ("train", "val", "test")
                  for p in (out / "labels" / split).glob("*.txt") if p.stat().st_size]
        assert labels
        # (0,0)-(100,50) in 200x100 -> centre (0.25,0.25), size (0.5,0.5)
        assert labels[0].read_text(encoding="utf-8").strip() == "0 0.250000 0.250000 0.500000 0.500000"

    def test_backgrounds_get_empty_labels(self, voc_root, tmp_path):
        """Empty frames must survive as negatives.

        The old implementation iterated the boxes frame, so every image without a
        box vanished from the dataset entirely.
        """
        _, _, data_yaml = self._prepare(voc_root, tmp_path / "prep")
        out = data_yaml.parent
        images = {p.stem for split in ("train", "val", "test")
                  for p in (out / "images" / split).glob("*")}
        assert "empty" in images
        label = next(p for split in ("train", "val", "test")
                     for p in (out / "labels" / split).glob("empty.txt"))
        assert label.read_text(encoding="utf-8") == ""

    def test_split_is_the_only_membership_authority(self, voc_root, tmp_path):
        """prepare writes exactly what the split selected - no second filter.

        Two independent background filters could disagree; there is deliberately
        only one.
        """
        _, _, data_yaml = self._prepare(
            voc_root, tmp_path / "prep", split_kwargs={"background_policy": "none"}
        )
        out = data_yaml.parent
        images = {p.stem for split in ("train", "val", "test")
                  for p in (out / "images" / split).glob("*")}
        assert "bg" not in images and "empty" not in images

    def test_policy_all_prepares_untagged_blanks(self, voc_root, tmp_path):
        _, _, data_yaml = self._prepare(
            voc_root, tmp_path / "prep", split_kwargs={"background_policy": "all"}
        )
        out = data_yaml.parent
        images = {p.stem for split in ("train", "val", "test")
                  for p in (out / "images" / split).glob("*")}
        assert "bg" in images and "empty" in images

    def test_no_filename_collisions(self, voc_root, tmp_path):
        """Two 'a.jpg' in different folders must both appear.

        Flattening on the bare name silently dropped one image and one label.
        """
        _, _, data_yaml = self._prepare(voc_root, tmp_path / "prep")
        out = data_yaml.parent
        stems = [p.stem for split in ("train", "val", "test")
                 for p in (out / "images" / split).glob("*")]
        assert len(stems) == len(set(stems))
        assert "a" in stems and "sub__a" in stems

    def test_label_uses_real_image_size(self, voc_root, tmp_path):
        """The 'mismatch' box must be normalised against 200x100, not 400x200."""
        _, cm, data_yaml = self._prepare(voc_root, tmp_path / "prep")
        out = data_yaml.parent
        label = next(p for split in ("train", "val", "test")
                     for p in (out / "labels" / split).glob("mismatch.txt"))
        parts = label.read_text(encoding="utf-8").split()
        # (10,10)-(60,60) in 200x100 -> centre (0.175, 0.35), size (0.25, 0.5)
        assert float(parts[1]) == pytest.approx(0.175)
        assert float(parts[2]) == pytest.approx(0.35)

    def test_writes_manifest_and_class_map(self, simple_root, tmp_path):
        _, cm, data_yaml = self._prepare(simple_root, tmp_path / "prep")
        out = data_yaml.parent
        assert json.loads((out / "classes.json").read_text(encoding="utf-8")) == cm
        manifest = json.loads((out / "dataset_manifest.json").read_text(encoding="utf-8"))
        assert manifest["num_classes"] == len(cm)
        assert sum(manifest["images_per_split"].values()) > 0

    @pytest.mark.parametrize("link_mode", ["copy", "auto"])
    def test_link_modes_produce_readable_images(self, simple_root, tmp_path, link_mode):
        from PIL import Image

        _, _, data_yaml = self._prepare(simple_root, tmp_path / f"prep_{link_mode}", link_mode=link_mode)
        out = data_yaml.parent
        img = next(p for split in ("train", "val", "test")
                   for p in (out / "images" / split).glob("*.jpg"))
        with Image.open(img) as im:
            assert im.size == (200, 100)

    def test_warns_loudly_when_linking_falls_back_to_copy(self, simple_root, tmp_path, monkeypatch, capsys):
        """A silent fallback to copy duplicates the whole dataset.

        Hit whenever prepared/ sits on a Windows filesystem: hardlinks cannot
        cross filesystems and drvfs symlinks usually need Developer Mode. Finding
        out via `df` hours later is not acceptable.
        """
        import os as _os

        def _no_symlinks(src, dst, **kwargs):
            raise OSError("operation not permitted")

        monkeypatch.setattr(_os, "symlink", _no_symlinks)
        monkeypatch.setattr(_os, "link", _no_symlinks)

        r = audit_dataset(simple_root, progress=False)
        cm = make_class_map(r.boxes)
        s = split_images(r.images, r.boxes, seed=1)
        data_yaml = prepare_yolo_dataset(r, tmp_path / "prep", "proj", cm, s,
                                         overwrite=True, link_mode="symlink", progress=False)

        out = capsys.readouterr().out
        assert "had to be COPIED" in out
        assert "duplicates" in out
        manifest = json.loads((data_yaml.parent / "dataset_manifest.json").read_text(encoding="utf-8"))
        assert manifest["link_mode_used"] == {"copy": 12}

    def test_no_warning_when_copy_was_requested(self, simple_root, tmp_path, capsys):
        r = audit_dataset(simple_root, progress=False)
        cm = make_class_map(r.boxes)
        s = split_images(r.images, r.boxes, seed=1)
        prepare_yolo_dataset(r, tmp_path / "prep", "proj", cm, s,
                             overwrite=True, link_mode="copy", progress=False)
        assert "had to be COPIED" not in capsys.readouterr().out

    def test_unknown_class_is_skipped_not_crash(self, simple_root, tmp_path):
        r = audit_dataset(simple_root, progress=False)
        partial = {"elephant": 0}   # 'lion' deliberately absent
        s = split_images(r.images, r.boxes, seed=1)
        data_yaml = prepare_yolo_dataset(r, tmp_path / "prep", "proj", partial, s,
                                         overwrite=True, progress=False)
        manifest = json.loads((data_yaml.parent / "dataset_manifest.json").read_text(encoding="utf-8"))
        assert manifest["classes_not_in_map"] == ["lion"]


class TestVerifyPrepared:
    def test_clean_dataset_passes(self, simple_root, tmp_path):
        r = audit_dataset(simple_root, progress=False)
        cm = make_class_map(r.boxes)
        s = split_images(r.images, r.boxes, seed=1)
        data_yaml = prepare_yolo_dataset(r, tmp_path / "prep", "proj", cm, s,
                                         overwrite=True, progress=False)
        report = verify_prepared(data_yaml)
        assert report["ok"], report["problems"]
        assert report["splits"]["train"]["images"] > 0

    def test_detects_out_of_range_class_id(self, simple_root, tmp_path):
        r = audit_dataset(simple_root, progress=False)
        cm = make_class_map(r.boxes)
        s = split_images(r.images, r.boxes, seed=1)
        data_yaml = prepare_yolo_dataset(r, tmp_path / "prep", "proj", cm, s,
                                         overwrite=True, progress=False)
        label = next(p for p in (data_yaml.parent / "labels" / "train").glob("*.txt")
                     if p.stat().st_size)
        label.write_text("99 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        report = verify_prepared(data_yaml)
        assert not report["ok"]
        assert any("class id" in p for p in report["problems"])
