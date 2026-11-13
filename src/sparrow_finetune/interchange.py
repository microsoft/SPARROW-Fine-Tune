"""Format-neutral annotation records shared by the COCO, Camera Trap DP and Pascal VOC importers.

A reader turns an upload into one ``Record`` per image it describes. The API layer matches each
record to a dataset image and replaces that item's annotations; the readers never see the
database. Boxes are kept as corners (``x0, y0, x1, y1``) in either pixels or 0..1 fractions of
the image, because a fraction can only become pixels once the image (and its size) is known.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from typing import Optional

from .voc import MAX_MEMBERS, MAX_TOTAL_BYTES, _leaf, _unsafe_member


class FormatError(ValueError):
    """The upload as a whole cannot be read (the message says why)."""


@dataclass
class Ann:
    label: str
    corners: Optional[tuple[float, float, float, float]] = None
    normalized: bool = False  # corners are fractions of the image width/height


@dataclass
class Record:
    source: (
        str  # where the record came from, for messages ("media.csv row 4", member name)
    )
    filename: str  # leaf file name of the image
    alt_names: list[str] = field(
        default_factory=list
    )  # other leaf names worth matching on
    width: int = 0  # 0 when the file does not say
    height: int = 0
    anns: list[Ann] = field(default_factory=list)  # empty = reviewed blank image
    skipped: list[str] = field(default_factory=list)


def leaf(name: str) -> str:
    return _leaf(name)


def zip_members(
    data: bytes, wanted: set[str], max_member_bytes: int
) -> dict[str, bytes]:
    """Read the members whose lower-cased leaf name is in ``wanted`` (first one wins).

    Bounded like the VOC reader: member count, per-member size and total size are measured on
    the bytes actually read, and absolute or ``..`` member names are refused.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, OSError) as exc:
        raise FormatError(f"body is not a zip archive: {exc}") from exc
    out: dict[str, bytes] = {}
    total = 0
    with zf:
        infos = zf.infolist()
        if len(infos) > MAX_MEMBERS:
            raise FormatError(
                f"archive has {len(infos)} members; the limit is {MAX_MEMBERS}"
            )
        for info in infos:
            name = info.filename
            key = _leaf(name).lower()
            if (
                info.is_dir()
                or name.startswith("__MACOSX/")
                or key not in wanted
                or key in out
            ):
                continue
            unsafe = _unsafe_member(name)
            if unsafe:
                raise FormatError(f"unsafe member name {name!r}: {unsafe}")
            if info.file_size > max_member_bytes:
                raise FormatError(f"{name} is larger than {max_member_bytes} bytes")
            try:
                with zf.open(info) as fh:
                    body = fh.read(max_member_bytes + 1)
            except (
                zipfile.BadZipFile,
                RuntimeError,
                NotImplementedError,
                OSError,
                EOFError,
            ) as exc:
                raise FormatError(f"cannot read {name}: {exc}") from exc
            if len(body) > max_member_bytes:
                raise FormatError(f"{name} is larger than {max_member_bytes} bytes")
            total += len(body)
            if total > MAX_TOTAL_BYTES:
                raise FormatError(
                    f"archive expands to more than {MAX_TOTAL_BYTES} bytes"
                )
            out[key] = body
    return out


def zipped(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, body in members.items():
            zf.writestr(name, body)
    return buf.getvalue()


__all__ = ["Ann", "FormatError", "Record", "leaf", "zip_members", "zipped"]
