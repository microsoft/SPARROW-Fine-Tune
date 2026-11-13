"""Image embedders: turn PIL images into fixed-size float32 feature vectors.

Selected at runtime by the ``SPARROW_FT_EMBEDDER`` setting (see ``make_embedder``).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image

ENV_VAR = "SPARROW_FT_EMBEDDER"
ENGINE_MAX_BATCH = 32
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
POOLED_OUTPUT_NAMES = ("pooler_output", "pooled_output", "pooled", "image_embeds", "embedding", "embeddings")


class EmbedderError(RuntimeError):
    """An embedding backend failed or returned an invalid response."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class Embedder(ABC):
    id: str
    dim: int
    model_hash: str

    @abstractmethod
    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Return float32 ``(len(images), dim)`` features."""

    def preprocess_crop(self, image: Image.Image) -> Image.Image:
        """Shrink a crop as early as possible without changing ``embed``'s result (default: unchanged)."""
        return image

    def close(self) -> None:
        """Release resources the embedder created itself (default: nothing to release)."""

    def __enter__(self) -> "Embedder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OnnxEmbedder(Embedder):
    """Runs an ONNX image encoder locally with onnxruntime (input NCHW float32)."""

    def __init__(
        self,
        model_path,
        *,
        input_size: int = 224,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
        batch_size: int = 16,
    ) -> None:
        import onnxruntime as ort

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"ONNX embedder model not found: {path}")
        self.model_path = path
        self.input_size = int(input_size)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 3, 1, 1)
        self.batch_size = max(1, int(batch_size))
        self.id = f"onnx:{path.stem}"
        self.model_hash = _sha256_file(path)
        self._session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output = self._select_output()
        self.dim = self._static_dim()

    def _select_output(self):
        """A pooled rank-2 output if the model has one (by name first), else the first output.

        A rank-3 output ``[N, T, D]`` (e.g. ``last_hidden_state``) is reduced to its CLS token in ``embed``.
        """
        outputs = self._session.get_outputs()
        rank2 = [o for o in outputs if o.shape is not None and len(o.shape) == 2]
        for name in POOLED_OUTPUT_NAMES:
            for o in rank2:
                if o.name == name:
                    return o
        for o in rank2:
            if "pool" in o.name.lower():
                return o
        return rank2[0] if rank2 else outputs[0]

    def _static_dim(self) -> Optional[int]:
        shape = self._output.shape or []
        dims = shape[2:] if len(shape) == 3 else shape[1:]
        if dims and all(isinstance(d, int) and d > 0 for d in dims):
            return int(np.prod(dims))
        return None

    def _resize(self, image: Image.Image) -> Image.Image:
        size = (self.input_size, self.input_size)
        image = image.convert("RGB")
        return image if image.size == size else image.resize(size, Image.BILINEAR)

    def preprocess_crop(self, image: Image.Image) -> Image.Image:
        return self._resize(image)

    def _preprocess(self, images: Sequence[Image.Image]) -> np.ndarray:
        arrays = [np.asarray(self._resize(im), dtype=np.float32) for im in images]
        batch = np.stack(arrays).transpose(0, 3, 1, 2) / 255.0
        return ((batch - self.mean) / self.std).astype(np.float32)

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        images = list(images)
        if not images:
            return np.zeros((0, self.dim or 0), dtype=np.float32)
        chunks = []
        for start in range(0, len(images), self.batch_size):
            batch = self._preprocess(images[start : start + self.batch_size])
            (out,) = self._session.run([self._output.name], {self._input_name: batch})
            out = np.asarray(out, dtype=np.float32)
            if out.ndim == 3:
                out = out[:, 0, :]
            chunks.append(out.reshape(len(batch), -1))
        features = np.concatenate(chunks)
        if self.dim is None:
            self.dim = int(features.shape[1])
        return features


class EngineEmbedder(Embedder):
    """Calls sparrow-engine ``POST /v1/embed/batch`` (multipart ``images``, ``store=false``)."""

    def __init__(
        self,
        base_url: str,
        model_id: str,
        *,
        dim: Optional[int] = None,
        batch_size: int = 32,
        client=None,
        timeout: float = 60,
        input_size: int = 224,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.dim = dim
        self.batch_size = max(1, min(int(batch_size), ENGINE_MAX_BATCH))
        self.id = f"engine:{model_id}"
        self.model_hash = f"engine:{model_id}"
        # The preprocessing the engine applies; recorded in exported manifests.
        self.input_size = int(input_size)
        self.mean = [float(v) for v in mean]
        self.std = [float(v) for v in std]
        self._timeout = timeout
        self._owns_client = client is None
        self._client = client if client is not None else self._new_client()

    def _new_client(self):
        import httpx

        return httpx.Client(timeout=self._timeout)

    def close(self) -> None:
        """Close the HTTP client if this embedder created it; a later ``embed`` opens a new one."""
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    @staticmethod
    def _jpeg(image: Image.Image) -> bytes:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=95)
        return buf.getvalue()

    def _embed_batch(self, images: list[Image.Image]) -> np.ndarray:
        import httpx

        files = [
            ("images", (f"{i}.jpg", self._jpeg(im), "image/jpeg"))
            for i, im in enumerate(images)
        ]
        if self._client is None:
            self._client = self._new_client()
        try:
            response = self._client.post(
                f"{self.base_url}/v1/embed/batch",
                params={"model": self.model_id, "store": "false"},
                files=files,
            )
        except httpx.HTTPError as exc:
            raise EmbedderError(f"embedding request failed: {exc}") from exc
        if not response.is_success:
            raise EmbedderError(
                f"engine returned HTTP {response.status_code}: {response.text[:200]}",
                status_code=response.status_code,
            )
        try:
            results = response.json()["results"]
        except (ValueError, KeyError, TypeError) as exc:
            raise EmbedderError(f"malformed engine response: {exc}") from exc
        if not isinstance(results, list) or len(results) != len(images):
            got = len(results) if isinstance(results, list) else type(results).__name__
            raise EmbedderError(
                f"engine returned {got} embeddings for {len(images)} images"
            )
        by_index: dict[int, list] = {}
        for entry in results:
            index = entry.get("index") if isinstance(entry, dict) else None
            if not isinstance(index, int) or not 0 <= index < len(images):
                raise EmbedderError(f"engine returned an invalid index: {index!r}")
            if index in by_index:
                raise EmbedderError(f"engine returned duplicate index {index}")
            by_index[index] = entry.get("embedding")
        try:
            out = np.asarray(
                [by_index[i] for i in range(len(images))], dtype=np.float32
            )
        except (ValueError, TypeError) as exc:
            raise EmbedderError(f"engine returned malformed embeddings: {exc}") from exc
        if out.ndim != 2 or (self.dim is not None and out.shape[1] != self.dim):
            raise EmbedderError(
                f"engine embeddings have shape {out.shape}, expected dim {self.dim}"
            )
        return out

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        images = list(images)
        if not images:
            return np.zeros((0, self.dim or 0), dtype=np.float32)
        chunks = []
        for start in range(0, len(images), self.batch_size):
            chunk = self._embed_batch(images[start : start + self.batch_size])
            if self.dim is None:
                self.dim = int(chunk.shape[1])
            chunks.append(chunk)
        return np.concatenate(chunks)


def make_embedder(spec: str) -> Embedder:
    """``onnx:<path>`` or ``engine:<base_url>#<model_id>`` -> an ``Embedder``."""
    spec = (spec or "").strip()
    if not spec:
        raise ValueError(
            f"{ENV_VAR} is not set: use 'onnx:<path>' or 'engine:<url>#<model_id>'"
        )
    kind, sep, rest = spec.partition(":")
    rest = rest.strip()
    if sep and kind == "onnx" and rest:
        return OnnxEmbedder(rest)
    if sep and kind == "engine":
        url, hash_sep, model_id = rest.rpartition("#")
        if hash_sep and url.strip() and model_id.strip():
            return EngineEmbedder(url.strip(), model_id.strip())
    raise ValueError(
        f"invalid {ENV_VAR} value {spec!r}: use 'onnx:<path>' or 'engine:<url>#<model_id>'"
    )


# --------------------------------------------------------------------------- embedding cache

CACHE_FORMAT_VERSION = 1


def embedder_fingerprint(embedder: "Embedder") -> str:
    """sha256 over everything that decides an embedding besides the crop pixels.

    The embedder class, id and ``model_hash`` (the ONNX file's sha256; for the engine only the
    model id, which the engine does not attest), plus the preprocessing it records (input size,
    mean, std). Two embedders with equal fingerprints must return equal vectors for equal crops.
    """
    payload = {
        "format": CACHE_FORMAT_VERSION,
        "class": f"{type(embedder).__module__}.{type(embedder).__qualname__}",
        "id": str(getattr(embedder, "id", "")),
        "model_hash": str(getattr(embedder, "model_hash", "")),
        "input_size": getattr(embedder, "input_size", None),
        "mean": [float(v) for v in np.ravel(getattr(embedder, "mean", ()))],
        "std": [float(v) for v in np.ravel(getattr(embedder, "std", ()))],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


class EmbeddingCache:
    """Content-addressed crop embeddings on local disk, shared by every job of a service.

    Layout: ``<root>/<fingerprint>/<key[:2]>/<key>.npy``, one little-endian float32 vector per
    file, with ``key = sha256(fingerprint, image sha256, integer crop box)``. Files are written to a
    unique temporary name and renamed into place, so concurrent jobs never see a partial entry
    (both writers produce the same bytes). Entries that fail to load or validate are counted as
    corrupt, treated as misses and overwritten. Read with ``allow_pickle=False``.

    There is no eviction: the cache grows by about ``4 * dim + 128`` bytes per distinct crop and
    embedder; delete ``<root>`` (or one fingerprint directory) to reclaim space.
    """

    def __init__(self, root: Path, fingerprint: str) -> None:
        self.root = Path(root)
        self.fingerprint = fingerprint
        self.dir = self.root / fingerprint
        self.hits = 0
        self.misses = 0
        self.corrupt = 0
        self.writes = 0
        self.write_errors = 0

    def key(self, image_sha256: str, box: Sequence[int]) -> str:
        text = "\x1f".join(
            [self.fingerprint, image_sha256, ",".join(str(int(v)) for v in box)]
        )
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.npy"

    def get(self, key: str, dim: Optional[int] = None) -> Optional[np.ndarray]:
        """The cached vector, or None (a miss; unreadable or invalid entries also count corrupt)."""
        try:
            vector = np.load(self.path(key), allow_pickle=False)
        except FileNotFoundError:
            self.misses += 1
            return None
        except (OSError, ValueError, EOFError):
            self.corrupt += 1
            self.misses += 1
            return None
        if (
            not isinstance(vector, np.ndarray)
            or vector.dtype != np.dtype("<f4")
            or vector.ndim != 1
            or vector.size == 0
            or (dim is not None and vector.shape[0] != dim)
            or not np.isfinite(vector).all()
        ):
            self.corrupt += 1
            self.misses += 1
            return None
        self.hits += 1
        return vector

    def put(self, key: str, vector: np.ndarray) -> None:
        """Store ``vector``; a failed write is counted, never raised (the cache is best effort)."""
        path = self.path(key)
        tmp = path.parent / f".{key}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as fh:
                np.save(fh, np.ascontiguousarray(vector, dtype="<f4").reshape(-1))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
            self.writes += 1
        except OSError:
            self.write_errors += 1
            try:
                tmp.unlink()
            except OSError:
                pass

    def stats(self) -> dict:
        return {
            "enabled": True,
            "fingerprint": self.fingerprint,
            "hits": self.hits,
            "misses": self.misses,
            "corrupt": self.corrupt,
            "writes": self.writes,
            "write_errors": self.write_errors,
        }
