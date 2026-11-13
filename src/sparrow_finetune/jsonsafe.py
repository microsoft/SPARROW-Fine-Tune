"""Non-finite floats (NaN, ±Infinity) are not JSON; Postgres ``jsonb`` rejects them."""

from __future__ import annotations

import math
from typing import Any, Optional


def find_non_finite(value: Any, path: str = "") -> Optional[str]:
    """Return the path of the first non-finite float inside ``value``, or ``None``."""
    if isinstance(value, float):
        return None if math.isfinite(value) else (path or "<root>")
    if isinstance(value, dict):
        for key, item in value.items():
            found = find_non_finite(item, f"{path}.{key}" if path else str(key))
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            found = find_non_finite(item, f"{path}[{i}]")
            if found is not None:
                return found
    return None


def sanitise_non_finite(value: Any) -> Any:
    """Copy ``value`` with every non-finite float replaced by ``None`` (JSON null)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: sanitise_non_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitise_non_finite(item) for item in value]
    return value
