"""SPARROW Workbench web application.

A Flask UI over :mod:`sparrow_yolo`. This package depends on that one; never the
reverse — the CLI and library must keep working with this uninstalled.

Everything the UI does is a call into the same functions the CLI uses, so any
behaviour here is scriptable and testable without a browser.
"""

from __future__ import annotations

__version__ = "0.1.0"
