"""Safe file I/O helpers with path traversal protection and atomic writes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def validate_path(path: Path, root: Path) -> Path:
    """Resolve *path* and assert it sits inside *root*.

    Args:
        path: Candidate file path.
        root: Trusted project root directory.

    Returns:
        The resolved :class:`~pathlib.Path`.

    Raises:
        ValueError: If *path* escapes *root* (path traversal guard).
    """
    resolved = path.resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        raise ValueError(
            f"Path traversal detected: {resolved} is outside root {root_resolved}"
        )
    return resolved


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write *content* to *path* atomically (write-to-tmp then os.replace).

    Args:
        path: Destination file path.
        content: Text content to write.
        encoding: Text encoding (default: utf-8).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(content)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Serialize *data* to JSON and write atomically to *path*.

    Args:
        path: Destination .json file path.
        data: JSON-serializable Python object.
        indent: JSON indentation level.
    """
    atomic_write_text(path, json.dumps(data, indent=indent) + "\n")


def read_json(path: Path) -> Any:
    """Read and parse a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed Python object.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file.

    Args:
        path: Path to the file.

    Returns:
        Lowercase hex-encoded SHA-256 digest string.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
