#!/usr/bin/env python3
"""Download C. elegans connectome CSV data from OpenWorm GitHub.

Usage::

    python data/download.py

Files are saved to ``data/raw/``. Fully idempotent — skips re-download if
checksum matches. Falls back gracefully through three sources:

    1. Network download (OpenWorm GitHub — multiple URL variants)
    2. Embedded real Varshney et al. (2011) data (offline, always available)

The ``SYNTHETIC`` fallback of a previous version has been removed. The
embedded Varshney data is real biological data from the published connectome.

Bug fixes vs v2
---------------
- ``_validate_csv_format`` previously checked for literal ``Origin/Target/Number``
  headers, causing the real OpenWorm CSVs (which use ``Pre/Post/Number``,
  ``Sending Cell Body/Receiving Cell Body/...``, etc.) to always fail validation
  and silently write synthetic data instead.
- The download now normalises column names immediately after fetching and saves
  the canonical ``Origin/Target/Number/Type`` format to disk.
- Added additional URL sources covering all known OpenWorm CSV formats.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column aliases — must match celegans/connectome.py
# ---------------------------------------------------------------------------
_ORIGIN_ALIASES = {
    "origin", "pre", "from", "neuron1", "neuron 1", "source",
    "preneuroname", "sending cell body",
}
_TARGET_ALIASES = {
    "target", "post", "to", "neuron2", "neuron 2", "destination",
    "postneuroname", "receiving cell body",
}
_NUMBER_ALIASES = {
    "number", "sections", "count", "weight", "synapses", "n",
    "number of gap junctions", "number of chemical junctions",
    "nbconnections", "strength", "number of connections",
}

# ---------------------------------------------------------------------------
# Download sources
# ---------------------------------------------------------------------------
_CHEMICAL_URLS: List[str] = [
    # c302 canonical table (Pre/Post/Number of Chemical Junctions)
    "https://raw.githubusercontent.com/openworm/c302/master/c302/CElegansNeuronTables.csv",
    # PyOpenWorm test data (Sending Cell Body / Receiving Cell Body / Number)
    "https://raw.githubusercontent.com/openworm/PyOpenWorm/master/tests/test_data/Chemical%20Synapse.csv",
    # WormAtlas mirror on OpenWorm data
    "https://raw.githubusercontent.com/openworm/data/master/connectome/chemical.csv",
]
_GAP_URLS: List[str] = [
    "https://raw.githubusercontent.com/openworm/c302/master/c302/CElegansNeuronTablesGapJunctions.csv",
    "https://raw.githubusercontent.com/openworm/PyOpenWorm/master/tests/test_data/Electrical%20Synapse.csv",
    "https://raw.githubusercontent.com/openworm/data/master/connectome/gap.csv",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_column(headers: List[str], aliases: set) -> Optional[str]:
    """Return the first header whose .strip().lower() is in *aliases*."""
    for h in headers:
        if h.strip().lower() in aliases:
            return h
    return None


def _normalize_csv_bytes(raw_bytes: bytes, source_label: str) -> Optional[str]:
    """Parse CSV bytes, rename columns to Origin/Target/Number/Type, return CSV string.

    Returns None if the file cannot be mapped to the required columns.
    """
    import io
    text = raw_bytes.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    headers = list(reader.fieldnames or [])

    origin_col = _find_column(headers, _ORIGIN_ALIASES)
    target_col = _find_column(headers, _TARGET_ALIASES)
    number_col = _find_column(headers, _NUMBER_ALIASES)

    if not (origin_col and target_col and number_col):
        logger.warning(
            "[%s] Cannot map columns %s -- missing: %s",
            source_label, headers,
            [k for k, v in [("Origin", origin_col), ("Target", target_col),
                             ("Number", number_col)] if v is None],
        )
        return None

    if origin_col != "Origin" or target_col != "Target" or number_col != "Number":
        logger.info(
            "[%s] Normalising columns: %s->Origin, %s->Target, %s->Number",
            source_label, origin_col, target_col, number_col,
        )

    rows = []
    for row in reader:
        origin = row.get(origin_col, "").strip()
        target = row.get(target_col, "").strip()
        raw_n = row.get(number_col, "").strip()
        if not origin or not target or not raw_n:
            continue
        try:
            n = int(float(raw_n))
        except ValueError:
            continue
        if n < 0:
            continue
        rows.append({"Origin": origin, "Target": target,
                     "Number": n, "Type": row.get("Type", "unknown").strip()})

    if not rows:
        logger.warning("[%s] No valid rows after normalisation", source_label)
        return None

    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=["Origin", "Target", "Number", "Type"])
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def _fetch(url: str, timeout: int = 30) -> Optional[bytes]:
    """Attempt HTTP GET; return bytes or None on any error."""
    try:
        import requests  # type: ignore
        with requests.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            return resp.content
    except Exception as exc:
        logger.warning("Network fetch failed for %s: %s", url, exc)
        return None


def _try_urls(urls: List[str], label: str) -> Optional[str]:
    """Try each URL in order; return normalised CSV string or None."""
    for url in urls:
        logger.info("Trying %s from %s", label, url)
        raw = _fetch(url)
        if raw is None:
            continue
        normalised = _normalize_csv_bytes(raw, url)
        if normalised is not None:
            logger.info("Successfully fetched and normalised %s (%d bytes)", label, len(raw))
            return normalised
        logger.warning("Fetched %s but column normalisation failed -- trying next URL", url)
    return None


def _validate_normalised_csv(csv_str: str, label: str) -> bool:
    """Quick sanity check: Origin/Target/Number all present and non-empty."""
    reader = csv.DictReader(csv_str.splitlines())
    if not {"Origin", "Target", "Number"}.issubset(set(reader.fieldnames or [])):
        logger.error("[%s] Missing required columns after normalisation", label)
        return False
    rows = list(reader)
    if not rows:
        logger.error("[%s] No rows in normalised CSV", label)
        return False
    return True


def _write_atomic(dest: Path, content: str) -> None:
    """Write *content* to *dest* atomically."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(content)
        os.replace(tmp, dest)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return sum(1 for _ in csv.reader(fh)) - 1  # subtract header


def _existing_file_valid(path: Path) -> bool:
    """Return True if the file exists and has canonical Origin/Target/Number columns."""
    if not path.exists():
        return False
    try:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            headers = set(reader.fieldnames or [])
            if not {"Origin", "Target", "Number"}.issubset(headers):
                logger.info(
                    "Existing file %s has non-canonical columns %s -- will re-fetch",
                    path.name, headers,
                )
                return False
            rows = list(reader)
        if not rows:
            return False
        logger.info("Existing file valid: %s (%d rows)", path.name, len(rows))
        return True
    except Exception as exc:
        logger.warning("Could not read existing %s: %s", path.name, exc)
        return False


def _get_varshney_csv(chemical: bool) -> Tuple[str, int]:
    """Return (csv_string, row_count) from the embedded Varshney module."""
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    try:
        import varshney_connectome as vc  # type: ignore
    finally:
        sys.path.pop(0)

    chem_df, gap_df = vc.to_dataframes()
    import io
    df = chem_df if chemical else gap_df
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    csv_str = buf.getvalue()
    return csv_str, len(df)


def download_file(
    local_name: str,
    dest_dir: Path,
    urls: List[str],
    is_chemical: bool,
) -> Path:
    """Download, normalise, and save a single connectome CSV.

    Resolution order:
    1. Existing valid file (idempotent check).
    2. Network download (tries each URL until one succeeds and normalises).
    3. Embedded Varshney et al. 2011 data (real biological data, offline).

    Args:
        local_name:  Filename to save under *dest_dir*.
        dest_dir:    Destination directory (created if missing).
        urls:        Ordered list of URLs to try.
        is_chemical: True for chemical synapses, False for gap junctions.

    Returns:
        Path to the saved CSV.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / local_name

    # Step 1 -- idempotency
    if _existing_file_valid(dest_path):
        n = _row_count(dest_path)
        logger.info("  Ready: %s (%.1f KB, %d rows)", local_name,
                    dest_path.stat().st_size / 1024, n)
        return dest_path

    # Step 2 -- network
    csv_str = _try_urls(urls, local_name)
    source = "network"

    # Step 3 -- embedded Varshney data (real, offline)
    if csv_str is None:
        logger.warning(
            "All network sources failed for %s -- using embedded Varshney (2011) data.",
            local_name,
        )
        try:
            csv_str, n_rows = _get_varshney_csv(is_chemical)
            source = "embedded Varshney (2011)"
            logger.info(
                "Loaded %d rows from embedded Varshney data for %s", n_rows, local_name
            )
        except Exception as exc:
            logger.error(
                "Failed to load embedded Varshney data: %s\n"
                "Cannot produce %s without either network access or the "
                "varshney_connectome.py module in the data/ directory.",
                exc, local_name,
            )
            raise RuntimeError(
                f"Could not obtain connectome data for {local_name}. "
                "Check network connectivity or ensure data/varshney_connectome.py exists."
            ) from exc

    if csv_str is None or not _validate_normalised_csv(csv_str, local_name):
        raise RuntimeError(
            f"Normalised CSV for {local_name} failed validation -- this is a bug."
        )

    _write_atomic(dest_path, csv_str)
    n = _row_count(dest_path)
    logger.info("Wrote %d rows -> %s  [source: %s]", n, dest_path, source)

    # Re-read and confirm canonical columns
    with dest_path.open(newline="", encoding="utf-8-sig") as fh:
        headers = set(next(csv.reader(fh)))
    if {"Origin", "Target", "Number"}.issubset(headers):
        logger.info("Columns canonical: %s (%d rows)", local_name, n)
    else:
        logger.warning("Unexpected columns in %s: %s", local_name, headers)

    logger.info("  Ready: %s (%.1f KB, %d rows)", local_name,
                dest_path.stat().st_size / 1024, n)
    return dest_path


def main() -> None:
    """Entry point: download / verify all connectome files to data/raw/."""
    script_dir = Path(__file__).resolve().parent
    dest_dir = script_dir / "raw"
    logger.info("Destination: %s", dest_dir)

    files: List[Tuple[str, List[str], bool]] = [
        ("connectome_chemical.csv", _CHEMICAL_URLS, True),
        ("connectome_gap.csv",      _GAP_URLS,      False),
    ]

    results: Dict[str, int] = {}
    for local_name, urls, is_chem in files:
        path = download_file(local_name, dest_dir, urls, is_chem)
        results[local_name] = _row_count(path)

    # Count unique neurons across both files
    neurons: set = set()
    for fname in ("connectome_chemical.csv", "connectome_gap.csv"):
        p = dest_dir / fname
        if p.exists():
            with p.open(newline="", encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    neurons.add(row.get("Origin", ""))
                    neurons.add(row.get("Target", ""))
    neurons.discard("")

    logger.info(
        "  connectome_chemical.csv: %d synapses, %d unique neurons",
        results.get("connectome_chemical.csv", 0), len(neurons),
    )
    logger.info(
        "  connectome_gap.csv: %d synapses, %d unique neurons",
        results.get("connectome_gap.csv", 0), len(neurons),
    )
    logger.info("Done. All connectome files ready.")


if __name__ == "__main__":
    main()
