"""Download NHTSA flat files and convert them, unchanged, to Parquet.

Every value is kept as text here; parsing happens in clean.py. Each download is recorded in
data/raw/manifest.json with its size, SHA-256 and the server's Last-Modified date.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import httpx
from tqdm import tqdm

from pipeline.config import MANIFEST, RAW, SOURCES, Source


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(source: Source, force: bool = False) -> dict:
    dest = RAW / f"{source.name}.zip"
    if dest.exists() and not force:
        print(f"[ingest] {dest.name} exists, skipping download (use --force to refetch)")
        return {"file": dest.name, "url": source.url, "reused": True}

    tmp = dest.with_suffix(".part")
    with httpx.stream("GET", source.url, follow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        with tmp.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                       desc=dest.name) as bar:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
        last_modified = r.headers.get("last-modified")
    tmp.replace(dest)
    return {
        "file": dest.name,
        "url": source.url,
        "last_modified": last_modified,
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def tsv_to_parquet(con: duckdb.DuckDBPyConnection, txt: Path, columns: tuple[str, ...],
                   out: Path) -> int:
    """Read an NHTSA tab-delimited file (no header, no quoting) into Parquet, all VARCHAR.

    A line with the wrong number of fields (e.g. a narrative split by a stray newline) raises an
    error naming the line instead of being silently padded with NULLs. Blank lines, which the
    recall files contain, are skipped.
    """
    cols = "{" + ", ".join(f"'{c}': 'VARCHAR'" for c in columns) + "}"
    con.execute(f"""
        COPY (
            SELECT * FROM read_csv(
                '{txt.as_posix()}',
                delim = '\t', header = false, quote = '', escape = '',
                columns = {cols}, null_padding = false, encoding = 'utf-8',
                max_line_size = 1048576
            )
        ) TO '{out.as_posix()}' (FORMAT parquet, COMPRESSION zstd)
    """)
    return con.execute(f"SELECT count(*) FROM '{out.as_posix()}'").fetchone()[0]


def zip_to_parquet(con: duckdb.DuckDBPyConnection, source: Source) -> int:
    zpath = RAW / f"{source.name}.zip"
    out = RAW / f"{source.name}.parquet"
    with zipfile.ZipFile(zpath) as z, tempfile.TemporaryDirectory() as tmp:
        members = [m for m in z.namelist() if m.lower().endswith(".txt")]
        if len(members) != 1:
            raise RuntimeError(f"{zpath.name}: expected one .txt inside, found {members}")
        txt = Path(z.extract(members[0], tmp))
        return tsv_to_parquet(con, txt, source.columns, out)


def run(force: bool = False) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    con = duckdb.connect()
    for source in SOURCES:
        entry = download(source, force=force)
        prev = manifest.get(source.name, {})
        if entry.pop("reused", False):
            entry = {**prev, **entry}
        zpath = RAW / f"{source.name}.zip"
        entry["bytes"] = zpath.stat().st_size
        entry["sha256"] = sha256(zpath)
        # Save download metadata before converting, so a failed conversion doesn't lose it.
        manifest[source.name] = entry
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"[ingest] {source.name}: converting to Parquet")
        entry["rows"] = zip_to_parquet(con, source)
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"[ingest] {source.name}: {entry['rows']:,} rows")
