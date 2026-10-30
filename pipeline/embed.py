"""Embed scrubbed complaint narratives -> data/clean/embeddings/<model>/part-*.parquet.

One row per distinct narrative text: narrative_id (joins to narratives.parquet) and a unit-length
float16 vector, so cosine similarity is a plain dot product. float16 halves the disk needed
(about 1.3 GB for 1.6M texts) and moves cosine similarities by about 0.001.

Runs are incremental: only texts without a vector are encoded, a chunk at a time, each chunk in
its own file. An interrupted run continues where it stopped, and after a monthly NHTSA refresh
only new complaints are encoded. A different model writes to its own folder.
"""

from __future__ import annotations

import shutil
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.config import CLEAN, EMBED_MODEL

MIN_FREE_BYTES = 2 << 30  # stop before the disk fills; rerunning picks up where it stopped

Encoder = Callable[[list[str]], np.ndarray]


def load_encoder(model_name: str, batch_size: int = 64) -> Encoder:
    from sentence_transformers import SentenceTransformer  # pip install -e ".[text]"

    model = SentenceTransformer(model_name)
    if model.device.type in ("mps", "cuda"):
        # On an M1 GPU: 279 texts/s instead of 223 at float32, with cosine 1.00000 between the two.
        model.half()
    print(f"[embed] {model_name} on {model.device}, "
          f"first {model.max_seq_length} tokens of each text")

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False)
    return encode


def run(clean: Path = CLEAN, model_name: str = EMBED_MODEL, limit: int | None = None,
        chunk_size: int = 50_000, encode: Encoder | None = None) -> None:
    out_dir = clean / "embeddings" / model_name.replace("/", "--")
    out_dir.mkdir(parents=True, exist_ok=True)
    for leftover in out_dir.glob("*.tmp"):  # from a run killed mid-write
        leftover.unlink()
    narratives = (clean / "narratives.parquet").as_posix()
    done = (f"WHERE narrative_id NOT IN (SELECT narrative_id FROM "
            f"read_parquet('{out_dir.as_posix()}/*.parquet'))"
            if any(out_dir.glob("*.parquet")) else "")
    # One row per text, picked through a small (narrative_id, complaint_id) table so the texts
    # stream from the file instead of all being held in memory at once.
    todo = f"""
        SELECT narrative_id, description FROM '{narratives}'
        SEMI JOIN (SELECT narrative_id, min(complaint_id) AS complaint_id FROM '{narratives}'
                   {done} GROUP BY narrative_id) USING (narrative_id, complaint_id)
        {f"LIMIT {int(limit)}" if limit else ""}
    """
    con = duckdb.connect()
    total = con.execute(f"SELECT count(*) FROM ({todo})").fetchall()[0][0]
    if total == 0:
        print("[embed] every narrative already has a vector")
        return
    encode = encode or load_encoder(model_name)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    finished, start = 0, time.perf_counter()
    for batch in con.sql(todo).to_arrow_reader(chunk_size):
        if shutil.disk_usage(out_dir).free < MIN_FREE_BYTES:
            raise SystemExit(f"[embed] stopped with {finished:,} of {total:,} done: under "
                             f"{MIN_FREE_BYTES >> 30} GB of disk left. Free space and rerun.")
        vectors = np.asarray(encode(batch.column("description").to_pylist()), dtype=np.float16)
        table = pa.table({
            "narrative_id": batch.column("narrative_id"),
            "embedding": pa.FixedSizeListArray.from_arrays(pa.array(vectors.reshape(-1)),
                                                           vectors.shape[1]),
        })
        part = out_dir / f"part-{stamp}-{uuid.uuid4().hex[:12]}.parquet"  # never reuse a name
        pq.write_table(table, part.with_suffix(".tmp"), compression="zstd")
        part.with_suffix(".tmp").rename(part)  # only finished files ever carry the .parquet name
        finished += len(table)
        rate = finished / (time.perf_counter() - start)
        print(f"[embed] {finished:,}/{total:,} narratives, {rate:,.0f}/s, "
              f"~{(total - finished) / rate / 60:.0f} min left", flush=True)
