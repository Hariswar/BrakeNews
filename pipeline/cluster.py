"""Group complaint narratives into recurring failure themes -> data/clean/clusters/.

HDBSCAN cannot run on 1.6M points in 8 GB, so clusters are found on a fixed random sample and
then every narrative is assigned to the nearest one:
  0. Most narratives open with NHTSA's "THE CONTACT OWNS A 2012 FORD FUSION", so raw vectors
     group by vehicle as much as by failure: a first fit gave 55 of 143 clusters where one make
     filed most complaints ("owns Ford Fusion", "Jeep Grand Cherokee"). Each vector has its make
     and model's mean vector subtracted (or its make's, for models with under MIN_GROUP
     narratives) and is re-normalised, which cut that to 17 of 120. The ones left are mostly
     real single-make problems (Takata airbags, Tesla phantom braking).
  1. UMAP reduces a sample of vectors to a few dimensions (HDBSCAN needs that to find dense
     regions; on raw 384-d vectors nearly everything is noise).
  2. HDBSCAN finds clusters of at least MIN_CLUSTER_SIZE sampled narratives and marks the rest
     as noise.
  3. Each cluster's centroid is the unit-length mean of its members' adjusted 384-d vectors. Every
     narrative goes to the centroid it is most similar to, or to -1 (no theme) when it is less
     similar than that cluster's own members usually are (their 5th percentile).

Fitting is slow (UMAP) and changes cluster ids, which would orphan the Claude-written names, so
a fit is kept until `cluster --refit` is asked for. A plain `cluster` run only re-assigns, which
after a monthly NHTSA refresh puts the new complaints into the existing themes in seconds.

Outputs (fit_id ties them together, and names.py checks it):
  clusters.parquet     one row per cluster: centroid (adjusted vectors), raw_centroid (original
                       vectors, for comparing with other texts such as recall summaries),
                       assignment threshold, sizes, distinctive
                       terms (class-based TF-IDF over the sample), top component groups, and
                       the narratives nearest the centroid (for naming and inspection)
  assignments.parquet  one row per distinct narrative: narrative_id, cluster_id (-1 = none),
                       similarity to the assigned centroid
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from pipeline.config import CLEAN, EMBED_MODEL

SAMPLE_SIZE = 150_000
MIN_CLUSTER_SIZE = 200  # in the sample: about 0.13%, ~2,000 narratives across the full set
MIN_SAMPLES = 10
UMAP_DIMS = 10
UMAP_NEIGHBORS = 30
THRESHOLD_QUANTILE = 0.05
MIN_GROUP = 200  # narratives a make or model needs before its mean vector is subtracted
N_EXAMPLES = 25
N_TERMS = 15
SEED = 0

# Words in nearly every narrative, plus the scrubber's placeholders.
_EXTRA_STOP_WORDS = """
    vehicle car truck suv van contact stated owner consumer manufacturer dealer dealership
    xxx phone vin id name number url email address zip plate approximately mileage miles
    failure failed issue problem also would could just said told time times did does year
    tl dt owns driving received
""".split()


def embeddings_dir(clean: Path, model_name: str) -> Path:
    return clean / "embeddings" / model_name.replace("/", "--")


def load_vectors(clean: Path, model_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Ids and float16 vectors (1.2 GB for 1.6M narratives) of the current narratives. Vectors
    of texts that a scrub rule has since changed stay on disk under their old id and are
    skipped here."""
    table = ds.dataset(embeddings_dir(clean, model_name), format="parquet").to_table()
    ids = table.column("narrative_id").to_numpy(zero_copy_only=False)
    current = duckdb.sql(f"SELECT DISTINCT narrative_id FROM "
                         f"'{(clean / 'narratives.parquet').as_posix()}'").fetchnumpy()
    wanted = set(current["narrative_id"].tolist())  # np.isin on object arrays is quadratic
    keep = np.fromiter((i in wanted for i in ids), dtype=bool, count=len(ids))
    if not keep.all():
        print(f"[cluster] skipping {(~keep).sum():,} vectors of texts no longer in narratives")
    missing = len(current["narrative_id"]) - keep.sum()
    if missing:
        print(f"[cluster] {missing:,} narratives have no vector yet; run `make embed` first")
    emb = table.column("embedding").combine_chunks()
    vectors = emb.values.to_numpy(zero_copy_only=False).reshape(len(ids), emb.type.list_size)
    ids, vectors = ids[keep], vectors[keep]
    order = np.argsort(ids)  # file order depends on the embed run; sort so samples repeat
    return ids[order], vectors[order]


def vehicle_offsets(clean: Path, ids: np.ndarray, vectors: np.ndarray,
                    min_group: int = MIN_GROUP) -> tuple[np.ndarray, np.ndarray]:
    """Per narrative, the row of `means` to subtract: its make and model's mean vector, else its
    make's, else the last row (zeros). A narrative filed for several vehicles uses the most
    common one."""
    con = duckdb.connect()
    con.register("ids", pa.table({"narrative_id": ids, "row": np.arange(len(ids))}))
    rows = con.execute(f"""
        SELECT coalesce(v.make, ''), coalesce(v.make || '|' || v.model, '') FROM ids
        LEFT JOIN (SELECT narrative_id, mode(make) AS make, mode(model) AS model
                   FROM '{(clean / 'narratives.parquet').as_posix()}'
                   JOIN '{(clean / 'complaints.parquet').as_posix()}' USING (complaint_id)
                   GROUP BY narrative_id) v USING (narrative_id)
        ORDER BY row""").fetchall()
    make = np.array([r[0] for r in rows], dtype=object)
    model = np.array([r[1] for r in rows], dtype=object)
    group = np.full(len(ids), -1, dtype=np.int64)
    n = 0
    for keys in (model, make):  # models first; makes only fill what models left
        u, inv, cnt = np.unique(keys.astype(str), return_inverse=True, return_counts=True)
        big = (cnt >= min_group) & (u != "")
        code = np.where(big, np.cumsum(big) - 1 + n, -1)[inv]
        group = np.where(group < 0, code, group)
        n += int(big.sum())
    group[group < 0] = n
    sums = np.zeros((n + 1, vectors.shape[1]))
    for i in range(0, len(ids), 200_000):
        np.add.at(sums, group[i:i + 200_000], vectors[i:i + 200_000].astype(np.float32))
    means = sums / np.maximum(np.bincount(group, minlength=n + 1), 1)[:, None]
    means[n] = 0
    return group, means.astype(np.float32)


def adjust(vectors: np.ndarray, group: np.ndarray, means: np.ndarray) -> np.ndarray:
    x = vectors.astype(np.float32) - means[group]
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)


def reduce(x: np.ndarray, seed: int = SEED) -> np.ndarray:
    import umap  # pip install -e ".[text]"

    return umap.UMAP(n_components=UMAP_DIMS, n_neighbors=UMAP_NEIGHBORS, min_dist=0.0,
                     metric="cosine", random_state=seed, low_memory=True).fit_transform(x)


def find_clusters(points: np.ndarray, min_cluster_size: int = MIN_CLUSTER_SIZE,
                  min_samples: int = MIN_SAMPLES) -> np.ndarray:
    from sklearn.cluster import HDBSCAN

    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                   copy=True).fit_predict(points)


def centroids_of(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    k = labels.max() + 1
    sums = np.zeros((k, x.shape[1]), dtype=np.float64)
    np.add.at(sums, labels[labels >= 0], x[labels >= 0])
    return (sums / np.linalg.norm(sums, axis=1, keepdims=True)).astype(np.float32)


def raw_centroids(vectors: np.ndarray, labels: np.ndarray, k: int,
                  chunk: int = 200_000) -> np.ndarray:
    """Unit-length mean of each cluster's original (unadjusted) vectors."""
    import scipy.sparse as sp

    sums = np.zeros((k, vectors.shape[1]), dtype=np.float64)
    for i in range(0, len(vectors), chunk):
        lab = labels[i:i + chunk]
        rows = np.flatnonzero(lab >= 0)
        member = sp.csr_matrix((np.ones(len(rows), np.float32), (lab[rows], rows)),
                               shape=(k, len(lab)))
        sums += member @ vectors[i:i + chunk].astype(np.float32)
    return (sums / np.maximum(np.linalg.norm(sums, axis=1, keepdims=True), 1e-9)).astype(
        np.float32)


def assign(vectors: np.ndarray, group: np.ndarray, means: np.ndarray, centroids: np.ndarray,
           thresholds: np.ndarray, chunk: int = 200_000) -> tuple[np.ndarray, np.ndarray]:
    """Nearest centroid by cosine, or -1 when below that centroid's threshold."""
    labels = np.empty(len(vectors), dtype=np.int32)
    sims = np.empty(len(vectors), dtype=np.float32)
    for i in range(0, len(vectors), chunk):
        s = adjust(vectors[i:i + chunk], group[i:i + chunk], means) @ centroids.T
        best = s.argmax(axis=1)
        sim = s[np.arange(len(best)), best]
        labels[i:i + chunk] = np.where(sim >= thresholds[best], best, -1)
        sims[i:i + chunk] = sim
    return labels, sims


def distinctive_terms(texts: list[str], labels: np.ndarray, n: int = N_TERMS) -> list[list[str]]:
    """Class-based TF-IDF: terms frequent in one cluster and rare across the others."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

    keep = labels >= 0
    k = labels.max() + 1
    cv = CountVectorizer(stop_words=list(ENGLISH_STOP_WORDS | set(_EXTRA_STOP_WORDS)),
                         ngram_range=(1, 2), min_df=5, max_features=60_000,
                         token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    counts = cv.fit_transform([t for t, m in zip(texts, keep, strict=True) if m])
    import scipy.sparse as sp

    member = sp.csr_matrix((np.ones(keep.sum()), (labels[keep], np.arange(keep.sum()))),
                           shape=(k, keep.sum()))
    per_cluster = np.asarray((member @ counts).todense(), dtype=np.float64)
    tf = per_cluster / np.maximum(per_cluster.sum(axis=1, keepdims=True), 1)
    idf = np.log(1 + per_cluster.sum(axis=1).mean() / np.maximum(per_cluster.sum(axis=0), 1))
    score = tf * idf
    vocab = cv.get_feature_names_out()
    return [[vocab[j] for j in np.argsort(-row)[:n] if row[j] > 0] for row in score]


def _texts(con: duckdb.DuckDBPyConnection, narratives: Path, ids: np.ndarray) -> dict[str, str]:
    con.register("wanted", pa.table({"narrative_id": ids}))
    rows = con.execute(f"""
        SELECT narrative_id, any_value(description) FROM '{narratives.as_posix()}'
        SEMI JOIN wanted USING (narrative_id) GROUP BY narrative_id""").fetchall()
    con.unregister("wanted")
    return dict(rows)


def fit(clean: Path, ids: np.ndarray, vectors: np.ndarray, group: np.ndarray,
        means: np.ndarray, sample_size: int = SAMPLE_SIZE,
        min_cluster_size: int = MIN_CLUSTER_SIZE, reducer=reduce) -> pa.Table:
    """Find clusters on a sample; return the clusters table (without sizes)."""
    rng = np.random.default_rng(SEED)
    pick = np.sort(rng.choice(len(ids), min(sample_size, len(ids)), replace=False))
    x = adjust(vectors[pick], group[pick], means)
    t = time.perf_counter()
    points = reducer(x)
    print(f"[cluster] UMAP on {len(pick):,} sampled narratives: {time.perf_counter() - t:.0f} s",
          flush=True)
    labels = find_clusters(points, min(min_cluster_size, max(2, len(pick) // 10)))
    k = int(labels.max()) + 1
    if k == 0:
        raise SystemExit("[cluster] HDBSCAN found no clusters; lower MIN_CLUSTER_SIZE")
    print(f"[cluster] HDBSCAN: {k} clusters, {np.mean(labels < 0):.0%} of the sample is noise")

    centroids = centroids_of(x, labels)
    own = np.einsum("ij,ij->i", x[labels >= 0], centroids[labels[labels >= 0]])
    thresholds = np.array([np.quantile(own[labels[labels >= 0] == c], THRESHOLD_QUANTILE)
                           for c in range(k)], dtype=np.float32)

    con = duckdb.connect()
    texts = _texts(con, clean / "narratives.parquet", ids[pick])
    terms = distinctive_terms([texts[i] for i in ids[pick]], labels)
    fit_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return pa.table({
        "fit_id": [fit_id] * k,
        "cluster_id": pa.array(range(k), pa.int32()),
        "sample_members": pa.array(np.bincount(labels[labels >= 0], minlength=k), pa.int32()),
        "threshold": pa.array(thresholds),
        "centroid": pa.FixedSizeListArray.from_arrays(pa.array(centroids.reshape(-1)),
                                                      centroids.shape[1]),
        "terms": pa.array(terms, pa.list_(pa.string())),
    })


def summarize(con: duckdb.DuckDBPyConnection, clean: Path, out: Path, clusters: pa.Table,
              ids: np.ndarray, labels: np.ndarray,
              sims: np.ndarray) -> pa.Table:
    """Add full-data sizes, component groups and the narratives nearest each centroid."""
    k = clusters.num_rows
    nearest = []
    for c in range(k):
        members = np.flatnonzero(labels == c)
        nearest.append(list(ids[members[np.argsort(-sims[members])[:N_EXAMPLES]]]))
    con.register("cl", clusters.drop_columns(["centroid"]))
    stats = con.execute(f"""
        WITH a AS (SELECT * FROM '{(out / 'assignments.parquet').as_posix()}'),
        rows AS (
            SELECT a.cluster_id, c.odi_number, c.component_group
            FROM '{(clean / 'narratives.parquet').as_posix()}' n
            JOIN a USING (narrative_id)
            JOIN '{(clean / 'complaints.parquet').as_posix()}' c USING (complaint_id)
            WHERE a.cluster_id >= 0
        ),
        groups AS (
            SELECT cluster_id, component_group, count(DISTINCT odi_number) AS n
            FROM rows WHERE component_group IS NOT NULL GROUP BY ALL
        )
        SELECT cl.cluster_id,
               (SELECT count(*) FROM a WHERE a.cluster_id = cl.cluster_id) AS narratives,
               (SELECT count(DISTINCT odi_number) FROM rows r
                WHERE r.cluster_id = cl.cluster_id) AS complaints,
               (SELECT list(component_group ORDER BY n DESC, component_group)[1:5]
                FROM groups g WHERE g.cluster_id = cl.cluster_id) AS component_groups
        FROM cl ORDER BY cl.cluster_id
    """).to_arrow_table()
    con.unregister("cl")
    t = clusters
    for name in ("narratives", "complaints", "component_groups"):
        if name in t.column_names:
            t = t.drop_columns([name])
        t = t.append_column(name, stats.column(name))
    if "examples" in t.column_names:
        t = t.drop_columns(["examples"])
    return t.append_column("examples", pa.array(nearest, pa.list_(pa.string())))


def run(clean: Path = CLEAN, model_name: str = EMBED_MODEL, refit: bool = False,
        sample_size: int = SAMPLE_SIZE, min_cluster_size: int = MIN_CLUSTER_SIZE,
        min_group: int = MIN_GROUP, reducer=reduce) -> None:
    out = clean / "clusters"
    out.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    ids, vectors = load_vectors(clean, model_name)
    print(f"[cluster] {len(ids):,} narrative vectors loaded in {time.perf_counter() - t:.0f} s")
    group, means = vehicle_offsets(clean, ids, vectors, min_group)
    path = out / "clusters.parquet"
    if refit or not path.exists():
        clusters = fit(clean, ids, vectors, group, means, sample_size, min_cluster_size, reducer)
    else:
        clusters = pq.read_table(path)
        print(f"[cluster] keeping fit {clusters.column('fit_id')[0]} "
              f"({clusters.num_rows} clusters); `cluster --refit` to refit")

    cent = clusters.column("centroid").combine_chunks()
    centroids = cent.values.to_numpy(zero_copy_only=False).reshape(clusters.num_rows, -1)
    thresholds = clusters.column("threshold").to_numpy()
    labels, sims = assign(vectors, group, means, centroids.astype(np.float32), thresholds)
    pq.write_table(pa.table({"narrative_id": ids, "cluster_id": labels, "similarity": sims}),
                   out / "assignments.parquet", compression="zstd")

    raw = raw_centroids(vectors, labels, clusters.num_rows)
    if "raw_centroid" in clusters.column_names:
        clusters = clusters.drop_columns(["raw_centroid"])
    clusters = clusters.append_column(
        "raw_centroid", pa.FixedSizeListArray.from_arrays(pa.array(raw.reshape(-1)), raw.shape[1]))

    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")
    clusters = summarize(con, clean, out, clusters, ids, labels, sims)
    pq.write_table(clusters, path.with_suffix(".tmp"), compression="zstd")
    path.with_suffix(".tmp").rename(path)
    print(f"[cluster] {clusters.num_rows} clusters; {np.mean(labels >= 0):.0%} of "
          f"{len(ids):,} narratives assigned to one ({time.perf_counter() - t:.0f} s)")
