"""BrakeNews command line: `python -m pipeline <step>` or `brakenews <step>`."""

from __future__ import annotations

import argparse

from pipeline import clean, cluster, embed, ingest, names, scrub, signals, spotcheck
from pipeline.config import EMBED_MODEL, NAMING_MODEL


def main() -> None:
    p = argparse.ArgumentParser(prog="brakenews")
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ingest", help="download NHTSA files and convert to Parquet")
    i.add_argument("--force", action="store_true", help="re-download even if files exist")
    sub.add_parser("clean", help="build typed tables in data/clean")
    sub.add_parser("scrub", help="mask personal details in complaint narratives")
    b = sub.add_parser("build", help="ingest + clean + scrub (one-command rebuild)")
    b.add_argument("--force", action="store_true")
    s = sub.add_parser("spotcheck", help="compare sampled clean records with raw")
    s.add_argument("-n", type=int, default=20)
    e = sub.add_parser("embed", help="embed scrubbed narratives (incremental, resumable)")
    e.add_argument("--model", default=EMBED_MODEL)
    e.add_argument("--limit", type=int, help="embed at most this many new narratives")
    c = sub.add_parser("cluster", help="group narratives into themes (assign-only unless --refit)")
    c.add_argument("--model", default=EMBED_MODEL, help="embedding model whose vectors to use")
    c.add_argument("--refit", action="store_true", help="find new clusters (invalidates names)")
    nm = sub.add_parser("name", help="name clusters with Claude via the Batch API (resumable)")
    nm.add_argument("--model", default=NAMING_MODEL)
    nm.add_argument("--no-wait", action="store_true", help="submit or check, then exit")
    sg = sub.add_parser("signals", help="vehicle x theme early-warning signals and backtest")
    sg.add_argument("--model", default=EMBED_MODEL, help="embedding model for recall summaries")
    args = p.parse_args()

    if args.cmd == "ingest":
        ingest.run(force=args.force)
    elif args.cmd == "clean":
        clean.run()
    elif args.cmd == "scrub":
        scrub.run()
    elif args.cmd == "build":
        ingest.run(force=args.force)
        clean.run()
        scrub.run()
    elif args.cmd == "spotcheck":
        spotcheck.run(n=args.n)
    elif args.cmd == "embed":
        embed.run(model_name=args.model, limit=args.limit)
    elif args.cmd == "cluster":
        cluster.run(model_name=args.model, refit=args.refit)
    elif args.cmd == "name":
        names.run(model=args.model, wait=not args.no_wait)
    elif args.cmd == "signals":
        signals.run(model_name=args.model)


if __name__ == "__main__":
    main()
