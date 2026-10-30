"""BrakeNews command line: `python -m pipeline <step>` or `brakenews <step>`."""

from __future__ import annotations

import argparse

from pipeline import clean, ingest, spotcheck


def main() -> None:
    p = argparse.ArgumentParser(prog="brakenews")
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ingest", help="download NHTSA files and convert to Parquet")
    i.add_argument("--force", action="store_true", help="re-download even if files exist")
    sub.add_parser("clean", help="build typed tables in data/clean")
    b = sub.add_parser("build", help="ingest + clean (one-command rebuild)")
    b.add_argument("--force", action="store_true")
    s = sub.add_parser("spotcheck", help="compare sampled clean records with raw")
    s.add_argument("-n", type=int, default=20)
    args = p.parse_args()

    if args.cmd == "ingest":
        ingest.run(force=args.force)
    elif args.cmd == "clean":
        clean.run()
    elif args.cmd == "build":
        ingest.run(force=args.force)
        clean.run()
    elif args.cmd == "spotcheck":
        spotcheck.run(n=args.n)


if __name__ == "__main__":
    main()
