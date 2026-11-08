"""BrakeNews command line: `python -m pipeline <step>` or `brakenews <step>`."""

from __future__ import annotations

import argparse
import sys

from pipeline import (
    clean,
    cluster,
    embed,
    export,
    features,
    ingest,
    investigate,
    model,
    names,
    recall_match,
    scrub,
    signals,
    spotcheck,
    verify,
)
from pipeline.config import EMBED_MODEL, INVESTIGATE_MODEL, NAMING_MODEL, REVIEW_MODEL


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up even when redirected
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
    mt = sub.add_parser("match", help="Claude checks which themes each recall covers (Batch API)")
    mt.add_argument("--model", default=EMBED_MODEL, help="embedding model for recall summaries")
    mt.add_argument("--llm", default=NAMING_MODEL)
    mt.add_argument("--no-wait", action="store_true", help="submit or check, then exit")
    sg = sub.add_parser("signals", help="vehicle x theme early-warning signals and backtest")
    sg.add_argument("--model", default=EMBED_MODEL, help="embedding model for recall summaries")
    sub.add_parser("features", help="training table for the recall-risk model")
    sub.add_parser("select", help="compare model feature and parameter sets on earlier folds")
    sub.add_parser("model", help="train, evaluate and score the recall-risk model")
    vf = sub.add_parser("verify", help="Claude checks the complaints behind current alerts")
    vf.add_argument("--llm", default=NAMING_MODEL)
    vf.add_argument("--no-wait", action="store_true", help="submit or check, then exit")
    iv = sub.add_parser("investigate",
                        help="Claude agents investigate and fact-check signals (costs money)")
    iv.add_argument("--top", type=int, default=1,
                    help="the N highest-risk confirmed alerts")
    iv.add_argument("--vehicle", help='one pair instead: "MAKE/MODEL/YEAR" with --theme')
    iv.add_argument("--theme", type=int)
    iv.add_argument("--llm", default=INVESTIGATE_MODEL, help="investigator's Claude model")
    iv.add_argument("--review-llm", default=REVIEW_MODEL, help="reviewer's Claude model")
    iv.add_argument("--no-review", action="store_true", help="skip the fact-checking reviewer")
    iv.add_argument("--workers", type=int, default=investigate.WORKERS,
                    help="pairs investigated at once")
    iv.add_argument("--budget", type=float, default=5.0,
                    help="start no new pair once this run has spent this many dollars")
    iv.add_argument("--force", action="store_true", help="rerun even if the evidence is unchanged")
    sub.add_parser("export", help="publish approved reports on current alerts for the site")
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
    elif args.cmd == "match":
        recall_match.run(model_name=args.model, model=args.llm, wait=not args.no_wait)
    elif args.cmd == "signals":
        signals.run(model_name=args.model)
    elif args.cmd == "features":
        features.run()
    elif args.cmd == "select":
        model.select()
    elif args.cmd == "model":
        model.run()
    elif args.cmd == "verify":
        verify.run(model=args.llm, wait=not args.no_wait)
    elif args.cmd == "investigate":
        pair = None
        if args.vehicle:
            # Make first, year last; the model in between may itself contain a slash.
            make, _, rest = args.vehicle.partition("/")
            vmodel, _, year = rest.rpartition("/")
            if not (make and vmodel and year.isdigit()) or args.theme is None:
                p.error('--vehicle takes "MAKE/MODEL/YEAR" and needs --theme')
            pair = investigate.Pair(make.upper(), vmodel.upper(), int(year), args.theme)
        results = investigate.run(
            top=args.top, pair=pair, model=args.llm, force=args.force,
            review_model=None if args.no_review else args.review_llm, workers=args.workers,
            budget=args.budget)
        if any("error" in r for r in results):
            raise SystemExit(1)
    elif args.cmd == "export":
        export.run()


if __name__ == "__main__":
    main()
