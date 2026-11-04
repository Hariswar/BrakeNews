"""Message Batches helper shared by the bulk labelling steps (recall_match.py, verify.py).

Labels are cached per item by the caller; this module only gets answers for the items that are
still missing. The batch id is saved the moment a batch is created, so a run that is
interrupted (or started with wait=False) collects that batch next time instead of paying for
it twice. Requests ask for JSON through structured outputs; an answer that did not finish
(errored, expired, cut off) is reported as missing and asked again on the next run.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from pipeline.config import ROOT

POLL_SECONDS = 30
MAX_REQUESTS = 50_000  # per batch; the API allows 100,000


def make_client():
    import anthropic  # pip install -e ".[llm]"
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    return anthropic.Anthropic()


def request(custom_id: str, model: str, system: str, text: str, schema: dict,
            max_tokens: int = 4000) -> dict:
    return {"custom_id": custom_id,
            "params": {"model": model, "max_tokens": max_tokens, "system": system,
                       "messages": [{"role": "user", "content": text}],
                       "output_config": {"effort": "low",
                                         "format": {"type": "json_schema", "schema": schema}}}}


def _results(client, batch_id: str, attempts: int = 4) -> tuple[dict[str, dict], int]:
    """Download a finished batch's results. Large result files are streamed and a dropped or
    timed-out connection raises mid-stream; the download is simply repeated."""
    for attempt in range(1, attempts + 1):
        try:
            return _read_results(client, batch_id)
        except Exception as e:  # httpx2 transport errors are not anthropic.APIError
            transient = any(w in type(e).__name__ for w in ("Timeout", "Connect", "Read"))
            if not transient or attempt == attempts:
                raise
            print(f"[batch] downloading results of {batch_id} failed ({type(e).__name__}); "
                  f"retry {attempt}/{attempts - 1}", flush=True)
            time.sleep(10 * attempt)
    raise AssertionError("unreachable")


def _read_results(client, batch_id: str) -> tuple[dict[str, dict], int]:
    out, failed = {}, 0
    for r in client.messages.batches.results(batch_id):
        msg = r.result.message if r.result.type == "succeeded" else None
        if msg is None or msg.stop_reason != "end_turn":
            failed += 1
            continue
        try:
            out[r.custom_id] = json.loads(next(b.text for b in msg.content if b.type == "text"))
        except (StopIteration, json.JSONDecodeError):
            failed += 1
    return out, failed


def collect_pending(client, state: Path, tag: str, wait: bool,
                    poll_seconds: float = POLL_SECONDS) -> dict[str, dict] | None:
    """Answers of the batch saved in `state`, {} if there is none, None if it is still
    running and wait is False."""
    if not state.exists():
        return {}
    batch_id = json.loads(state.read_text())["batch_id"]
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        if not wait:
            print(f"[{tag}] batch {batch_id} still {batch.processing_status}; rerun to collect")
            return None
        print(f"[{tag}] batch {batch_id}: {batch.request_counts.processing:,} still processing",
              flush=True)
        time.sleep(poll_seconds)
    answers, failed = _results(client, batch_id)
    state.unlink()
    print(f"[{tag}] batch {batch_id}: {len(answers):,} answers, {failed:,} failed")
    return answers


def submit(client, requests: list[dict], state: Path, tag: str) -> None:
    if not requests:
        return
    requests = requests[:MAX_REQUESTS]  # the rest go in the next run's batch
    batch_id = client.messages.batches.create(requests=requests).id
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({"batch_id": batch_id, "requests": len(requests)}))
    print(f"[{tag}] submitted batch {batch_id}: {len(requests):,} requests")


def label(client, state: Path, tag: str, build: Callable[[], list[dict]],
          store: Callable[[dict[str, dict]], None], wait: bool = True,
          poll_seconds: float = POLL_SECONDS) -> bool:
    """Collect a pending batch into `store`, then submit `build()`'s requests (built after
    storing, so they cover only what is still missing) and, if waiting, collect those too.
    True when nothing is left to ask; failed items are asked again on the next run."""
    answers = collect_pending(client, state, tag, wait, poll_seconds)  # an earlier run's batch
    if answers is None:
        return False
    if answers:
        store(answers)
    todo = build()
    if not todo:
        return True
    submit(client, todo, state, tag)
    if not wait:
        return False
    answers = collect_pending(client, state, tag, wait, poll_seconds)
    if answers:
        store(answers)
    left = len(build())
    if left:
        print(f"[{tag}] {left:,} requests still unanswered; rerun to retry them")
    return not left


def label_or_warn(client, state: Path, tag: str, build: Callable[[], list[dict]],
                  store: Callable[[dict[str, dict]], None], wait: bool = True,
                  poll_seconds: float = POLL_SECONDS) -> bool:
    """label(), but an API refusal (no credit, bad key, network down) is reported instead of
    raised: what was already answered is kept and the caller carries on with gaps."""
    import anthropic

    try:
        return label(client, state, tag, build, store, wait, poll_seconds)
    except anthropic.APIError as e:
        print(f"[{tag}] the API refused ({type(e).__name__}: {getattr(e, 'message', e)}); "
              f"continuing with the answers so far, rerun to fill the gaps")
        return False
