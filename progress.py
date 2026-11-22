"""Live progress for the phone app.

Posts a GitHub commit status named progress-<run id> whose description is
"<percent>|<text>". The app reads it while a run is going. Does nothing (and never
fails) when not running on GitHub Actions.
"""
import os
import time

import requests

_state = {"last": 0.0, "text": "", "pct": -1}
SLICE = {"start": float(os.environ.get("PROGRESS_START", 0) or 0), "span": float(os.environ.get("PROGRESS_SPAN", 100) or 100)}


def set_slice(start, span):
    SLICE["start"], SLICE["span"] = float(start), float(span)


def _post(pct, text, state="pending"):
    token, repo, sha, run = (os.environ.get(k) for k in ("GITHUB_TOKEN", "GITHUB_REPOSITORY", "GITHUB_SHA", "GITHUB_RUN_ID"))
    if not (token and repo and sha and run):
        return
    try:
        requests.post(f"https://api.github.com/repos/{repo}/statuses/{sha}",
                      headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
                      json={"state": state, "context": f"progress-{run}", "description": f"{int(pct)}|{text}"[:138]},
                      timeout=10)
    except Exception:
        pass


def report(frac, text, force=False):
    """frac is 0..1 inside the current slice of the whole run. Sends at most one update every 4 seconds."""
    pct = SLICE["start"] + SLICE["span"] * max(0.0, min(1.0, frac))
    now = time.time()
    if not force and now - _state["last"] < 4:
        return
    _state.update(last=now, text=text, pct=pct)
    if os.environ.get("PROGRESS_PRINT"):
        print(f"  [{int(pct)}%] {text}", flush=True)
    _post(pct, text)


def stage(lo, hi):
    """Returns a function f(frac, text) that reports inside [lo, hi] of the current slice."""
    return lambda frac, text: report(lo + (hi - lo) * max(0.0, min(1.0, frac)), text)


def finish(ok, text=None):
    _post(100 if ok else _state["pct"] if _state["pct"] >= 0 else 0, text or ("Done" if ok else "Failed"),
          "success" if ok else "failure")
