#!/usr/bin/env python3
"""Find long videos / finished live streams worth clipping.

Sources (from config.json): search queries + specific channels.
No API key needed (uses yt-dlp). Run alone to preview what it would pick:
    python discover.py
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

import yt_dlp


def _opts(cookies, **extra):
    o = {"quiet": True, "no_warnings": True, "skip_download": True, **extra}
    if cookies:
        o["cookiefile"] = cookies
    return o


def _extract(target, opts):
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(target, download=False)


def _flat_entries(source, n, cookies):
    """Return flat entries (id, duration, views...) for a search query or channel URL."""
    info, last_err = None, None
    if source.startswith("http"):
        attempts = [(source, _opts(cookies, extract_flat=True, playlistend=n))]
    else:
        flat = _opts(cookies, extract_flat=True)
        attempts = [(f"ytsearchdate{n * 2}:{source}", flat), (f"ytsearch{n * 2}:{source}", flat)]
    for target, opts in attempts:
        try:
            info = _extract(target, opts)
            if info and info.get("entries"):
                break
        except Exception as e:
            last_err = e
    if not info or not info.get("entries"):
        print(f"  ! no results for ({source}): {last_err or 'empty'}")
        return []
    out = []
    for e in info["entries"]:
        if not e:
            continue
        if e.get("entries"):  # channel tabs nest entries
            out += [x for x in e["entries"] if x and x.get("id")]
        elif e.get("id") and len(e["id"]) == 11:
            out.append(e)
    return out[: n * 2]


def _details(video_id, cookies):
    try:
        return _extract(f"https://www.youtube.com/watch?v={video_id}", _opts(cookies, noplaylist=True))
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ("unavailable", "live event", "private", "members-only", "premiere", "removed")):
            return "SKIP"  # dead/not-ready video: don't try to clip it
        return None


def discover(cfg, processed, cookies=None):
    cutoff = datetime.utcnow() - timedelta(days=cfg["max_age_days"])
    min_dur = cfg["min_duration_minutes"] * 60
    max_dur = cfg.get("max_duration_minutes", 180) * 60
    candidates, seen = [], set(processed)
    for src in cfg.get("queries", []) + cfg.get("channels", []):
        print(f"Searching: {src}")
        for e in _flat_entries(src, cfg["results_per_source"], cookies):
            vid = e["id"]
            if vid in seen:
                continue
            seen.add(vid)
            # cheap pre-filter using the search result itself (no extra request)
            if e.get("live_status") in ("is_live", "is_upcoming"):
                continue
            if e.get("duration") and not (min_dur <= e["duration"] <= max_dur):
                continue
            if e.get("view_count") is not None and e["view_count"] < cfg.get("min_views", 0):
                continue
            # accurate check; if YouTube blocks it, fall back to the search data
            info = _details(vid, cookies)
            if info == "SKIP":
                continue
            info = info or e
            if info.get("live_status") in ("is_live", "is_upcoming"):
                continue
            if not (min_dur <= (info.get("duration") or 0) <= max_dur):
                continue
            if (info.get("view_count") or 0) < cfg.get("min_views", 0):
                continue
            up = info.get("upload_date")
            if up and datetime.strptime(up, "%Y%m%d") < cutoff:
                continue
            candidates.append({
                "id": vid, "url": f"https://www.youtube.com/watch?v={vid}",
                "title": info.get("title"), "channel": info.get("uploader") or info.get("channel"),
                "views": info.get("view_count") or 0, "duration": info.get("duration") or 0,
            })
    candidates.sort(key=lambda c: c["views"], reverse=True)
    return candidates[: cfg["max_videos_per_run"]]


if __name__ == "__main__":
    cfg = json.loads(Path("config.json").read_text())
    done = json.loads(Path("processed.json").read_text()) if Path("processed.json").exists() else []
    ck = "cookies.txt" if Path("cookies.txt").exists() else None
    for c in discover(cfg, done, ck):
        print(f"{c['views']:>9} views | {c['duration']//60:>4} min | {c['channel']} | {c['title']}\n          {c['url']}")
