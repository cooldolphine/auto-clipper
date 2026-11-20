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

FLAT = {"quiet": True, "extract_flat": True, "skip_download": True}
FULL = {"quiet": True, "skip_download": True, "noplaylist": True}


def _flat_ids(source, n):
    """Return list of video ids from a search query or channel URL."""
    if source.startswith("http"):
        target, opts = source, {**FLAT, "playlistend": n}
    else:
        target, opts = f"ytsearchdate{n}:{source}", FLAT
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(target, download=False)
    except Exception as e:  # keep going if one source fails
        print(f"  ! source failed ({source}): {e}")
        return []
    ids = []
    for e in info.get("entries") or []:
        if e and e.get("id") and len(e["id"]) == 11:
            ids.append(e["id"])
        elif e and e.get("entries"):  # channel tabs nest entries
            ids += [x["id"] for x in e["entries"] if x and x.get("id")]
    return ids[:n]


def _details(video_id):
    try:
        with yt_dlp.YoutubeDL(FULL) as ydl:
            return ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    except Exception:
        return None


def discover(cfg, processed):
    cutoff = datetime.utcnow() - timedelta(days=cfg["max_age_days"])
    candidates, seen = [], set(processed)
    for src in cfg.get("queries", []) + cfg.get("channels", []):
        print(f"Searching: {src}")
        for vid in _flat_ids(src, cfg["results_per_source"]):
            if vid in seen:
                continue
            seen.add(vid)
            info = _details(vid)
            if not info:
                continue
            status = info.get("live_status")
            if status in ("is_live", "is_upcoming"):
                print(f"  live/upcoming, skipping for now: {info.get('title')}")
                continue  # clip it after it ends (next run picks the VOD)
            if (info.get("duration") or 0) < cfg["min_duration_minutes"] * 60:
                continue
            if (info.get("view_count") or 0) < cfg.get("min_views", 0):
                continue
            up = info.get("upload_date")
            if up and datetime.strptime(up, "%Y%m%d") < cutoff:
                continue
            candidates.append({
                "id": vid, "url": info["webpage_url"], "title": info.get("title"),
                "channel": info.get("uploader"), "views": info.get("view_count") or 0,
                "duration": info.get("duration"),
            })
    candidates.sort(key=lambda c: c["views"], reverse=True)
    return candidates[: cfg["max_videos_per_run"]]


if __name__ == "__main__":
    cfg = json.loads(Path("config.json").read_text())
    done = json.loads(Path("processed.json").read_text()) if Path("processed.json").exists() else []
    for c in discover(cfg, done):
        print(f"{c['views']:>9} views | {c['duration']//60:>4} min | {c['channel']} | {c['title']}\n          {c['url']}")
