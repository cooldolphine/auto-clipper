"""Stock footage for B-roll cutaways.

Uses Pixabay (free key in the PIXABAY_API_KEY secret) or Pexels (PEXELS_API_KEY, if you
already have a key; Pexels has paused new keys). Both are free for commercial use.
"""
import os

import requests


def _download(url, dest):
    with requests.get(url, stream=True, timeout=120) as d:
        d.raise_for_status()
        with open(dest, "wb") as out:
            for chunk in d.iter_content(1 << 20):
                out.write(chunk)
    return dest


def _pixabay(query, dest, seconds, key):
    r = requests.get("https://pixabay.com/api/videos/", params={"key": key, "q": query, "per_page": 10, "safesearch": "true"}, timeout=30)
    if r.status_code != 200:
        return None
    for hit in r.json().get("hits", []):
        if (hit.get("duration") or 0) < seconds:
            continue
        for size in ("large", "medium"):
            v = (hit.get("videos") or {}).get(size) or {}
            if v.get("url") and (v.get("width") or 0) >= 720:
                return _download(v["url"], dest)
    return None


def _pexels(query, dest, seconds, key):
    r = requests.get("https://api.pexels.com/videos/search", headers={"Authorization": key},
                     params={"query": query, "per_page": 8}, timeout=30)
    if r.status_code != 200:
        return None
    for v in r.json().get("videos", []):
        if (v.get("duration") or 0) < seconds:
            continue
        files = [f for f in v.get("video_files", []) if f.get("file_type") == "video/mp4" and (f.get("width") or 0) >= 720]
        if files:
            f = min(files, key=lambda f: abs((f.get("height") or 0) - 1080))
            return _download(f["link"], dest)
    return None


def fetch(query, dest, seconds=3):
    """Download one short stock video for `query` to `dest`. Returns the path or None."""
    try:
        if os.environ.get("PIXABAY_API_KEY"):
            got = _pixabay(query, dest, seconds, os.environ["PIXABAY_API_KEY"])
            if got:
                return got
        if os.environ.get("PEXELS_API_KEY"):
            return _pexels(query, dest, seconds, os.environ["PEXELS_API_KEY"])
    except Exception as e:
        print(f"    B-roll download failed for '{query}': {e}")
    return None
