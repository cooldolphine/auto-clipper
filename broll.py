"""Stock footage from Pexels (free key in the PEXELS_API_KEY secret)."""
import os

import requests


def fetch(query, dest, seconds=3, key=None):
    """Download one short stock video for `query` to `dest`. Returns the path or None."""
    key = key or os.environ.get("PEXELS_API_KEY")
    if not key:
        return None
    try:
        r = requests.get("https://api.pexels.com/videos/search", headers={"Authorization": key},
                         params={"query": query, "per_page": 8}, timeout=30)
        if r.status_code != 200:
            return None
        for v in r.json().get("videos", []):
            if (v.get("duration") or 0) < seconds:
                continue
            files = [f for f in v.get("video_files", []) if f.get("file_type") == "video/mp4" and (f.get("width") or 0) >= 720]
            if not files:
                continue
            f = min(files, key=lambda f: abs((f.get("height") or 0) - 1080))
            with requests.get(f["link"], stream=True, timeout=120) as d:
                d.raise_for_status()
                with open(dest, "wb") as out:
                    for chunk in d.iter_content(1 << 20):
                        out.write(chunk)
            return dest
    except Exception as e:
        print(f"    B-roll download failed for '{query}': {e}")
    return None
