#!/usr/bin/env python3
"""discover -> clip each new video -> remember what's done -> package a release folder.

Settings come from config.json, and can be overridden per run by the phone app
(through the workflow inputs, passed here as RUN_* environment variables).
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from discover import discover


def words(name):
    """Space/comma separated list from an env var."""
    return [x for x in re.split(r"[,\s]+", os.environ.get(name, "")) if x]


def vid_id(url):
    m = re.search(r"(?:v=|youtu\.be/|/live/|/shorts/|/embed/)([\w-]{11})", url)
    return m.group(1) if m else re.sub(r"\W", "", url)[-11:]


cfg = json.loads(Path("config.json").read_text())

# ---- overrides from the phone app ----
urls = words("RUN_URLS")
topics = [t.strip() for t in re.split(r"[,\n]", os.environ.get("RUN_TOPICS", "")) if t.strip()]
channels = words("RUN_CHANNELS")
if channels == ["none"]:
    cfg["channels"], cfg["topics"] = [], []
    cfg["queries"] = topics  # no channels given: search all of YouTube for the topics
elif channels:
    cfg["channels"] = channels
if topics and channels != ["none"]:
    cfg["topics"], cfg["queries"] = topics, []
if os.environ.get("RUN_NUM"):
    cfg["clips_per_video"] = int(os.environ["RUN_NUM"])
if os.environ.get("RUN_LENGTH"):
    cfg["clip_length_seconds"] = int(os.environ["RUN_LENGTH"])
if os.environ.get("RUN_VIDEOS"):
    cfg["max_videos_per_run"] = int(os.environ["RUN_VIDEOS"])
if os.environ.get("RUN_MODEL"):
    cfg["whisper_model"] = os.environ["RUN_MODEL"]
if os.environ.get("RUN_LANGUAGE"):
    cfg["language"] = None if os.environ["RUN_LANGUAGE"] == "auto" else os.environ["RUN_LANGUAGE"]

pfile, index_file = Path("processed.json"), Path("output/index.json")
processed = json.loads(pfile.read_text()) if pfile.exists() else []
index = json.loads(index_file.read_text()) if index_file.exists() else []
Path("output").mkdir(exist_ok=True)

cookie_file = "cookies.txt" if Path("cookies.txt").exists() else None
if urls:  # explicit links: clip exactly these, even if clipped before
    videos = [{"id": vid_id(u), "url": u, "title": u} for u in urls]
else:
    videos = discover(cfg, processed, cookie_file)
print(f"\n{len(videos)} video(s) to clip")
cookies = ["--cookies", "cookies.txt"] if cookie_file else []

run_clips = []
for v in videos:
    print(f"\n=== {v['title']} ({v['url']})")
    out = Path("output") / v["id"]
    r = subprocess.run([
        sys.executable, "clipper.py", v["url"], "--out", str(out),
        "--num", str(cfg["clips_per_video"]), "--length", str(cfg["clip_length_seconds"]),
        "--model", cfg.get("whisper_model", "base"),
        "--height", str(cfg.get("max_height", 720)),
        *(["--language", cfg["language"]] if cfg.get("language") else []), *cookies,
    ])
    if v["id"] not in processed:
        processed.append(v["id"])  # mark even on failure so we don't loop forever
    if r.returncode == 0 and (out / "clips.json").exists():
        for clip in json.loads((out / "clips.json").read_text()):
            clip["file"] = f"{v['id']}/{clip['file']}"
            index.append(clip)
            run_clips.append(clip)
    else:
        print("  clipper failed, skipping this video")
    pfile.write_text(json.dumps(processed, indent=1))
    index_file.write_text(json.dumps(index, indent=2))

if videos and not run_clips:
    print("\nNo clips were made. If you saw 'Sign in to confirm you're not a bot', YouTube is "
          "blocking this server: refresh the YT_COOKIES secret.")

# ---- flat folder for the GitHub Release (what the phone app reads) ----
rel = Path("release")
shutil.rmtree(rel, ignore_errors=True)
rel.mkdir()
items = []
for c in run_clips:
    flat = c["file"].replace("/", "_")
    shutil.copy2(Path("output") / c["file"], rel / flat)
    items.append({**c, "file": flat})
(rel / "index.json").write_text(json.dumps(items, indent=1))

# remove big temp files (full video + wav) to save disk
for w in Path("output").glob("*/work"):
    for f in w.glob("*"):
        f.unlink()
