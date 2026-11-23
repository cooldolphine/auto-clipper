#!/usr/bin/env python3
"""discover -> clip each new video -> remember what's done -> package a release folder.

Settings come from config.json, and can be overridden per run by the phone app
(through the workflow inputs, passed here as RUN_* environment variables).
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

import progress
from discover import discover


def words(name):
    """Space/comma separated list from an env var."""
    return [x for x in re.split(r"[,\s]+", os.environ.get(name, "")) if x]


def vid_id(url):
    m = re.search(r"(?:v=|youtu\.be/|/live/|/shorts/|/embed/)([\w-]{11})", url)
    return m.group(1) if m else hashlib.sha1(url.encode()).hexdigest()[:11]  # Twitch, Kick, direct files...


PROGRESS = re.compile(r"^\[download\]\s+[\d.]+% of")


def run_clipper(cmd, env=None):
    """Run clipper.py, hide the thousands of download-progress lines, remember the errors."""
    errors = deque(maxlen=3)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    for line in proc.stdout:
        if PROGRESS.match(line) and "100% of" not in line:
            continue
        print(line, end="", flush=True)
        if "ERROR" in line or "Error" in line:
            errors.append(line.strip())
    proc.wait()
    return proc.returncode, list(errors)


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
if os.environ.get("RUN_HEIGHT"):
    cfg["max_height"] = int(os.environ["RUN_HEIGHT"])
if os.environ.get("RUN_VIDEOS"):
    cfg["max_videos_per_run"] = int(os.environ["RUN_VIDEOS"])
if os.environ.get("RUN_MODEL"):
    cfg["whisper_model"] = os.environ["RUN_MODEL"]
for env_name, key in (("RUN_FRAMING", "framing"), ("RUN_CAPTIONS", "captions"), ("RUN_HOOK", "hook"),
                      ("RUN_AI", "ai"), ("RUN_BROLL", "broll"), ("RUN_TRANSITIONS", "transitions"),
                      ("RUN_DIRECTOR", "director"), ("RUN_PUNCH", "punch"), ("RUN_CUT_SILENCE", "cut_silence"), ("RUN_EMOJI", "emoji")):
    if os.environ.get(env_name):
        cfg[key] = os.environ[env_name]
if os.environ.get("RUN_LANGUAGE"):
    cfg["language"] = None if os.environ["RUN_LANGUAGE"] == "auto" else os.environ["RUN_LANGUAGE"]

pfile, index_file = Path("processed.json"), Path("output/index.json")
processed = json.loads(pfile.read_text()) if pfile.exists() else []
index = json.loads(index_file.read_text()) if index_file.exists() else []
Path("output").mkdir(exist_ok=True)

cookie_file = "cookies.txt" if Path("cookies.txt").exists() else None
progress.set_slice(0, 5)
progress.report(0.0, "Finding videos", force=True)
if urls:  # explicit links: clip exactly these, even if clipped before
    videos = [{"id": vid_id(u), "url": u, "title": u} for u in urls]
else:
    videos = discover(cfg, processed, cookie_file)
print(f"\n{len(videos)} video(s) to clip")
cookies = ["--cookies", "cookies.txt"] if cookie_file else []

run_clips, failures = [], []
for i, v in enumerate(videos):
    span = 94 / len(videos)
    progress.set_slice(5 + span * i, span)
    progress.report(0.0, f"Video {i + 1} of {len(videos)}: starting", force=True)
    print(f"\n=== {v['title']} ({v['url']})")
    out = Path("output") / v["id"]
    code, errs = run_clipper([
        sys.executable, "clipper.py", v["url"], "--out", str(out),
        "--num", str(cfg["clips_per_video"]), "--length", str(cfg["clip_length_seconds"]),
        "--model", cfg.get("whisper_model", "base"),
        "--height", str(cfg.get("max_height", 1080)),
        "--framing", cfg.get("framing", "stack"),
        "--captions", cfg.get("captions", "karaoke"),
        "--caption-margin", str(cfg.get("caption_margin", 230)),
        "--hook", cfg.get("hook", "auto"),
        "--ai", cfg.get("ai", "on"),
        "--director", cfg.get("director", "on"),
        "--punch", cfg.get("punch", "on"),
        "--cut-silence", cfg.get("cut_silence", "on"),
        "--emoji", cfg.get("emoji", "on"),
        "--broll", cfg.get("broll", "off"),
        "--transitions", cfg.get("transitions", "off"),
        *(["--language", cfg["language"]] if cfg.get("language") else []), *cookies,
    ], env={**os.environ, "PROGRESS_START": str(5 + span * i), "PROGRESS_SPAN": str(span)})
    if v["id"] not in processed:
        processed.append(v["id"])  # mark even on failure so we don't loop forever
    if code == 0 and (out / "clips.json").exists():
        for clip in json.loads((out / "clips.json").read_text()):
            clip["file"] = f"{v['id']}/{clip['file']}"
            index.append(clip)
            run_clips.append(clip)
    else:
        print("  clipper failed, skipping this video")
        failures.append((v["title"] or v["url"], errs[-1] if errs else "see the log above"))
    pfile.write_text(json.dumps(processed, indent=1))
    index_file.write_text(json.dumps(index, indent=2))

progress.set_slice(0, 100)
progress.finish(bool(run_clips), "Clips ready" if run_clips else "No clips were made")

if failures:
    print("\n==== WHY CLIPS WERE NOT MADE ====")
    for title, why in failures:
        print(f"- {title}\n    {why}")
    print("If you see 'Sign in to confirm you're not a bot', refresh the YT_COOKIES secret.")

# ---- flat folder for the GitHub Release (what the phone app reads) ----
rel = Path("release")
shutil.rmtree(rel, ignore_errors=True)
rel.mkdir()
items = []
for c in run_clips:
    flat = c["file"].replace("/", "_")
    shutil.copy2(Path("output") / c["file"], rel / flat)
    items.append({**c, "file": flat, "topics": [] if urls else (cfg.get("topics") or cfg.get("queries") or [])})
(rel / "index.json").write_text(json.dumps(items, indent=1))

# remove big temp files (full video + wav) to save disk
for w in Path("output").glob("*/work"):
    for f in w.glob("*"):
        f.unlink()

# make the run show as FAILED (not "Done") when nothing was clipped
if not run_clips:
    print("\nNo clips were made.")
    sys.exit(1)
