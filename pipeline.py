#!/usr/bin/env python3
"""discover -> clip each new video -> remember what's done.

    python pipeline.py
Output: output/<video_id>/clip_XX.mp4 + clips.json, and output/index.json
(a review queue: set "approved": true on clips you want posted).
"""
import json
import subprocess
import sys
from pathlib import Path

from discover import discover

cfg = json.loads(Path("config.json").read_text())
pfile, index_file = Path("processed.json"), Path("output/index.json")
processed = json.loads(pfile.read_text()) if pfile.exists() else []
index = json.loads(index_file.read_text()) if index_file.exists() else []
Path("output").mkdir(exist_ok=True)

cookie_file = "cookies.txt" if Path("cookies.txt").exists() else None
videos = discover(cfg, processed, cookie_file)
print(f"\n{len(videos)} new video(s) to clip")
cookies = ["--cookies", "cookies.txt"] if Path("cookies.txt").exists() else []

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
    processed.append(v["id"])  # mark even on failure so we don't loop forever
    if r.returncode == 0 and (out / "clips.json").exists():
        for clip in json.loads((out / "clips.json").read_text()):
            clip["file"] = f"{v['id']}/{clip['file']}"
            index.append(clip)
    else:
        print("  clipper failed, skipping this video")
    pfile.write_text(json.dumps(processed, indent=1))
    index_file.write_text(json.dumps(index, indent=2))

if videos and not any((Path("output") / v["id"] / "clips.json").exists() for v in videos):
    print("\nAll downloads failed. If you saw 'Sign in to confirm you're not a bot', YouTube is "
          "blocking this server: run on your own PC, or add cookies.txt (see README).")

# always write state files, even when nothing was found
pfile.write_text(json.dumps(processed, indent=1))
index_file.write_text(json.dumps(index, indent=2))

# remove big temp files (full video + wav) to save disk
for w in Path("output").glob("*/work"):
    for f in w.glob("*"):
        f.unlink()
